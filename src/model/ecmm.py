import sys, os
import numpy as np
import tensorflow as tf
from base_tools import BaseTools
from ecmm_layer import BlockLayer, AvgTransformLayer, TransformLayer, ECMMTransformLayer, L0Regularizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from data_loader import DataLoader


def ecmm_loss(transform_layers):
    reg_sum = 0
    for layer in transform_layers:
        reg_sum += tf.reduce_sum()

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.etr_losses = []
        self.ctr_losses = []
        super(LossHistory, self).__init__()

    def on_train_begin(self, logs={}):
        """
        self.losses = []
        self.etr_losses = []
        self.ctr_losses = []
        """
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.etr_losses.append(logs.get('etr_loss'))
        self.ctr_losses.append(logs.get('ctr_loss'))


class SVHistory(tf.keras.callbacks.Callback):
    def __init__(self,
                 block_num,
                 task_num,
                 trans_type,
                 **kwargs):
        self.block_num = block_num
        self.task_num = task_num
        self.trans_type = trans_type
        self.beta = kwargs['beta']
        self.zeta = kwargs['zeta']
        self.gamma = kwargs['gamma']
        self.s_v = {}
        self.s_bar_v = {}
        self.alpha_v = {}
        self.u_v = {}
        self.z_v = {}
        self.tower_output = {}
        self.train_z = {}
        self.test_z = {}
        super(SVHistory, self).__init__()

    def on_train_batch_begin(self, batch, logs={}):
        layer_num = [_ for _ in self.block_num] + [self.task_num]
        for idx in range(len(layer_num) - 1):
            b_in, b_next_in = layer_num[idx], layer_num[idx + 1]
            layer_name = 'avg_transform_layer_%d' % idx if self.trans_type == "aver" else 'transform_layer_%d' % idx
            layer = self.model.get_layer(layer_name)
            tf.compat.v1.assign(layer.u,
                                tf.random.uniform(layer.u_shape, minval=1e-5, maxval=1.0 - 1e-5))

    def get_z(self):
        for idx in range(len(self.block_num)):
            layer_name = 'avg_transform_layer_%d' % idx if self.trans_type == "aver" else 'transform_layer_%d' % idx
            layer_obj = self.model.get_layer(layer_name)
            u, log_alpha = layer_obj.u, layer_obj.log_alpha
            beta, zeta, gamma = self.beta, self.zeta, self.gamma
            s = tf.nn.sigmoid((tf.math.log(u) - tf.math.log(1.0 - u) + log_alpha) / beta)
            s_ = s * (zeta - gamma) + gamma
            train_z = tf.minimum(1.0, tf.maximum(s_, 0.0)),
            test_z = tf.minimum(1.0,
                                tf.maximum(0.0,
                                           tf.nn.sigmoid(log_alpha) * (zeta - gamma) + gamma))

            self.train_z[layer_name] = tf.compat.v1.keras.backend.get_session([train_z]).run([train_z])
            self.test_z[layer_name] = tf.compat.v1.keras.backend.get_session([test_z]).run([test_z])

    def get_param_value(self):
        for idx in range(len(self.block_num)):
            layer_name = 'avg_transform_layer_%d' % idx if self.trans_type == "aver" else 'transform_layer_%d' % idx

            tensor_tmp_s = self.model.get_layer(layer_name).s
            self.s_v[layer_name] = tf.compat.v1.keras.backend.get_session([tensor_tmp_s]).run([tensor_tmp_s])

            tensor_tmp_s_bar = self.model.get_layer(layer_name).s_
            self.s_bar_v[layer_name] = tf.compat.v1.keras.backend.get_session([tensor_tmp_s_bar]).run(
                [tensor_tmp_s_bar])

            tensor_tmp_alpha = self.model.get_layer(layer_name).log_alpha
            self.alpha_v[layer_name] = tf.compat.v1.keras.backend.get_session([tensor_tmp_alpha]).run(
                [tensor_tmp_alpha])

            tensor_tmp_u = self.model.get_layer(layer_name).u
            self.u_v[layer_name] = tf.compat.v1.keras.backend.get_session([tensor_tmp_u]).run([tensor_tmp_u])

            tensor_tmp_z = self.model.get_layer(layer_name).z
            self.z_v[layer_name] = tf.compat.v1.keras.backend.get_session([tensor_tmp_z]).run([tensor_tmp_z])

    def on_train_batch_end(self, batch, logs={}):
        pass
        # self.get_param_value()

    def on_predict_batch_begin(self, batch, logs={}):
        pass

    def on_predict_batch_end(self, batch, logs={}):
        # self.get_param_value()
        pass

    def on_test_end(self, logs=None):
        self.get_z()
        pass


def _get_hash_feat(x, dim):
    y = tf.strings.to_number(tf.reshape(tf.strings.split(x, ',').values, [-1, dim]), out_type=tf.int64)
    return y


class ECMM(BaseTools):

    def __init__(self, logger,
                 **wargs):
        super().__init__(logger, **wargs)
        self.abs_model_dir = os.path.abspath(wargs['model_dir'])
        self.param_file = open(self.abs_model_dir + "/params_value.txt", "a+")
        self.field_weights = np.array(eval(wargs['field_weights'])) if "field_weights" in wargs else None
        self.logger = logger
        self.cate_feature_size = wargs['feature_count']  # feature cnt
        self.field_size = wargs['field_count']
        self.embedding_size = wargs['dim']
        self.lamba = wargs['lambda'] if 'lambda' in wargs else 0.0
        self.type = wargs['ecmm_type']
        self.sb_hidden_size = wargs['sb_hidden_size']
        self.trans_units = wargs['trans_units']
        self.block_num = wargs['block_num']
        self.beta = wargs['beta']
        self.epsilon = wargs['epsilon']
        self.gamma = wargs['gamma']
        self.l0_reg = None if self.lamba == 0.0 else L0Regularizer(
            lamba=self.lamba,
            beta=self.beta,
            gamma=self.gamma,
            zeta=self.epsilon
        )
        self.deep_layers = wargs['deep_layers']
        self.learning_rate = wargs['learning_rate']
        self.task_num = wargs['task_num']
        self.params = wargs
        self.model_dir = wargs['model_dir']

    def data_split_func_auxi(self, x):
        str_lst = tf.reshape(tf.strings.split(x, '\t').values, [-1, 6])  # [N, sep_num]
        auxi_info = str_lst[:, 0:3]
        return auxi_info

    def data_split_func_input(self, x):
        str_lst = tf.reshape(tf.strings.split(x, '\t').values, [-1, 6])  # [N, sep_num]
        user_hash_feat = _get_hash_feat(str_lst[:, 3], 6)
        item_hash_feat = _get_hash_feat(str_lst[:, 4], 6)
        cross_hash_feat = _get_hash_feat(str_lst[:, 5], 10)
        feat = tf.concat([user_hash_feat, item_hash_feat, cross_hash_feat], axis=-1)
        return feat

    def data_split_func(self, x):
        str_lst = tf.reshape(tf.strings.split(x, '\t').values, [-1, 6])  # [N, sep_num]
        session_id = str_lst[:, 0:1]
        itemid = str_lst[:, 1:2]
        label = tf.strings.to_number(tf.reshape(tf.strings.split(str_lst[:, 2], ',').values, [-1, 3]),
                                     out_type=tf.int64)
        user_hash_feat = _get_hash_feat(str_lst[:, 3], 6)
        item_hash_feat = _get_hash_feat(str_lst[:, 4], 6)
        cross_hash_feat = _get_hash_feat(str_lst[:, 5], 10)
        feat = tf.concat([user_hash_feat, item_hash_feat, cross_hash_feat], axis=-1)
        etr_label = label[:, 0:1] + label[:, 1:2]
        etr_label = tf.concat([etr_label, 1 - etr_label], axis=-1)
        ctr_label = label[:, 0:1]
        ctr_label = tf.concat([ctr_label, 1 - ctr_label], axis=-1)
        label = (ctr_label, etr_label)
        return feat, label

    def _init(self, model_dir=None):
        self._init_graph(model_dir)

    def _init_graph(self, model_dir=None):
        if model_dir and os.path.exists(model_dir):
            self.model = tf.keras.models.load_model(model_dir, custom_objects={'BlockLayer': BlockLayer,
                                                                               "AvgTransformLayer": AvgTransformLayer,
                                                                               "ECMMTransformLayer": ECMMTransformLayer,
                                                                               "TransformLayer": TransformLayer})
            self.logger.info('restore model finished.')
        else:
            input_layer = Input(shape=(self.field_size,), dtype='int64')
            embed_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
            embed_layer = tf.keras.layers.Embedding(
                self.cate_feature_size,
                self.embedding_size,
                input_length=self.field_size,
                embeddings_initializer=embed_initializer,
                name='embed')(input_layer)
            if self.field_weights is not None:
                sorted_index = np.argsort(-self.field_weights)[0:11]
                sorted_index = np.sort(sorted_index)
                embed_layer = tf.gather(embed_layer, tf.compat.v1.to_int32(sorted_index), axis=1)  # (N,s,K)
                self.field_size = 11
            emb = tf.keras.layers.Reshape([self.field_size * self.embedding_size], name='reshape_embed')(
                embed_layer)
            # Set up block and transform layer
            self.trans_layer = []
            layer_num = [_ for _ in self.block_num]
            layer_num.append(self.task_num)
            inp = None
            for idx in range(len(layer_num) - 1):
                b_in, b_next_in = layer_num[idx], layer_num[idx + 1]
                if idx == 0:
                    inp = [emb for _ in range(b_in)]

                inp = BlockLayer(block_num=b_in, out_dim=self.sb_hidden_size, name="blocklayer-" + str(idx))(inp)
                layer_name = "avg_transform_layer_%d" % idx if self.type == "aver" else "transform_layer_%d" % idx
                self.trans_layer.append(
                    ECMMTransformLayer(
                        name=layer_name,
                        in_num=b_in,
                        out_num=b_next_in,
                        units=self.trans_units,
                        zeta=self.epsilon,
                        gamma=self.gamma,
                        beta=self.beta,
                        type=self.type,
                        alpha_regularizer=self.l0_reg
                    )
                )
                inp = self.trans_layer[-1](inp)

            # output
            tower_inputs = inp
            output_layers = []
            output_info = [[2, 'ctr'], [2, 'etr']]
            # Build tower layer from MMoE layer
            for index, tower in enumerate(tower_inputs):
                layer = tower
                for layer_no, unit_num in enumerate(self.deep_layers):
                    layer = Dense(
                        name="tower-%d-%d" % (index, layer_no),
                        units=unit_num,
                        activation='relu',
                        kernel_initializer='glorot_normal',
                        bias_initializer='glorot_normal'
                    )(layer)
                output_layer = Dense(
                    units=output_info[index][0],
                    name="final_tower_%d" % index,
                    activation='softmax',
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal'
                )(layer)
                output_layer = Lambda(
                    lambda x: tf.cast(tf.clip_by_value(x, 1e-5, 1.0 - (1e-5)), dtype=tf.float64),
                    name=output_info[index][1]
                )(output_layer)
                output_layers.append(output_layer)
            # Compile model
            self.model = Model(inputs=[input_layer], outputs=output_layers)
            adam_optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            self.model.compile(
                loss={'ctr': cce, 'etr': cce},
                optimizer=adam_optimizer,
                metrics=['accuracy']
            )
            self.logger.info('init graph finished.')

        self.model.summary()

    def run_train(self, ds, f_loss):
        history = LossHistory()
        sv_callback = SVHistory(self.block_num, self.task_num, self.type, gamma=self.gamma, zeta=self.epsilon,
                                beta=self.beta)
        self.model.fit(
            x=ds,
            verbose=0,
            callbacks=[history, sv_callback])
        f_loss.write('total_loss\tetr_loss\tctr_loss\n')
        losses_l = list()
        for idx, loss in enumerate(history.losses):
            losses_l.append('\t'.join([str(loss), str(history.etr_losses[idx]), str(history.ctr_losses[idx])]))
        f_loss.write('\n'.join(losses_l))
        f_loss.write('\n')

    def predict(self, ds, ds_auxi, f_res):
        sv_callback = SVHistory(self.block_num, self.task_num, self.type, gamma=self.gamma, zeta=self.epsilon,
                                beta=self.beta)
        f_out = open(f_res, 'w')

        ctr_v, etr_v = self.model.predict(ds, callbacks=[sv_callback])
        print('=' * 20 + "sv_History test step" + '=' * 20, file=self.param_file)
        print("=" * 20 + "s:" + "=" * 20, file=self.param_file)
        print(sv_callback.s_v, file=self.param_file)
        print("=" * 20 + "s_bar:" + "=" * 20, file=self.param_file)
        print(sv_callback.s_bar_v, file=self.param_file)
        print("=" * 20 + "z:" + "=" * 20, file=self.param_file)
        print(sv_callback.z_v, file=self.param_file)
        print("=" * 20 + "u:" + "=" * 20, file=self.param_file)
        print(sv_callback.u_v, file=self.param_file)
        print("=" * 20 + "alpha:" + "=" * 20, file=self.param_file)
        print(sv_callback.alpha_v, file=self.param_file)
        print("=" * 20 + "train_z:" + "=" * 20, file=self.param_file)
        for k, v in sv_callback.train_z.items():
            print("=" * 20 + k + " train_z" + "=" * 20, file=self.param_file)
            print(v, file=self.param_file)
        for k, v in sv_callback.test_z.items():
            print("=" * 20 + k + " test_z" + "=" * 20, file=self.param_file)
            print(v, file=self.param_file)

        ctr_v = ctr_v.tolist()
        etr_v = etr_v.tolist()

        sids = list()
        itemids = list()
        labels = list()
        sess = tf.compat.v1.Session()
        sess.run(ds_auxi.initializer)
        auxi_info_iter = ds_auxi.get_next()
        while True:
            try:
                auxi_info_v = sess.run([auxi_info_iter])
                auxi_info_v = auxi_info_v[0]
            except tf.errors.OutOfRangeError:
                self.logger.info('infer auxi_info finished!')
                break
            for item in auxi_info_v:
                sids.append(item[0].decode())
                itemids.append(item[1].decode())
                labels.append(item[2].decode().split(','))

        for i in range(0, len(sids)):
            f_out.write(('{}\t' * 4 + '{}\n').format(sids[i], itemids[i], '\t'.join(map(str, labels[i])), ctr_v[i][0],
                                                     etr_v[i][0] - ctr_v[i][0]))

    def _dump_model(self):
        self.model.save(self.params['model_dir'] + '/model')

    def load_model(self, model_dir):
        return tf.keras.models.load_model(model_dir)

    def SingleTrain(self, f_loss):
        ds = DataLoader(self.params['train_dir'], "local", self.logger, self.params['begin_day'], self.params['end_day']).dataset_get(self.params['batch_size'],
                                                                                            self.params['parallel'],
                                                                                            self.data_split_func)
        checkpoint_path = self.params['model_dir'] + '/model'
        self._init(checkpoint_path)
        self.run_train(ds, f_loss)
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        self._dump_model()
        self.logger.info('save model finished!')

    def SinglePredict(self, local_predict_f):
        tf.keras.backend.clear_session()
        local_test_dir = self.params['test_dir']
        model_dir = self.params['model_dir']
        ds_input = DataLoader(local_test_dir, "local", self.logger).dataset_get(self.params['batch_size'], self.params['parallel'], self.data_split_func_input)
        ds_auxi_iter = DataLoader(local_test_dir, "local", self.logger).dataset_get(self.params['batch_size'], self.params['parallel'], self.data_split_func_auxi).make_initializable_iterator()
        self._init(model_dir+'/model')
        res = self.predict(ds_input, ds_auxi_iter, local_predict_f)
