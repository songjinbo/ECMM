import sys,os
from datetime import datetime, timedelta
import shutil
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from base_tools import BaseTools
from data_loader import DataLoader

EPSILON = 1e-5
O = [0.4 for _ in range(6)] + [1.5 for _ in range(6)] + [3.0 for _ in range(9)] + [1.5]

class FSCD(BaseTools):

    def __init__(self, logger,
                 **wargs):
        super().__init__(logger, **wargs)
        self.class_num = wargs['class_num'] if 'class_num' in wargs else 3
        self.logger = logger
        self.cate_feature_size = wargs['feature_count']  #feature cnt
        self.field_size = wargs['field_count']
        self.embedding_size = wargs['dim']
        self.deep_layers = wargs['deep_layers']
        self.total_size = self.field_size * self.embedding_size
        self.learning_rate = wargs['learning_rate']

        self.t = wargs['t']
        self.gamma1 = wargs['gamma1']
        self.gamma2 = wargs['gamma2']
        self.lamda = wargs['lambda']
        self.K = wargs['K']

        self.train_dir = wargs['train_dir']
        self.batch_size = wargs['batch_size']
        self.parallel = wargs['parallel']
        self.model_dir = wargs['model_dir']

        self.phase1_ckp_path = self.model_dir + "/phase1"
        self.phase1_ckp_tm_path = self.phase1_ckp_path + "/train_model"
        self.phase2_ckp_path = self.model_dir + "/checkpoint"
        self.phase2_ckp_tm_path = self.phase2_ckp_path + "/train_model"

        self.f_loss = open(self.model_dir + '/loss.txt', 'a+')
        self.delta_path = os.path.abspath(wargs['model_dir']) + "/delta.txt"

        self.params = wargs

    def _init(self, iterator, is_train, finetuning = True):
        self.session_id, self.itemid, self.label, \
        self.user_hash_feat, \
        self.item_hash_feat, \
        self.cross_hash_feat = iterator.get_next()
        self._init_graph(
            self.user_hash_feat, \
            self.item_hash_feat, \
            self.cross_hash_feat, \
            self.label, is_train, finetuning)

    def _init_graph(self, \
            user_hash_feat, \
            item_hash_feat, \
            cross_hash_feat, \
            label, \
            is_train, finetuning):
        label = tf.cast(label, dtype=tf.float32)
        hash_feature = tf.concat([
            user_hash_feat, \
            item_hash_feat, \
            cross_hash_feat], axis=1)
        self.weights = self._initialize_weights()
        # model
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], hash_feature) # N * F * K
        if finetuning:
            delta = np.loadtxt(self.delta_path)
            sorted_index = np.argsort(-delta)[0:self.K]
            mask = np.zeros(self.field_size)
            mask[sorted_index] = 1.0
            mask = tf.constant(mask.reshape((self.field_size, 1)), dtype=tf.float32)
            self.embeddings = self.embeddings * mask
        else:
            with tf.variable_scope("phase-1", reuse=tf.AUTO_REUSE):
                self.u = tf.get_variable(
                    initializer=tf.random.uniform((self.field_size, 1), minval=EPSILON, maxval=1 - EPSILON),
                    trainable=False,
                    name="u"
                )
                self.delta = tf.nn.sigmoid(
                    tf.get_variable(
                        initializer=tf.random.normal((self.field_size, 1)),
                        dtype=tf.float32,
                        name="delta"
                    )
                )
            self.z = tf.nn.sigmoid(
                1 / self.t * (tf.log(self.delta) - tf.log(1.0 - self.delta) + tf.log(self.u) - tf.log(1.0 - self.u))
            ) # (field_size, 1)
            self.o = tf.constant(O)
            self.c = self.gamma1 * self.o + self.gamma2 * self.embedding_size
            self.theta = 1 - tf.nn.sigmoid(self.c)
            self.alpha = tf.log(1.0 - self.theta) - tf.log(self.theta)
            self.embeddings = self.z * self.embeddings


        self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
        for i in range(0,len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%i])
            self.y_deep = tf.nn.relu(self.y_deep)
            self.y_deep = tf.layers.batch_normalization(inputs=self.y_deep)
        self.preds = tf.add(tf.matmul(self.y_deep,self.weights['output']),self.weights['output_bias'])


        if self.params['class_num'] == 1:
            self.preds = tf.cast(tf.clip_by_value(tf.nn.sigmoid(self.preds), 1e-5, 1.0 - (1e-5)), dtype=tf.float32, name="preds")
            self.preds = tf.concat([self.preds, 1 - self.preds], axis=-1)
        else:
            self.preds = tf.cast(tf.clip_by_value(tf.nn.softmax(self.preds), 1e-5, 1.0 - (1e-5)), dtype=tf.float32, name='preds')
        # loss
        self.loss_weight = 1.0
        if ('only_use_expose' in self.params) and self.params['only_use_expose'] and label is not None:
            self.loss_weight = 1 - label[:, 2]
        if is_train:
            if self.class_num == 1:
                self.ctr_label_tmp = tf.expand_dims(label[:, 0], axis=-1)
                self.ctr_label = tf.concat([self.ctr_label_tmp, 1 - self.ctr_label_tmp], axis=-1)
                self.loss = -tf.reduce_mean(
                    tf.reduce_sum(self.ctr_label * tf.log(self.preds), axis=-1) * self.loss_weight
                )
            else:
                self.loss = -tf.reduce_mean(tf.reshape(label, shape=[-1,self.class_num])* tf.log(self.preds), axis=-1)
                self.loss = tf.reduce_mean(self.loss)
            if not finetuning:
                norm = 0
                for k , v in self.weights.items():
                    norm += tf.norm(v, ord=2)
                norm *= self.lamda
                norm += tf.reduce_sum(self.alpha * self.z)
                self.loss += (norm / self.batch_size)

            self.loss_tuple = [self.loss]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
        self.logger.info('init graph finished.')

    def _initialize_weights(self, scope="weight"):
        weights = dict()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            #embeddings
            weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01,dtype=tf.float32),
                name='feature_embeddings',dtype=tf.float32)

            #deep layers
            num_layer = len(self.deep_layers)
            glorot = np.sqrt(2.0/(self.total_size + self.deep_layers[0]))

            weights['deep_layer_0'] = tf.Variable(
                np.random.normal(loc=0,scale=glorot,size=(self.total_size,self.deep_layers[0])),dtype=np.float32
            )
            weights['deep_bias_0'] = tf.Variable(
                np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32
            )
            for i in range(1,num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                weights["deep_layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                weights["deep_bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # final concat projection layer
            glorot = np.sqrt(2.0/(self.deep_layers[-1] + 1))
            weights['output'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(self.deep_layers[-1],self.class_num)),dtype=np.float32)
            weights['output_bias'] = tf.Variable(tf.constant([0.01]*self.class_num, dtype=tf.float32),dtype=np.float32)
        return weights

    def train(self):
        tf.reset_default_graph()
        self.train_pipe()
        tf.reset_default_graph()
        self.train_pipe(finetuning=True)

    def train_pipe(self, finetuning=False):
        start_time = time.time()
        self.logger.info("*" * 20 + "phase-1 train start...")
        self.logger.info("*" * 20 + "Training Start...")
        iterator = DataLoader(self.train_dir, "local", self.logger, self.params['begin_day'], self.params['end_day']). \
            dataset_get(self.batch_size, self.parallel, self.data_split_func). \
            make_initializable_iterator()

        sess = self._init_session(iterator, finetuning)
        loss = self._train(sess, finetuning)

        self.f_loss.write('{}\n'.format('\n'.join(map(str, loss))))
        self.f_loss.flush()

        self._save_model(sess, finetuning)
        if not finetuning:
            self._save_delta(sess)
        sess.close()

        end_time = time.time()
        self.f_loss.write('{} finished. time cost:{}\n'.format("finetuning" if finetuning else "phase-1 train", (end_time - start_time)))
        self.f_loss.flush()
        self.logger.info(f'save {"finetuning" if finetuning else "phase-1 train"} model loss finished!')

    def _save_delta(self, sess):
        delta = sess.run(self.delta)
        np.savetxt(self.delta_path, delta)

    def _save_model(self, sess, finetuning):
        ckp_path = self.phase1_ckp_path if not finetuning else self.phase2_ckp_path
        ckp_tm_path = self.phase1_ckp_tm_path if not finetuning else self.phase2_ckp_tm_path
        if os.path.exists(ckp_path):
            shutil.rmtree(ckp_path)
        os.mkdir(ckp_path)

        saver = tf.train.Saver()
        saver.save(sess, ckp_tm_path)
        self.logger.info("save ckpt finished")


    def _init_session(self, iterator, finetuning):
        self._init(iterator, is_train=True, finetuning=finetuning)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        ckp_path = self.phase1_ckp_path if not finetuning else self.phase2_ckp_path
        ckp_tm_path = self.phase1_ckp_tm_path if not finetuning else self.phase2_ckp_tm_path

        if os.path.exists(ckp_path):
            saver = tf.train.Saver()
            saver.restore(sess, ckp_tm_path)
        else:
            sess.run(tf.global_variables_initializer())
            if finetuning:
                variables = tf.trainable_variables()
                variables_to_restore = [v for v in variables if "weight" in v.name]
                re_saver = tf.train.Saver(variables_to_restore)
                re_saver.restore(sess, self.phase1_ckp_tm_path)
                ss = ",".join([v.name for v in variables_to_restore])
                self.logger.info("*" * 20 + f"Restore from {self.phase1_ckp_tm_path}:{ss}")

        sess.run(iterator.initializer)
        return sess

    def _train(self, sess, finetuning):
        loss_l = []
        while True:
            try:
                _, loss_tuple = sess.run([self.optimizer, self.loss_tuple])
                if not finetuning:
                    sess.run(tf.variables_initializer([self.u]))
                loss_l.append(loss_tuple)

            except tf.errors.OutOfRangeError as e:
                self.f_loss.write('single train finished. train step:{}\n'.format(len(loss_l)))
                self.f_loss.flush()
                break
        return loss_l
