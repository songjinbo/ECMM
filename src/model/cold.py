import sys,os
import numpy as np
from datetime import datetime, timedelta
import tensorflow.compat.v1 as tf
from base_tools import BaseTools
from data_loader import DataLoader

class COLD(BaseTools):

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
        self.params = wargs
        self.model_dir = wargs['model_dir']
        self.cold_infer_flag = "cold_infer" in self.params and self.params['cold_infer']
        if self.cold_infer_flag:
            self.select_num = wargs['select_num']
            self.field_size = self.select_num
            self.total_size = self.field_size * self.embedding_size

    def _init(self, iterator, is_train):
        self.session_id, self.itemid, self.label, \
        self.user_hash_feat, \
        self.item_hash_feat, \
        self.cross_hash_feat = iterator.get_next()
        self._init_graph(
            self.user_hash_feat, \
            self.item_hash_feat, \
            self.cross_hash_feat, \
            self.label, is_train)

    def _init_graph(self, \
            user_hash_feat, \
            item_hash_feat, \
            cross_hash_feat, \
            label, \
            is_train):
        hash_feature = tf.concat([
            user_hash_feat, \
            item_hash_feat, \
            cross_hash_feat], axis=1)
        self.weights = self._initialize_weights()
        # model
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], hash_feature) # N * F * K
        if self.cold_infer_flag:
            field_weights = self.params['field_weights']
            sorted_index = np.argsort(-field_weights)[0:self.select_num]
            sorted_index = np.sort(sorted_index)
            self.embeddings = tf.gather(self.embeddings, tf.to_int32(sorted_index), axis=1) # (N,s,K)?
        self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
        if not self.cold_infer_flag:
            self.field_weights = tf.nn.sigmoid(
                tf.expand_dims(tf.add(tf.matmul(self.y_deep, self.weights['cold_w']), self.weights['cold_b']), -1)
            ) # (N , F , 1)
            self.y_deep = tf.reshape(tf.multiply(self.field_weights, self.embeddings), shape=[-1 , self.total_size]) # (N , F * K)

        for i in range(0,len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%i])
            self.y_deep = tf.nn.relu(self.y_deep)
            self.y_deep = tf.layers.batch_normalization(inputs=self.y_deep)
        self.preds = tf.add(tf.matmul(self.y_deep,self.weights['output']),self.weights['output_bias'])
        if self.params['class_num'] == 1:
            self.preds = tf.cast(tf.clip_by_value(tf.nn.sigmoid(self.preds), 1e-5, 1.0 - (1e-5)), dtype=tf.float64, name="preds")
            self.preds = tf.concat([self.preds, 1 - self.preds], axis=-1)
        else:
            self.preds = tf.cast(tf.clip_by_value(tf.nn.softmax(self.preds), 1e-5, 1.0 - (1e-5)), dtype=tf.float64, name='preds')
        # loss
        self.loss_weight = 1.0
        if ('only_use_expose' in self.params) and self.params['only_use_expose'] and label is not None:
            self.loss_weight = 1 - label[:, 2]
        if is_train:
            if self.class_num == 1:
                self.ctr_label_tmp = tf.expand_dims(self.label[:, 0], axis=-1)
                self.ctr_label = tf.concat([self.ctr_label_tmp, 1 - self.ctr_label_tmp], axis=-1)
                self.loss = -tf.reduce_mean(
                    tf.reduce_sum(self.ctr_label * tf.log(self.preds), axis=-1) * self.loss_weight
                )
            else:
                self.loss = -tf.reduce_mean(tf.reshape(label, shape=[-1,self.class_num])* tf.log(self.preds), axis=-1)
                self.loss = tf.reduce_mean(self.loss)
            self.loss_tuple = [self.loss]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
        self.logger.info('init graph finished.')

    def _initialize_weights(self):
        weights = dict()
        #embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01,dtype=tf.float32),
            name='feature_embeddings',dtype=tf.float32)
        #deep layers
        num_layer = len(self.deep_layers)

        #cold layer
        glorot = np.sqrt(2.0 / (self.total_size + self.field_size))
        weights['cold_w'] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(self.total_size, self.field_size)),
			dtype=np.float32
		)
        weights['cold_b'] = tf.Variable(
			tf.constant([0.01] * self.field_size , dtype=tf.float32),
			dtype=np.float32
		)

        # deep layers
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

    def test(self):
        self.params['cold_infer'] = True
        self.params['field_weights'] = np.loadtxt(os.path.abspath(self.model_dir) + "/field_weights.txt")
        self.test_impl()

    def get_one_day_field_weights(self, params, model_name, logger, model_dir):
        tf.reset_default_graph()
        model = model_name(logger=logger, **params)
        iterator = DataLoader(params['train_dir'], "local", logger, params['begin_day'], params['end_day']).dataset_get(params['batch_size'],
                                                                                            params['parallel'],
                                                                                            model.data_split_func).make_initializable_iterator()
        model._init(iterator, is_train=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(iterator.initializer)
        batch_mean_field_weights_list = []
        with sess.as_default():
            tf.train.Saver().restore(sess, os.path.join(model_dir + '/checkpoint', 'train_model'))
            while True:
                try:
                    if 'only_use_expose' in params and params['only_use_expose']:
                        field_weights, data_mask = sess.run([model.field_weights, model.loss_weight]) # (n, f, 1), (n,)
                        data_mask = data_mask.reshape(-1, 1) # (n, 1)
                    else:
                        field_weights = sess.run(model.field_weights)
                        data_mask = np.ones(shape=(field_weights.shape[0], 1))
                    batch_mean_field_weights = \
                        (np.squeeze(field_weights, axis=-1) * data_mask).sum(axis=0, keepdims=True) / data_mask.sum(axis=0)
                    # (1, f)
                    batch_mean_field_weights_list.append(batch_mean_field_weights)
                except tf.errors.OutOfRangeError as e:
                    break
        mean_field_weights = np.concatenate(batch_mean_field_weights_list, axis=0).mean(axis=0, keepdims=True)  # (1 , f)
        return mean_field_weights

    def get_field_weights(self, params, model_name, logger, model_dir):
        if "weight_file" in params and os.path.exists(params['weight_file']):
            return np.loadtxt(params['weight_file'])
        days_field_weights_list = []
        logger.info('*' * 20 + 'calculate field weights')
        days_field_weights_list.append(self.get_one_day_field_weights(self.params, COLD, self.logger, self.model_dir))
        return np.concatenate(days_field_weights_list, axis=0).mean(axis=0)  #(f, )

    def train(self):
        self.logger.info("=" * 20 + "COLD phase-1 start..")
        self.train_impl()
        self.logger.info("=" * 20 + "COLD phase-1 done")
        field_weights = self.get_field_weights(self.params, COLD, self.logger, self.model_dir)
        self.logger.info("=" * 20 + "cal field weights successfully")
        self.logger.info(str(self.params))
        if os.path.isdir(self.model_dir):
            os.system('rm -rf ' + self.model_dir)
        os.system('mkdir -p ' + self.model_dir)
        np.savetxt(os.path.abspath(self.model_dir) + "/field_weights.txt", field_weights)
        self.logger.info(str(field_weights))
        self.logger.info("="*20 + "COLD phase-2 start...")
        self.train_impl()
        self.logger.info("="*20 + "COLD phase-2 done")
