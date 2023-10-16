import sys,os
import numpy as np
import tensorflow.compat.v1 as tf
from base_tools import BaseTools
from utils import _get_hash_feat

class ECM(BaseTools):

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
        self.y_deep  = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
        for i in range(0,len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%i])
            self.y_deep = tf.nn.relu(self.y_deep)
            self.y_deep = tf.layers.batch_normalization(inputs=self.y_deep)
        self.preds = tf.add(tf.matmul(self.y_deep,self.weights['output']),self.weights['output_bias'])
        self.preds = tf.cast(tf.clip_by_value(tf.nn.softmax(self.preds), 1e-5, 1.0 - (1e-5)), dtype=tf.float64, name='preds')
        # loss
        if is_train:
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
