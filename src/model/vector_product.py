import sys,os
import numpy as np
import tensorflow.compat.v1 as tf
from base_tools import BaseTools
from utils import _get_hash_feat

class VECTOR_PRODUCT(BaseTools):

    def __init__(self, logger,
                 **wargs):
        super().__init__(logger, **wargs)
        self.logger = logger
        self.user_cate_feature_size = int(wargs['feature_count']/2)  # user feature cnt
        self.ad_cate_feature_size = int(wargs['feature_count']/2)  # user feature cnt
        self.user_field_size = wargs['user_feat_size']
        self.ad_field_size = wargs['item_feat_size']
        self.embedding_size = wargs['dim']
        self.deep_layers = wargs['deep_layers']
        self.user_total_size = self.user_field_size * self.embedding_size
        self.ad_total_size = self.ad_field_size * self.embedding_size
        self.learning_rate = wargs['learning_rate']
        self.params = wargs
        self.model_dir = wargs['model_dir']

    def data_split_func(self, x):
        str_lst = tf.string_split([x],'\t').values
        session_id = str_lst[0]
        itemid = str_lst[1]
        label = tf.string_to_number(tf.string_split([str_lst[2]], ',').values, out_type=tf.float64)
        user_hash_feat = tf.string_split([str_lst[3]], ',').values
        item_hash_feat = tf.string_split([str_lst[4]], ',').values
        cross_hash_feat = tf.string_split([str_lst[5]], ',').values
        return session_id, itemid, label, user_hash_feat, item_hash_feat, cross_hash_feat

    def _initialize_weights(self, scope, input_size, feature_size):
        with tf.variable_scope(scope):
            weights = dict()
            #embeddings
            weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([feature_size, self.embedding_size],0.0,0.01,dtype=tf.float32),
                name='feature_embeddings',dtype=tf.float32)
            #deep layers
            num_layer = len(self.deep_layers)
            glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))

            weights['deep_layer_0'] = tf.Variable(
                np.random.normal(loc=0,scale=glorot,size=(input_size, self.deep_layers[0])),dtype=np.float32
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
            return weights

    def _build_ff(self, y_deep, weights):
        # model
        for i in range(0,len(self.deep_layers)):
            y_deep = tf.add(tf.matmul(y_deep,weights["deep_layer_%d" %i]), weights["deep_bias_%d"%i])
            y_deep = tf.nn.relu(y_deep)
            y_deep = tf.layers.batch_normalization(inputs=y_deep)
        return y_deep

    def _build_user_tower(self, user_feat):
        self.user_hash_feat = tf.strings.to_hash_bucket(user_feat, self.user_cate_feature_size)
        self.user_weight = self._initialize_weights(scope='user', input_size=self.user_total_size, feature_size=self.user_cate_feature_size)
        self.user_embeddings = tf.nn.embedding_lookup(self.user_weight['feature_embeddings'], self.user_hash_feat) # N * F * K
        self.user_embeddings  = tf.reshape(self.user_embeddings, shape=[-1, self.user_field_size*self.embedding_size])
        self.user_output = tf.identity(self._build_ff(self.user_embeddings, self.user_weight), name='user_output')

    def _build_ad_tower(self, item_feat):
        self.ad_hash_feat = tf.strings.to_hash_bucket(item_feat, self.ad_cate_feature_size)
        self.ad_weight = self._initialize_weights(scope='ad', input_size=self.ad_total_size, feature_size=self.ad_cate_feature_size)
        self.ad_embeddings = tf.nn.embedding_lookup(self.ad_weight['feature_embeddings'], self.ad_hash_feat) # N * F * K
        self.ad_embeddings = tf.reshape(self.ad_embeddings, shape=[-1, self.ad_field_size*self.embedding_size])
        self.ad_output = tf.identity(self._build_ff(self.ad_embeddings, self.ad_weight), name='ad_output')

    def _init(self, iterator, is_train):
        self.session_id, self.itemid, self.label, \
        self.user_hash_feat, self.item_hash_feat, _ = iterator.get_next()
        self._init_graph(self.user_hash_feat, self.item_hash_feat)
        # loss
        if is_train:
            self.ctr_label_tmp = tf.expand_dims(self.label[:, 0], axis=-1)
            self.ctr_label = tf.concat([self.ctr_label_tmp, 1-self.ctr_label_tmp], axis=-1)
            self.loss = -tf.reduce_mean(tf.reduce_sum(self.ctr_label * tf.log(self.preds), axis=-1))
            self.loss_tuple = [self.loss]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

    def _init_graph(self, user_hash_feat, item_hash_feat):
        # build user tower
        self._build_user_tower(user_hash_feat)
        # build ad tower
        self._build_ad_tower(item_hash_feat)
        # output
        self.logits = tf.expand_dims(tf.reduce_sum(self.user_output * self.ad_output, axis=-1), axis=-1, name='logits')
        self.preds = tf.cast(tf.clip_by_value(tf.nn.sigmoid(self.logits), 1e-5, 1.0 - (1e-5)), dtype=tf.float64, name='preds')
        self.preds = tf.concat([self.preds, 1-self.preds], axis=-1)
