import sys,os
import numpy as np
import tensorflow.compat.v1 as tf
from base_tools import BaseTools

class ESMM(BaseTools):

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
        self.mode = wargs['mode']
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

    def _mlp(self, input_emb, mode="etr"):
        deep_layers = self.deep_layers[mode]

        for i in range(0,len(deep_layers)):
            input_emb = tf.add(tf.matmul(input_emb, self.weights[mode + "deep_layer_%d" %i]), self.weights[mode + "deep_bias_%d"%i])
            input_emb = tf.nn.relu(input_emb)
            input_emb = tf.layers.batch_normalization(inputs=input_emb)
        preds = tf.add(tf.matmul(input_emb,self.weights[mode + 'output']),self.weights[mode + 'output_bias'])
        preds = tf.cast(tf.clip_by_value(tf.nn.sigmoid(preds), 1e-5, 1.0 - (1e-5)), dtype=tf.float64, name= mode + 'preds')
        return preds

    def _binary_loss(self, preds, label, weight = 1):
        preds = tf.reshape(preds, (-1, 1))
        label = tf.reshape(label, (-1, 1))
        cross_entropy_loss = label * tf.log(preds) + (1 - label) * tf.log(1 - preds)
        cross_entropy_loss = cross_entropy_loss * weight
        return -tf.reduce_mean(cross_entropy_loss)

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
        deep_input  = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
        etr_preds = self._mlp(deep_input, mode='etr')
        ctr_preds = self._mlp(deep_input, mode='ctr')
        etctr = etr_preds * ctr_preds
        output_2 = tf.clip_by_value(etr_preds - etctr, 1e-5, 1.0 - (1e-5))
        output_3 = tf.clip_by_value(1 - etr_preds, 1e-5, 1.0 - (1e-5))
        self.preds = tf.concat([etctr, output_2, output_3], axis = -1)
        # loss
        if is_train:
            label = tf.reshape(label, shape=[-1, self.class_num])
            etr_label = label[:, 0] + label[:, 1]
            ctr_label = label[:, 0]
            etctr_label = label[:, 0]
            loss_etctr_weight = 1 - label[:, 2]
            loss_etr = self._binary_loss(etr_preds, etr_label)
            loss_etctr = self._binary_loss(etctr, etctr_label)
            loss_ctr = self._binary_loss(ctr_preds, ctr_label, weight=loss_etctr_weight)
            if self.mode == 1:
                self.loss = loss_etr + loss_etctr + 0.0 * loss_ctr
            elif self.mode == 2:
                self.loss = loss_etr + loss_etctr + loss_ctr
            elif self.mode == 3:
                self.loss = loss_etr + 0.0 * loss_etctr + loss_ctr

            else:
                self.logger.info("Parameter mode is invalid.")

            self.loss_tuple = [self.loss]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
        self.logger.info('init graph finished.')

    def _init_mlp(self, mode = "etr"):
        weights = dict()
        deep_layers = self.deep_layers[mode]
        num_layer = len(deep_layers)
        glorot = np.sqrt(2.0/(self.total_size + deep_layers[0]))

        weights[mode + 'deep_layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(self.total_size, deep_layers[0])),dtype=np.float32
        )
        weights[mode + 'deep_bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1, deep_layers[0])),dtype=np.float32
        )
        for i in range(1,num_layer):
            glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
            weights[mode + "deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(deep_layers[i - 1], deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights[mode + "deep_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]
        # final concat projection layer
        glorot = np.sqrt(2.0/(deep_layers[-1] + 1))
        weights[mode + 'output'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(deep_layers[-1], 1)),dtype=np.float32)
        weights[mode + 'output_bias'] = tf.Variable(tf.constant([0.01], dtype=tf.float32),dtype=np.float32)
        return weights


    def _initialize_weights(self):
        weights = dict()
        #embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01,dtype=tf.float32),
            name='feature_embeddings',dtype=tf.float32)
        weights.update(self._init_mlp(mode = "etr"))
        weights.update(self._init_mlp(mode = "ctr"))

        return weights
