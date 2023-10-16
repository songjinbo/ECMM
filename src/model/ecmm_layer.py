import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec, Dense
from tensorflow.keras.initializers import glorot_normal, RandomNormal

EPSILON = 1e-5


# LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION
class L0Regularizer(regularizers.Regularizer):
    def __init__(self,
                 lamba,
                 beta,
                 gamma,
                 zeta,
                 **kwargs):
        self.lamba = lamba
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

    def __call__(self, x):
        # log_alpha => x
        l0 = tf.clip_by_value(
            tf.nn.sigmoid(x - self.beta * tf.math.log(-self.gamma / self.zeta)),
            clip_value_min=EPSILON,
            clip_value_max=1 - EPSILON
        )
        return self.lamba * tf.reduce_sum(l0, axis=[0, 1])

    def get_config(self):
        return {
            "lamba": float(self.lamba),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "zeta": float(self.zeta)
        }


# [(B , d), ...] -> [(B , d'), ...]
# one-layer full-connected network
class BlockLayer(Layer):
    def __init__(self,
                 block_num,
                 out_dim,
                 **kwargs):
        self.block_num = block_num
        self.out_dim = out_dim
        self.layer_name = kwargs['name']

        super(BlockLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == self.block_num
        self.blocks_kernel = [
            self.add_weight(
                name="%s-Dense-%d" % (self.layer_name, idx),
                shape=[int(input_shape[idx][-1]), self.out_dim],
                initializer=glorot_normal(),
            ) for idx in range(self.block_num)
        ]

        super(BlockLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        for idx, inp in enumerate(inputs):
            outputs.append(tf.nn.relu(tf.matmul(inp, self.blocks_kernel[idx])))

        return outputs

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.out_dim) for _ in range(self.block_num)]

    def get_config(self):
        config = {
            'block_num': self.block_num,
            "out_dim": self.out_dim
        }
        base_config = super(BlockLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# [(B,d), ...] => [(B, d)， ..., (B,d)]
class AvgTransformLayer(Layer):
    def __init__(self,
                 in_num,
                 out_num,
                 epsilon,
                 gamma,
                 beta,
                 **kwargs):
        self.in_num = in_num
        self.out_num = out_num
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.th = 1e-5
        self.u = tf.Variable(tf.random.uniform([out_num, in_num], minval=0, maxval=1),
                             trainable=False)  # matrix? trainable?
        # self.u = tf.Variable(tf.constant(0.3, shape=[out_num, in_num]), trainable=False)
        super(AvgTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        in_num = len(input_shape)
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.out_num, in_num),
            initializer=RandomNormal(0, 1.0)
        )
        super(AvgTransformLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        alpha = tf.nn.relu(self.alpha) + self.th
        self.s = tf.nn.sigmoid((tf.math.log(self.u) - tf.math.log(1 - self.u) + tf.math.log(alpha)) / self.beta)
        self.s_ = self.s * (self.epsilon - self.gamma) + self.gamma
        self.z = K.in_train_phase(
            tf.minimum(1.0, tf.maximum(self.s_, 0.0)),
            tf.minimum(1.0,
                       tf.maximum(0.0,
                                  tf.nn.sigmoid(tf.math.log(alpha)) * (self.epsilon - self.gamma) + self.gamma))
        )
        inp = tf.stack(inputs, axis=1)  # (B, b, d)
        assert len(inp.shape) == 3
        weighted_sum = tf.einsum("ijk,bj->ibk", inp, self.z)
        return tf.unstack(weighted_sum, self.out_num, axis=1)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1]) for _ in range(self.out_num)]

    def get_config(self):
        config = {
            "in_num": self.in_num,
            'out_num': self.out_num,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "beta": self.beta
        }
        base_config = super(AvgTransformLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# [(B,d), ...] => [(B, d)， ..., (B,d)]
class TransformLayer(Layer):
    def __init__(self,
                 in_num,
                 out_num,
                 units,
                 epsilon,
                 gamma,
                 beta,
                 **kwargs):
        self.in_num = in_num
        self.out_num = out_num
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.units = units
        self.th = 1e-5
        self.u = tf.Variable(tf.random.uniform([in_num, out_num], minval=0, maxval=1),
                             trainable=False)  # matrix? trainable?
        super(TransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.in_num, self.out_num),
            initializer=RandomNormal(0, 1.0)
        )
        input_dim = int(input_shape[0][-1])
        self.trans_weights = [[self.add_weight(
            name="trans_weight_%d%d" % (i, j),
            shape=(input_dim, self.units),
            initializer=glorot_normal()
        ) for j in range(self.out_num)] for i in range(self.in_num)]
        super(TransformLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        alpha = tf.nn.relu(self.alpha) + self.th
        self.s = tf.nn.sigmoid((tf.math.log(self.u) - tf.math.log(1.0 - self.u) + tf.math.log(alpha)) / self.beta)
        self.s_ = self.s * (self.epsilon - self.gamma) + self.gamma
        self.z = K.in_train_phase(
            tf.minimum(1.0, tf.maximum(self.s_, 0.0)),
            tf.minimum(1.0,
                       tf.maximum(0.0,
                                  tf.nn.sigmoid(tf.math.log(alpha)) * (self.epsilon - self.gamma) + self.gamma))
        )
        inp = tf.concat(inputs, axis=1)  # (B, in_num*input_dim)
        z_weights = []
        for i in range(self.in_num):
            tmp_list = []
            for j in range(self.out_num):
                tmp_list.append(self.trans_weights[i][j] * self.z[i, j])
            z_weights.append(tf.concat(tmp_list, axis=1))
        z_weights = tf.concat(z_weights, axis=0)  # (in_num * input_dim, out_num * out_dim)
        weighted_sum = tf.matmul(inp, z_weights)  # (B , out_num * out_dim)
        return tf.split(weighted_sum, self.out_num, axis=1)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.units) for _ in range(self.out_num)]

    def get_config(self):
        config = {
            "in_num": self.in_num,
            'out_num': self.out_num,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "beta": self.beta,
            "units": self.units
        }
        base_config = super(TransformLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class ECMMTransformLayer(Layer):
    def __init__(self,
                 in_num,
                 out_num,
                 units,
                 zeta,
                 gamma,
                 beta,
                 type,
                 alpha_regularizer,
                 **kwargs):
        self.in_num = in_num
        self.out_num = out_num
        self.zeta = zeta
        self.gamma = gamma
        self.beta = beta
        self.units = units if type != "aver" else None
        epsilon = 1e-5
        self.type = type
        self.u_shape = [out_num, in_num] if type == "aver" else [in_num, out_num]
        self.alpha_regularizer = alpha_regularizer
        self.u = tf.Variable(tf.random.uniform(self.u_shape, minval=epsilon, maxval=1 - epsilon),
                             trainable=False)  # matrix? trainable?
        super(ECMMTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.log_alpha = self.add_weight(
            name="log_alpha",
            shape=tuple(self.u_shape),
            initializer=RandomNormal(0, 0.01),
            regularizer=self.alpha_regularizer
        )
        input_dim = int(input_shape[0][-1])
        if self.type != "aver":
            self.trans_weights = [[self.add_weight(
                name="trans_weight_%d%d" % (i, j),
                shape=(input_dim, self.units),
                initializer=glorot_normal()
            ) for j in range(self.out_num)] for i in range(self.in_num)]
        super(ECMMTransformLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        self.s = tf.nn.sigmoid((tf.math.log(self.u) - tf.math.log(1.0 - self.u) + self.log_alpha) / self.beta)
        self.s_ = self.s * (self.zeta - self.gamma) + self.gamma
        self.z = K.in_train_phase(
            tf.minimum(1.0, tf.maximum(self.s_, 0.0)),
            tf.minimum(1.0,
                       tf.maximum(0.0,
                                  tf.nn.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma))
        )

        if self.type == 'aver':
            inp = tf.stack(inputs, axis=1)  # (B, b, d)
            assert len(inp.shape) == 3
            weighted_sum = tf.einsum("ijk,bj->ibk", inp, self.z)
            return tf.unstack(weighted_sum, self.out_num, axis=1)
        else:
            inp = tf.concat(inputs, axis=1)  # (B, in_num*input_dim)
            z_weights = []
            for i in range(self.in_num):
                tmp_list = []
                for j in range(self.out_num):
                    tmp_list.append(self.trans_weights[i][j] * self.z[i, j])
                z_weights.append(tf.concat(tmp_list, axis=1))
            z_weights = tf.concat(z_weights, axis=0)  # (in_num * input_dim, out_num * out_dim)
            weighted_sum = tf.matmul(inp, z_weights)  # (B , out_num * out_dim)
            return tf.split(weighted_sum, self.out_num, axis=1)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1] if self.type == "aver" else self.units) for _ in
                range(self.out_num)]

    def get_config(self):
        config = {
            "in_num": self.in_num,
            'out_num': self.out_num,
            "zeta": self.zeta,
            "gamma": self.gamma,
            "beta": self.beta,
            "units": self.units,
            "type": self.type,
            "alpha_regularizer" : self.alpha_regularizer
        }
        base_config = super(ECMMTransformLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
