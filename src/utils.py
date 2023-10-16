#coding=utf-8
import tensorflow.compat.v1 as tf

def _get_hash_feat(x):
    y = tf.string_to_number(tf.string_split([x], ',').values, out_type=tf.int64)
    return y
