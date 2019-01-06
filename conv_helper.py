import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import *

def conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False):
    with tf.variable_scope(scope_name,reuse=reuse):
        filter = tf.Variable(tf.random_normal([ksize, ksize, in_channels, out_channels], stddev=0.01))
        output = tf.nn.conv2d(input_image, filter, strides=[1, stride, stride, 1], padding='SAME')
        output = slim.batch_norm(output)
        if activation_function:
            output = activation_function(output)
        return output, filter
