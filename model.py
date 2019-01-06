import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


from utils import *
from conv_helper import *


def generator(input):
    output = tf.layers.conv2d(input, 64, 3, padding='SAME', activation=tf.nn.relu, name="G_conv1")

    for layers in range(2,17,1):
        output2 = tf.layers.conv2d(output, 64, 3, padding='SAME', name='G_conv%d_res1' % layers, use_bias=False)
        output2 = tf.nn.relu(tf.layers.batch_normalization(output2))
        output2 = tf.layers.conv2d(output2, 64, 3, padding='SAME', name='G_conv%d_res2' % layers, use_bias=False)
        output2 = tf.layers.batch_normalization(output2)
        output = output2+output
        output = tf.nn.relu(output)

    output = tf.layers.conv2d(output, 3, 3, padding='SAME', name='G_conv17')

    return input-output


def discriminator(input, reuse2=False):
    conv1, conv1_weights = conv_layer(input, 3, 3, 64, 1, "D_conv1", activation_function=lrelu, reuse=reuse2)
    conv2, conv2_weights = conv_layer(conv1, 3, 64, 64, 2, "D_conv2", activation_function=lrelu, reuse=reuse2)
    conv3, conv3_weights = conv_layer(conv2, 3, 64, 128, 1, "D_conv3", activation_function=lrelu, reuse=reuse2)
    conv4, conv4_weights = conv_layer(conv3, 3, 128, 128, 2, "D_conv4", activation_function=lrelu, reuse=reuse2)
    conv5, conv5_weights = conv_layer(conv4, 3, 128, 256, 1, "D_conv5", activation_function=lrelu, reuse=reuse2)
    conv6, conv6_weights = conv_layer(conv5, 3, 256, 256, 2, "D_conv6", activation_function=lrelu, reuse=reuse2)
    conv7, conv7_weights = conv_layer(conv6, 3, 256, 1, 1, "D_conv7", activation_function=tf.nn.sigmoid, reuse=reuse2)

    return conv7
