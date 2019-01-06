import time

import tensorflow as tf
import numpy as np

from utils import *
from model import *

from skimage import measure


def test(image, filename):
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32,
                            name='generated_image')

    Gz = generator(gen_in)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = initialize(sess)
        initial_step = global_step.eval()

        start_time = time.time()

        image = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), image))
        image = sess.run(Gz, feed_dict={gen_in: image})
        end_time = time.time()
        interval = end_time-start_time
        print("Cost:%s s"%interval)
        image = np.resize(image[0], BATCH_SHAPE[1:])
        scipy.misc.imsave(filename + '-output.png', image)
        return image


def denoise(image):
    image = scipy.misc.imread(image, mode='RGB').astype('float32')
    image = np.expand_dims(image, axis=0)
    print(image[0].shape)
    output = test(image)
    return output


if __name__ == '__main__':
    image = scipy.misc.imread(sys.argv[1], mode='RGB').astype('float32')
    image = np.expand_dims(image, axis=0)
    print(image[0].shape)
    test(image, os.path.splitext(sys.argv[1])[0])
