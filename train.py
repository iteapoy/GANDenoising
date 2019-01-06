import time

import tensorflow as tf
import numpy as np

from utils import *
from model import *

from skimage import measure


def train():
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32,
                            name='generated_image')
    real_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32,
                             name='groundtruth_image')

    ###========================== DEFINE MODEL ============================###

    Gz = generator(gen_in)
    Dx = discriminator(real_in)
    Dg = discriminator(Gz, reuse2=True)

    D_loss = tf.reduce_mean(Dx) - tf.reduce_mean(Dg)
    G_loss = ADVERSARIAL_LOSS_FACTOR * tf.reduce_mean(tf.scalar_mul(-1, Dg)) + \
             PSNR_LOSS_FACTOR * PSNR(real_in, Gz) + SSIM_LOSS_FACTOR * SSIM_three(real_in, Gz)

    t_vars = tf.trainable_variables()
    D_vars = [var for var in t_vars if 'D_' in var.name]
    G_vars = [var for var in t_vars if 'G_' in var.name]


    D_solver = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(D_loss, var_list=D_vars, global_step=global_step)
    G_solver = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(G_loss, var_list=G_vars)

    # clip op
    clip_D = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in D_vars]

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        saver = initialize(sess)
        initial_step = global_step.eval()

        validation_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), validation))

        N_iteration = int(N_EPOCHS * N_IMAGES / BATCH_SIZE)

        for step in range(initial_step, N_iteration):
            if step < 25 or step % 500 == 0:
                critic_num = 25
            else:
                critic_num = CRITIC_NUM

            # Update D
            for _ in range(critic_num):
                input_batch = load_next_training_batch()
                training_batch, groundtruth_batch = np.split(input_batch, 2, axis=2)

                training_batch = sess.run(
                    tf.map_fn(lambda img: tf.image.per_image_standardization(img), training_batch))
                groundtruth_batch = sess.run(
                    tf.map_fn(lambda img: tf.image.per_image_standardization(img), groundtruth_batch))

                _, D_loss_cur = sess.run([D_solver, D_loss],
                                         feed_dict={gen_in: training_batch, real_in: groundtruth_batch})
                sess.run(clip_D)

            # Update G
            _, G_loss_cur = sess.run([G_solver, G_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})
            _, G_loss_cur = sess.run([G_solver, G_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})

            if (step + 1) % SKIP_STEP == 0:
                saver.save(sess, CKPT_DIR, step + 1)
                image = sess.run(Gz, feed_dict={gen_in: validation_batch})
                image = np.resize(image[1], BATCH_SHAPE[1:])

                imsave('val_%d' % (step + 1), image)
                image = scipy.misc.imread(IMG_DIR + 'val_%d.png' % (step + 1), mode='RGB').astype('float32')
                psnr = measure.compare_psnr(metrics_image, image, data_range=255)
                ssim = measure.compare_ssim(metrics_image, image, multichannel=True, data_range=255, win_size=11)

                print(
                    "Step {}/{} Gen Loss: ".format(step + 1, N_iteration) + str(G_loss_cur) + " Disc Loss: " + str(
                        D_loss_cur) + " PSNR: " + str(psnr) + " SSIM: " + str(ssim))


if __name__ == '__main__':
    training_dir_list = training_dataset_init()
    validation = load_validation()
    train()
