'''
Script to train the net
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import ipdb
import numpy as np
import tensorflow as tf
import SENN

# data_dir = '/media/nca/data/raw_data_multi/train_bin'
# val_dir = '/media/nca/data/raw_data_multi/val_bin'
data_dir = '/home/nca/Downloads/train_bin'
val_dir = '/home/nca/Downloads/val_bin'
# dir to write summary
sum_dir = '/home/nca/Downloads/multichannel_mt_large_ft/sum_20'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_dir',
    '/home/nca/Downloads/multichannel_mt_large_ft/ckpt_20',
    """Directory where to write event logs """
    """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000000,
                            """Number of batches to run.""")

reg_fac = 1e-20  # regularization factor
LR = 0.00001  # learning rate

batch_size = 256
NEFF = 129  # FFT points
NFFT = 256  # effective FFT points
N_IN = 8  # number of frames fed to the model
N_OUT = 1  # number of output frames
mtl_fac = 0.3  # multi-task learning factor
ep_example = 3959151  # examples per epoch(training)
# batches per epoch(training)
ep_batch = np.round(ep_example / batch_size)
# ipdb.set_trace()
ep_example_val = 693959  # examples per epoch(validation)
# batches per epoch(validation)
val_batch = np.round(ep_example_val / batch_size)
val_loss = np.zeros([10000, 2])


def train():

    with tf.Graph().as_default():

        SE_net = SENN.SE_NET(data_dir, batch_size, NEFF, N_IN, N_OUT, reg_fac)

        # training reader tf queue
        imagesf_t, imagesb_t, targets_t, vads_t = SENN.input(
            False, data_dir, batch_size)

        # validation reader
        imagesf_v, imagesb_v, targets_v, vads_v = SENN.input(
            True, val_dir, batch_size)

        # placeholder to be fed with spectrogram and reference
        imagesf = tf.placeholder(tf.float32, shape=[batch_size, N_IN, NEFF])
        imagesb = tf.placeholder(tf.float32, shape=[batch_size, N_IN, NEFF])
        targets = tf.placeholder(tf.float32, shape=[batch_size, NEFF])
        vads = tf.placeholder(tf.float32, shape=[batch_size, NEFF])

        inf_targets, inf_vads = SE_net.inference(
            imagesf, imagesb, is_train=True)

        # loss and loss plus regularization
        loss, loss_o = SE_net.loss(
            inf_targets, inf_vads, targets, vads, mtl_fac)

        # training op of optimizer
        train_op = SE_net.train(loss, LR)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(
            sum_dir,
            sess.graph)

        val_loss_id = 0
        # stop when no progress of validation set in 4 epochs
        counter_no_progress = 0
        min_val_loss = 1000

        time.sleep(30)

        for step in xrange(FLAGS.max_steps):

            start_time = time.time()
            # training queue
            imagefs_batch, imagebs_batch, targets_batch, vads_batch = sess.run(
                [imagesf_t, imagesb_t, targets_t, vads_t])

            # optimization
            _, loss_value = sess.run(
                [train_op, loss],
                feed_dict={
                    imagesf: imagefs_batch,
                    imagesb: imagebs_batch,
                    targets: targets_batch,
                    vads: vads_batch})
            # ipdb.set_trace()
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 100 == 0:
                # display progress every 100 steps
                # if step % 10000000 == 0:
                #     ipdb.set_trace()
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch, epoch %d)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                # add sum every 100 steps
                summary_str = sess.run(
                    summary_op, feed_dict={
                        imagesf: imagefs_batch,
                        imagesb: imagebs_batch,
                        targets: targets_batch,
                        vads: vads_batch})
                summary_writer.add_summary(summary_str, step)

            if step % ep_batch == 0:
                # doing validation after every epoch of training
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                np_val_loss = 0
                np_val_loss_o = 0
                print('Doing validation, please wait ...')
                # average validation loss and save
                for j in range(int(val_batch)):
                    (imagefs_batch, imagebs_batch,
                        targets_batch, vads_batch) = sess.run(
                        [imagesf_v, imagesb_v, targets_v, vads_v])
                    loss_value, loss_o_value = sess.run(
                        [loss, loss_o],
                        feed_dict={
                            imagesf: imagefs_batch,
                            imagesb: imagebs_batch,
                            targets: targets_batch,
                            vads: vads_batch})
                    np_val_loss += loss_value
                    np_val_loss_o += loss_o_value
                mean_val_loss = np_val_loss / val_batch
                mean_val_loss_o = np_val_loss_o / val_batch
                if mean_val_loss >= min_val_loss:
                    counter_no_progress += 1
                else:
                    counter_no_progress = 0
                    min_val_loss = mean_val_loss
                print('validation loss %.2f' % mean_val_loss_o)
                print('reg validation loss %.2f' % mean_val_loss)

                val_loss[val_loss_id, 0] = mean_val_loss
                val_loss[val_loss_id, 1] = mean_val_loss_o
                val_loss_id += 1
                np.save(sum_dir + '/val_loss.npy', val_loss)
                # stop when no progress is made in 4 epochs
                if counter_no_progress >= 4:
                    break


train()
