'''
Script for training the model
'''
import tensorflow as tf
import numpy as np
from input import BatchGenerator
from model import MultiRnn
import time
from datetime import datetime
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


sum_dir = 'sum'  # dir to write summary
train_dir = 'ckpt'  # dir to store the model
data_dir = 'train.pkl'  # dir of the data set
NEFF = 129  # effective FFT points
batch_size = 128
num_steps = 20
epochs = 2000
cell_type = 'NL_LSTM'
state_size = 256
output_size = 129
num_layer = 3
learning_rate = 0.0001

# build the model
rnn_model = MultiRnn(
    cell_type, state_size, output_size,
    batch_size, num_layer, learning_rate, num_steps)

# input data and referene data placeholder
in_data = tf.placeholder(
    tf.float32, [batch_size, num_steps, 2 * NEFF])
ref_data = tf.placeholder(
    tf.float32, [batch_size, num_steps, NEFF])

# make inference
init_state, final_state, inf_data = rnn_model.inference(in_data)

# compute loss
loss = rnn_model.loss(inf_data, ref_data)

saver = tf.train.Saver(tf.all_variables())

summary_op = tf.merge_all_summaries()

train_op = rnn_model.train(loss)

batch_gen = BatchGenerator(data_dir, batch_size, num_steps, epochs)

with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(
        sum_dir, sess.graph)
    sess.run(tf.initialize_all_variables())
    steps = 0
    # generator for epoch data
    for idx, epoch in enumerate(batch_gen.gen_epochs()):
        training_state = None
        # generator for batch data
        for f_data, b_data, r_data, v_data in epoch:
            start_time = time.time()
            steps += 1
            in_data_np = np.concatenate((f_data, b_data), axis=2)
            if steps % 100 == 0:
                feed_dict = {in_data: in_data_np, ref_data: r_data}
                if training_state is not None:
                    feed_dict[init_state] = training_state
                # training the net
                loss_value, training_state, _, summary_str, test_inf = sess.run(
                    [loss, final_state, train_op, summary_op, inf_data], feed_dict)
                duration = time.time() - start_time
                sec_per_batch = float(duration)
                examples_per_sec = batch_size / duration
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch, epoch %d)')
                print (format_str % (datetime.now(), steps, loss_value,
                                     examples_per_sec, sec_per_batch,
                                     idx))
                summary_writer.add_summary(summary_str, steps)
            else:
                feed_dict = {in_data: in_data_np, ref_data: r_data}
                if training_state is not None:
                    feed_dict[init_state] = training_state

                loss_value, training_state, _ = sess.run(
                    [loss, final_state, train_op], feed_dict)
            if steps % 10000 == 0:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=steps)
