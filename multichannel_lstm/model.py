'''
Class: The multichannel speech enhancement lstm model
'''
from ln_lstm import LayerNormalizedLSTMCell
from ln_lstm2 import BLayerNormalizedLSTMCell
import numpy as np
import tensorflow as tf


class MultiRnn(object):
    '''The multichannel speech enhancement lstm model'''
    def __init__(self, cell_type, state_size, output_size, batch_size,
                 num_layers, learning_rate, num_steps):
        '''cell_type: this param doesn't matter cause we only use ln lstm
           state_size: hidden state size
           output_size: output frames, only 1 permitted
           num_layers: this param doesn't matter cause the net structure is
                       fixed after testing
        '''
        self.cell_type = cell_type
        self.state_size = state_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_steps = num_steps

    def inference(self, rnn_inputs):
        '''
        Take the batch of data and compute the output
        of the RNN using previous states
        '''
        # structure
        with tf.variable_scope('lstm1'):
            cell = BLayerNormalizedLSTMCell(self.state_size)
        with tf.variable_scope('lstm2'):
            cell2 = BLayerNormalizedLSTMCell(self.state_size)
        with tf.variable_scope('lstm3'):
            cell3 = BLayerNormalizedLSTMCell(self.state_size)
        with tf.variable_scope('lstm4'):
            cell4 = LayerNormalizedLSTMCell(self.state_size)
        with tf.variable_scope('lstm5'):
            cell5 = LayerNormalizedLSTMCell(self.state_size)

        # cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)

        # stack them
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell, cell2, cell3, cell4, cell5], state_is_tuple=True)

        # initial state
        init_state = cell.zero_state(self.batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, rnn_inputs, initial_state=init_state)

        # not soft max but fully connected layer
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.output_size])
            b = tf.get_variable(
                'b', [self.output_size],
                initializer=tf.constant_initializer(0.0))
            rnn_outputs = tf.reshape(rnn_outputs, [-1, self.state_size])
            final_outputs = tf.matmul(rnn_outputs, W) + b
        return init_state, final_state, tf.reshape(
            final_outputs, [-1, self.num_steps, self.output_size])

    def loss(self, infs, refs):
        '''Define loss'''
        loss = tf.nn.l2_loss(infs - refs) / self.batch_size / self.num_steps
        tf.scalar_summary('loss', loss)
        return loss

    def train(self, loss):
        '''Optimizer'''
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        optimizer = tf.train.AdamOptimizer()
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        train_op = optimizer.apply_gradients(
            zip(gradients, v))
        # train_op = optimizer.minimize(loss)
        return train_op
