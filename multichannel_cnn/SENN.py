'''
Class SE_NET:
The multichannel speech enhancement network
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import os
import ipdb
# import sys
import tensorflow as tf
import numpy as np
import SENN_input

log10_fac = 1 / np.log(10)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor
    (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tensor_name = var.op.name
        mean = tf.reduce_mean(var)
        tf.scalar_summary(tensor_name + 'mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary(tensor_name + 'stddev', stddev)
        tf.scalar_summary(tensor_name + 'max', tf.reduce_max(var))
        tf.scalar_summary(tensor_name + 'min', tf.reduce_min(var))
        tf.histogram_summary(tensor_name + 'histogram', var)


def conv2d(x, W):
    '''1 Dimentional convolution(only along 1 axis)
    The name is not accurate and we haven't changed that'''
    return tf.nn.conv2d(x, W, strides=[1, 100, 1, 1], padding='SAME')


def weight_variable(shape, regularizer):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial)
    return tf.get_variable(
        name='weight',
        shape=shape,
        initializer=tf.truncated_normal_initializer(
            stddev=0.1),
        regularizer=regularizer
    )


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def input(eval_data, data_dir, batch_size):
        return SENN_input.inputs(eval_data, data_dir, batch_size)


class SE_NET(object):
    """Multichannel speech enhancement network"""
    def __init__(
            self, data_dir, batch_size, NEFF, N_IN,
            N_OUT, reg_fac, DECAY=0.999):
        self.batch_size = batch_size
        self.NEFF = NEFF  # number of effective FFT points
        self.N_IN = N_IN  # number of frames fed to the net
        self.N_OUT = N_OUT  # number of frames output by the net
        self.DECAY = DECAY  # decay to estimate global statistics
        self.data_dir = data_dir  # Binary file dir
        self.reg_fac = reg_fac  # regularization factor

    def _batch_norm_wrapper(self, inputs, is_trianing, epsilon=1e-6):
        '''Wrap up all the ops for batch norm
        is_training == True: use batch statistics
        is_training == False: use gloabl statistics'''
        decay = self.DECAY
        scale = tf.Variable(tf.ones(inputs.get_shape()[-1]))
        beta = tf.Variable(tf.zeros(inputs.get_shape()[-1]))
        pop_mean = tf.Variable(
            tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(
            tf.ones([inputs.get_shape()[-1]]), trainable=False)
        if is_trianing:
            # update global statistics
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay +
                                   batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay +
                                  batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(
                    inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(
                inputs, pop_mean, pop_var, beta, scale, epsilon)

    def _conv_layer_wrapper(self,
                            input, out_feature_maps, filter_length, is_train):
        '''Wrap up all the ops for a convolution layer'''
        # equals to kernel size: filter_width * 1
        filter_width = input.get_shape()[1].value
        in_feature_maps = input.get_shape()[-1].value
        W_conv = weight_variable(
            [filter_width, filter_length, in_feature_maps, out_feature_maps],
            regularizer=tf.contrib.layers.l2_regularizer(self.reg_fac))
        b_conv = bias_variable([out_feature_maps])
        h_conv_t = tf.add(conv2d(input, W_conv), b_conv)
        h_conv_b = self._batch_norm_wrapper(h_conv_t, is_train)
        return tf.nn.relu(h_conv_b)

    def _double_conv_layer_wrapper(self, input1, input2, out_feature_maps,
                                   filter_length, is_train):
        '''Two parallele convolution layers for each channel
        using shared weights'''
        filter_width = input1.get_shape()[1].value
        in_feature_maps = input1.get_shape()[-1].value
        # shared weights
        W_conv = weight_variable(
            [filter_width, filter_length, in_feature_maps, out_feature_maps],
            regularizer=tf.contrib.layers.l2_regularizer(self.reg_fac))
        # shared bias
        b_conv = bias_variable([out_feature_maps])
        h_conv_t1 = tf.add(conv2d(input1, W_conv), b_conv)
        h_conv_b1 = self._batch_norm_wrapper(h_conv_t1, is_train)
        h_conv_t2 = tf.add(conv2d(input2, W_conv), b_conv)
        h_conv_b2 = self._batch_norm_wrapper(h_conv_t2, is_train)
        return tf.nn.relu(h_conv_b1), tf.nn.relu(h_conv_b2)

    def inference(self, imagefs, imagebs, is_train):
        '''Structure of the net'''
        imagef_input = tf.reshape(imagefs, [-1, self.N_IN, self.NEFF, 1])
        imageb_input = tf.reshape(imagebs, [-1, self.N_IN, self.NEFF, 1])
        # ipdb.set_trace()
        # convolutional layers
        with tf.variable_scope('con1') as scope:
            h_conv1f, h_conv1b = self._double_conv_layer_wrapper(
                imagef_input, imageb_input, 12, 13, is_train)
        with tf.variable_scope('con2') as scope:
            h_conv2f, h_conv2b = self._double_conv_layer_wrapper(
                h_conv1f, h_conv1b, 16, 11, is_train)
        with tf.variable_scope('con3') as scope:
            h_conv3f, h_conv3b = self._double_conv_layer_wrapper(
                h_conv2f, h_conv2b, 20, 9, is_train)
        with tf.variable_scope('con4') as scope:
            h_conv4f, h_conv4b = self._double_conv_layer_wrapper(
                h_conv3f, h_conv3b, 24, 7, is_train)
        with tf.variable_scope('con5') as scope:
            i_conv5 = tf.concat(1, [h_conv4f, h_conv4b])
            h_conv5 = self._conv_layer_wrapper(i_conv5, 32, 7, is_train)
        with tf.variable_scope('con6') as scope:
            h_conv6 = self._conv_layer_wrapper(h_conv5, 24, 7, is_train)
        with tf.variable_scope('con7') as scope:
            h_conv7 = self._conv_layer_wrapper(h_conv6, 20, 9, is_train)
        with tf.variable_scope('con8') as scope:
            h_conv8 = self._conv_layer_wrapper(h_conv7, 16, 11, is_train)
        with tf.variable_scope('con9') as scope:
            h_conv9 = self._conv_layer_wrapper(h_conv8, 12, 13, is_train)

        # Two output heads: 1 for speech inference 2 for bin-wise VAD
        with tf.variable_scope('fc1') as scope:
            reshape = tf.reshape(h_conv9, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            W_fc1 = weight_variable(
                [dim, self.NEFF],
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_fac))
            b_fc1 = bias_variable([self.NEFF])
            fc1 = tf.add(tf.matmul(reshape, W_fc1), b_fc1)
        with tf.variable_scope('fc2') as scope:
            W_fc2 = weight_variable(
                [dim, self.NEFF],
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_fac))
            b_fc2 = bias_variable([self.NEFF])
            fc2 = tf.nn.sigmoid(tf.add(tf.matmul(reshape, W_fc2), b_fc2))
        # ipdb.set_trace()
        return tf.reshape(
            fc1, [-1, self.NEFF]), tf.reshape(fc2, [-1, self.NEFF])

    # def inference(self, images, is_train):
    #     """Just use the last frame data"""
    #     with tf.variable_scope('fc1') as scope:
    #         image_input = tf.reshape(images, [-1, self.N_IN, self.NEFF, 1])
    #         # ipdb.set_trace()
    #         sel = tf.transpose(image_input, [1, 0, 2, 3])
    #         sel2 = sel[7][:][:][:]
    #         rst = tf.reshape(sel2, [-1, self.NEFF])
    #     return rst

    # def inference(self, images, is_train):
    #     image_input = tf.reshape(images, [-1, self.N_IN, self.NEFF, 1])
    #     # ipdb.set_trace()
    #     with tf.variable_scope('con1') as scope:
    #         h_conv1 = self._conv_layer_wrapper(image_input, 12, 13, is_train)
    #     with tf.variable_scope('con2') as scope:
    #         h_conv2 = self._conv_layer_wrapper(h_conv1, 16, 11, is_train)
    #     with tf.variable_scope('con3') as scope:
    #         h_conv3 = self._conv_layer_wrapper(h_conv2, 20, 9, is_train)
    #     with tf.variable_scope('con4') as scope:
    #         h_conv4 = self._conv_layer_wrapper(h_conv3, 16, 11, is_train)
    #     with tf.variable_scope('con5') as scope:
    #         h_conv5 = self._conv_layer_wrapper(h_conv4, 12, 9, is_train)
    #     with tf.variable_scope('con6') as scope:
    #         h_conv6 = self._conv_layer_wrapper(h_conv5, 5, 13, is_train)
    #     # with tf.variable_scope('con7') as scope:
    #     #     h_conv7 = self._conv_layer_wrapper(h_conv6, 1, 129, is_train)
    #     with tf.variable_scope('fc1') as scope:
    #         reshape = tf.reshape(h_conv6, [self.batch_size, -1])
    #         dim = reshape.get_shape()[1].value
    #         W_fc1 = weight_variable([dim, self.NEFF])
    #         b_fc1 = bias_variable([self.NEFF])
    #         fc1 = tf.matmul(reshape, W_fc1) + b_fc1
    #     return tf.reshape(fc1, [-1, self.NEFF])

    def loss(self, inf_targets, inf_vads, targets, vads, mtl_fac):
        '''
        Loss definition
        Only speech inference loss is defined and work quite well
        Add VAD cross entropy loss if you want
        '''
        loss_v1 = tf.nn.l2_loss(inf_targets - targets) / self.batch_size
        loss_o = loss_v1
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # ipdb.set_trace()
        loss_v = loss_o + tf.add_n(reg_loss)
        tf.scalar_summary('loss', loss_v)
        # loss_merge = tf.cond(
        #     is_val, lambda: tf.scalar_summary('val_loss_batch', loss_v),
        #     lambda: tf.scalar_summary('loss', loss_v))
        return loss_v, loss_o
        # return tf.reduce_mean(tf.nn.l2_loss(inf_targets - targets))

    def train(self, loss, lr):
        '''Optimizer'''
        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        # optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
        return train_op
