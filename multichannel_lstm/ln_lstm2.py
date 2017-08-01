'''
Class BLayerNormalizedLSTMCell:
    Two parallele Layer Norm LSTM cell blocks that
    share the same weights for each channel
Class LayerNormalizedLSTMCell:
    adapted from BasicLSTMCell to use Layer Norm
'''
import numpy as np
import tensorflow as tf
import ipdb
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


def _blinear(args, args2, output_size, bias, bias_start=0.0):
    '''Apply _linear ops to the two parallele layers with same
    wights'''
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError(
                "linear expects shape[1] to be provided for shape %s, "
                "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value
    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            'weight', [total_arg_size, output_size / 2], dtype=dtype)
        # apply weights
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
            res2 = math_ops.matmul(args2[0], weights)
        else:
            # ipdb.set_trace()
            res = math_ops.matmul(array_ops.concat(1, args), weights)
            res2 = math_ops.matmul(array_ops.concat(1, args2), weights)
        if not bias:
            return res, res2
        # apply bias
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                'bias', [output_size] / 2,
                dtype=dtype,
                initializer=init_ops.constant_initializer(
                    bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases), nn_ops.bias_add(res2, biases)


def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift


class BLayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Two parallele Layer Norm LSTM cell blocks that
    share the same weights for each channel
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        '''Add layer norm to the two channels'''
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            n_length = c.get_shape()[1]
            i_length = inputs.get_shape()[1]

            # import ipdb; ipdb.set_trace()
            c1 = c[:, 0:n_length / 2]
            c2 = c[:, n_length / 2:n_length]
            h1 = h[:, 0:n_length / 2]
            h2 = h[:, n_length / 2:n_length]
            inputs1 = inputs[:, 0:i_length / 2]
            inputs2 = inputs[:, i_length / 2:i_length]

            # change bias argument to False since LN will add bias via shift
            concat1, concat2 = _blinear(
                [inputs1, h1], [inputs2, h2], 4 * self._num_units, False)
            i1, j1, f1, o1 = tf.split(1, 4, concat1)
            i2, j2, f2, o2 = tf.split(1, 4, concat2)

            # add layer normalization to each gate
            with tf.variable_scope('l1') as scope:
                i1 = ln(i1, scope='i/')
                j1 = ln(j1, scope='j/')
                f1 = ln(f1, scope='f/')
                o1 = ln(o1, scope='o/')
                # ipdb.set_trace()
                new_c1 = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                          tf.nn.sigmoid(i1) * self._activation(j1))
                new_h1 = self._activation(
                    ln(new_c1, scope='new_h/')) * tf.nn.sigmoid(o1)
            with tf.variable_scope('l2') as scope:
                i2 = ln(i2, scope='i/')
                j2 = ln(j2, scope='j/')
                f2 = ln(f2, scope='f/')
                o2 = ln(o2, scope='o/')
                new_c2 = (c2 * tf.nn.sigmoid(f2 + self._forget_bias) +
                          tf.nn.sigmoid(i2) * self._activation(j2))
                new_h2 = self._activation(
                    ln(new_c2, scope='new_h/')) * tf.nn.sigmoid(o2)

            # add layer_normalization in calculation of new hidden state

            new_c = tf.concat(1, (new_c1, new_c2))
            new_h = tf.concat(1, (new_h1, new_h2))

            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state


class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = tf.nn.rnn_cell._linear(
                [inputs, h], 4 * self._num_units, False)
            # ipdb.set_trace()

            i, j, f, o = tf.split(1, 4, concat)

            # add layer normalization to each gate
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f = ln(f, scope='f/')
            o = ln(o, scope='o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) +
                     tf.nn.sigmoid(i) * self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(
                ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state
