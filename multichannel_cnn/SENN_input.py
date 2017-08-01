'''
Tensorflow reader for reading the binary files
'''
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import ipdb

NFFT = 256  # number of FFT points
NEFF = NFFT / 2 + 1  # number of effective FFT points
N_IN = 8  # number of frames fed to the CNN
N_OUT = 1  # number of output frames of the CNN
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3959151
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 693959


def read_SENN(filename_queue):
    class dataRecord():
        pass

    result = dataRecord()
    image_bytes = NEFF * N_IN
    target_bytes = NEFF * N_OUT
    # 1 image_bytes for forward beamformer
    # 1 image_bytes for backward beamformer
    # 1 target_bytes for reference signal
    # 1 target_bytes for reference VAD
    record_bytes = image_bytes * 2 + target_bytes * 2
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    # forward beamformer
    result.imagef = tf.reshape(tf.slice(record_bytes, [0], [image_bytes]),
                               [N_IN, NEFF])
    # backward beamformer
    result.imageb = tf.reshape(tf.slice(
        record_bytes, [image_bytes], [image_bytes]), [N_IN, NEFF])
    # reference speech
    result.target = tf.reshape(tf.slice(
        record_bytes, [2 * image_bytes], [target_bytes]),
        [NEFF])
    # reference VAD
    result.vad = tf.reshape(tf.slice(
        record_bytes, [2 * image_bytes + target_bytes], [target_bytes]),
        [NEFF])
    return result


# def _generate_image_and_label_batch(batch_data,
#                                     min_queue_examples, batch_size, shuffle):
#     # ipdb.set_trace()
#     if shuffle:
#         num_preprocessing_threads = 8
#         batch = tf.train.shuffle_batch(
#             [batch_data],
#             batch_size=batch_size,
#             num_threads=num_preprocessing_threads,
#             capacity=min_queue_examples + 10 * batch_size,
#             min_after_dequeue=min_queue_examples,
#             enqueue_many=True)
#     else:
#         num_preprocessing_threads = 2
#         batch = tf.train.batch(
#             [batch_data],
#             batch_size=batch_size,
#             num_threads=num_preprocessing_threads,
#             capacity=min_queue_examples + 5 * batch_size,
#             enqueue_many=True)
#     # ipdb.set_trace()
#     imagefs = batch[:, 0:N_IN, :]
#     imagebs = batch[:, N_IN:2 * N_IN, :]
#     targets = tf.reshape(batch[:, 2 * N_IN, :], [-1, NEFF])
#     vads = tf.reshape(batch[:, 2 * N_IN + 1, :], [-1, NEFF])
#     return imagefs, imagebs, targets, vads


# def inputs(eval_data, data_dir, batch_size):
#     if not eval_data:
#         filenames = [os.path.join(data_dir, 'train%d.bin' % i)
#                      for i in range(8)]
#         # ipdb.set_trace()
#         # filenames = ['datam.bin']
#         num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#         min_fraction_of_examples_in_queue = 0.04
#         min_queue_examples = int(num_examples_per_epoch *
#                                  min_fraction_of_examples_in_queue)
#     else:
#         # ipdb.set_trace()
#         filenames = [os.path.join(data_dir, 'train%d.bin' % i)
#                      for i in range(8)]
#         num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
#         min_fraction_of_examples_in_queue = 0
#         min_queue_examples = int(num_examples_per_epoch *
#                                  min_fraction_of_examples_in_queue)

#     for f in filenames:
#         if not tf.gfile.Exists(f):
#             raise ValueError('Failed to find file: ' + f)

#     filename_queue = tf.train.string_input_producer(filenames)
#     batch_data = []
#     for i in range(batch_size):
#         read_input = read_SENN(filename_queue)
#         reshaped_imagef = tf.cast(read_input.imagef, tf.float32)
#         reshaped_imageb = tf.cast(read_input.imageb, tf.float32)
#         reshaped_target = tf.cast(read_input.target, tf.float32)
#         reshaped_vad = tf.cast(read_input.vad, tf.float32)
#         mean = 44
#         stddev2 = 15.5
#         whitened_imagef = (reshaped_imagef - mean) / stddev2
#         whitened_imageb = (reshaped_imageb - mean) / stddev2
#         whitened_target = (reshaped_target - mean) / stddev2
#         # ipdb.set_trace()
#         batch_data.append(
#             tf.concat(
#                 0,
#                 [whitened_imagef, whitened_imageb,
#                  whitened_target, reshaped_vad]))

#     if not eval_data:
#         return _generate_image_and_label_batch(batch_data,
#                                                min_queue_examples,
#                                                batch_size,
#                                                shuffle=True)
#     else:
#         return _generate_image_and_label_batch(batch_data,
#                                                min_queue_examples,
#                                                batch_size,
#                                                shuffle=False)


def _generate_image_and_label_batch(imagef, imageb, target, vad,
                                    min_queue_examples, batch_size, shuffle):
    '''Using reader to generate data batches'''
    if shuffle:
        num_preprocessing_threads = 8
        imagefs, imagebs, targets, vads = tf.train.shuffle_batch(
            [imagef, imageb, target, vad],
            batch_size=batch_size,
            num_threads=num_preprocessing_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        num_preprocessing_threads = 1
        imagefs, imagebs, targets, vads = tf.train.batch(
            [imagef, imageb, target, vad],
            batch_size=batch_size,
            num_threads=num_preprocessing_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return imagefs, imagebs, targets, vads


def inputs(eval_data, data_dir, batch_size):
    '''Read inputs
    if eval_data True: use validation set
    else: use training set'''
    if not eval_data:
        # read training data
        filenames = [os.path.join(data_dir, 'train%d.bin' % i)
                     for i in range(8)]
        # ipdb.set_trace()
        # filenames = ['datam.bin']
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        # read validation data
        filenames = [os.path.join(data_dir, 'train%d.bin' % i)
                     for i in range(8)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_SENN(filename_queue)
    reshaped_imagef = tf.cast(read_input.imagef, tf.float32)
    reshaped_imageb = tf.cast(read_input.imageb, tf.float32)
    reshaped_target = tf.cast(read_input.target, tf.float32)
    reshaped_vad = tf.cast(read_input.vad, tf.float32)
    # ipdb.set_trace()
    # in_Data = tf.concat(0, [reshaped_imagef, reshaped_imageb])
    # mean = tf.reduce_mean(in_Data)
    # # ipdb.set_trace()
    # stddev = tf.sqrt(
    #     tf.reduce_mean(tf.square(in_Data - mean)))
    # stddev2 = tf.maximum(stddev, 0.000001)

    # normalize the data using global mean and variance
    mean = 44
    stddev2 = 15.5
    whitened_imagef = (reshaped_imagef - mean) / stddev2
    whitened_imageb = (reshaped_imageb - mean) / stddev2
    whitened_target = (reshaped_target - mean) / stddev2
    # whitened_imagef = reshaped_imagef / 255
    # whitened_imageb = reshaped_imageb / 255

    # whitened_target = reshaped_target / 255
    if not eval_data:
        min_fraction_of_examples_in_queue = 0.05
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        return _generate_image_and_label_batch(whitened_imagef,
                                               whitened_imageb,
                                               whitened_target,
                                               reshaped_vad,
                                               min_queue_examples,
                                               batch_size,
                                               shuffle=True)
    else:
        min_fraction_of_examples_in_queue = 0.0001
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        return _generate_image_and_label_batch(whitened_imagef,
                                               whitened_imageb,
                                               whitened_target,
                                               reshaped_vad,
                                               min_queue_examples,
                                               batch_size,
                                               shuffle=False)

    # return _generate_image_and_label_batch(reshaped_imagef,
    #                                        reshaped_imageb,
    #                                        reshaped_target,
    #                                        min_queue_examples,
    #                                        batch_size,
    #                                        shuffle=False)
