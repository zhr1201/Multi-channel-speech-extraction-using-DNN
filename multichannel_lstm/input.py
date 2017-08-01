'''
Class BatchGenerator:
    read data set from pkl files and feed them
    to the LSTM network
'''
import numpy as np
import cPickle as pickle


class BatchGenerator(object):
    def __init__(self, data_dir, batch_size, num_steps, epochs):
        '''data_dir: dir to store the .pkl
           num_steps: time steps used for unfolding
                      RNN'''
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epochs = epochs
        # prewhitening params
        self.mean = 44
        self.stddev2 = 1.0 / 15.5

        with open(data_dir, 'r') as f:
            # load forward beamformer spectrogram
            self.for_data = pickle.load(f)
            (self.length, self.NEFF) = self.for_data.shape
            self.batch_partition_length = self.length // self.batch_size
            self.epoch_size = self.batch_partition_length // self.num_steps
            self.for_data = np.reshape(
                self.for_data,
                [self.batch_size, self.epoch_size, self.num_steps, self.NEFF])
            self.for_data = np.transpose(self.for_data, [1, 0, 2, 3])
            # load backward beamformer spectrogram
            self.back_data = pickle.load(f)
            self.back_data = np.reshape(
                self.back_data,
                [self.batch_size, self.epoch_size, self.num_steps, self.NEFF])
            self.back_data = np.transpose(self.back_data, [1, 0, 2, 3])
            # load reference data spectrogram
            self.ref_data = pickle.load(f)
            self.ref_data = np.reshape(
                self.ref_data,
                [self.batch_size, self.epoch_size, self.num_steps, self.NEFF])
            self.ref_data = np.transpose(self.ref_data, [1, 0, 2, 3])
            # load VAD spectrogram
            self.vad_data = pickle.load(f)
            self.vad_data = np.reshape(
                self.vad_data,
                [self.batch_size, self.epoch_size, self.num_steps, self.NEFF])
            self.vad_data = np.transpose(self.vad_data, [1, 0, 2, 3])

    def gen_batch(self):
        '''Generate a batch of data'''
        for i in range(self.epoch_size):
            f_batch = self.for_data[i, :, :, :]
            b_batch = self.back_data[i, :, :, :]
            r_batch = self.ref_data[i, :, :, :]
            v_batch = self.vad_data[i, :, :, :]
            f_batch = (f_batch.astype('float32') - self.mean) * self.stddev2
            b_batch = (b_batch.astype('float32') - self.mean) * self.stddev2
            r_batch = (r_batch.astype('float32') - self.mean) * self.stddev2
            v_batch = v_batch.astype('float32')
            yield f_batch, b_batch, r_batch, v_batch

    def gen_epochs(self):
        '''Generate a epoch of data'''
        for i in range(self.epochs):
            yield self.gen_batch()
