'''
Script for evluating a test sample using trained model
'''
import tensorflow as tf
import numpy as np
import ipdb
from numpy.lib import stride_tricks
import librosa
from model import MultiRnn


def stft(sig, frameSize, overlapFac=0.75, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    # samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


N_IN = 1  # input frames
cell_type = 'NL_LSTM'
N_OUT = 1  # output frames
batch_size = 1  # only 1 sample per batch for testing
num_steps = 1  # output 1 sample during every iteration
NEFF = 129  # effective FFT points
NFFT = 256  # number of FFT points
output_size = NEFF
Overlap = 0.75  # overlap factor of frames
NMOVE = (1 - Overlap) * NFFT

# dir for forward beamformer audio file
audiof_dir = '/media/nca/data/raw_data_multi/audio_test/10_f.wav'
# audiof_dir = '/home/zhr/remotedata/raw_data_multi/audio_test/800_f.wav'
# audiof_dir = 'ex_sample/test_0e.wav'

# dir for backward beamformer audio file
audiob_dir = '/media/nca/data/raw_data_multi/audio_test/10_b.wav'
# audiob_dir = '/home/zhr/remotedata/raw_data_multi/audio_test/800_b.wav'
# audiob_dir = 'ex_sample/test_180e.wav'

# dir to write inference
out_audio_dir = 'ex_sample/testfun.wav'
in_audiof, _ = librosa.load(audiof_dir, sr=None)

# load and transform the data
# in_audiof = np.random.randn(in_audiof.shape[0]) * 1e-5 + in_audiof
in_audiob, _ = librosa.load(audiob_dir, sr=None)
# in_audiob = null=Truep.random.randn(in_audiob.shape[0]) * 1e-5 + in_audiob

# transform the data to frequency domain
inf_stft = stft(in_audiof, NFFT, Overlap)
inb_stft = stft(in_audiob, NFFT, Overlap)

inf_stft_amp = np.abs(inf_stft)
inf_stft_amp = np.maximum(np.abs(inf_stft_amp),
                          np.max(np.abs(inf_stft_amp)) / 10000)

# take log of the spectrum
inf_data = 20. * np.log10(inf_stft_amp / 1e-4)
inf_data = np.maximum(inf_data, 0)
inf_data = np.round(inf_data)
phase_data = inf_stft / inf_stft_amp

# print(np.mean(inf_data))
# print(np.var(inf_data))

# same operation for the backward beamformer data
inb_stft_amp = np.abs(inb_stft)
inb_stft_amp = np.maximum(np.abs(inb_stft_amp),
                          np.max(np.abs(inb_stft_amp)) / 10000)
inb_data = 20. * np.log10(inb_stft_amp / 1e-4)
inb_data = np.maximum(inb_data, 0)
inb_data = np.round(inb_data)

print(np.mean(inb_data))
print(np.var(inb_data))

# ipdb.set_trace()
data_len = inf_data.shape[0]
assert NEFF == inb_data.shape[1], 'Uncompatible image height'
out_len = data_len - N_IN + 1
out_audio = np.zeros(shape=[(out_len - 1) * NMOVE + NFFT])

# with tf.Graph().as_default():

# build the computational graph
# the model
rnn_model = MultiRnn(
    cell_type, 256, output_size,
    batch_size, 3, 0, num_steps)

# input log spectrum placeholder
in_data = tf.placeholder(
    tf.float32, [batch_size, num_steps, 2 * NEFF])
# not useful during test
ref_data = tf.placeholder(
    tf.float32, [batch_size, num_steps, NEFF])

# make inference
init_state, final_state, infer_data = rnn_model.inference(in_data)

saver = tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
    training_state = None
    # restore the model
    saver.restore(sess, 'ckpt/model.ckpt-2230000')
    print("Model restored")
    i = 0
    while(i < out_len):
        if i % 100 == 0:
            print('frame num: %d' % (i))

        # prewhitening
        data_mean = 44
        data_stdv = 15.5
        feed_in_dataf = (inf_data[i][:] - data_mean) / data_stdv
        feed_in_datab = (inb_data[i][:] - data_mean) / data_stdv
        feed_in_data = np.concatenate(
            (feed_in_dataf, feed_in_datab), axis=0)
        feed_in_data.shape = (1, 1, 2 * NEFF)
        feed_dict = {in_data: feed_in_data}
        if training_state is not None:
            feed_dict[init_state] = training_state

        # run the graph and compute output
        inf_frame, training_state = sess.run(
            [infer_data, final_state], feed_dict)
        # ipdb.set_trace()

        # overlap and save to rebuild the waveform
        out_data = inf_frame[0, 0, :] * data_stdv + data_mean
        out_amp_tmp = 10 ** ((out_data) / 20.0) / 10000.0
        out_stft = out_amp_tmp * phase_data[i + N_IN - 1][:]
        out_stft.shape = (NEFF, )
        con_data = out_stft[-2:0:-1].conjugate()
        out_amp = np.concatenate((out_stft, con_data))
        frame_out_tmp = np.fft.ifft(out_amp).astype(np.float64)
        # frame_out_tmp = (frame_out_tmp * data_stdv + data_mean)
        out_audio[i * NMOVE: i * NMOVE + NFFT] += frame_out_tmp * 0.5016
        # ipdb.set_trace()
        i = i + 1
    # length = img.shape[]

# ipdb.set_trace()
librosa.output.write_wav('beam_0.wav', in_audiof, 16000)

librosa.output.write_wav(out_audio_dir, out_audio, 16000)
