'''
Script for reading in a forward beamformer audio and
a backward beamformer audio and output an enhanced audio
'''
import tensorflow as tf
import numpy as np
import ipdb
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
import SENN
import librosa


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


# all the definition of the flowing variable can be found
# train_net.py
N_IN = 8
N_OUT = 1
NEFF = 129
NFFT = 256
Overlap = 0.75
NMOVE = (1 - Overlap) * NFFT

# dir for forward beamformer
audiof_dir = '/media/nca/data/raw_data_multi/audio_val/1_b.wav'
# audiof_dir = '/home/zhr/remotedata/raw_data_multi/audio_test/800_b.wav'
# audiof_dir = 'ex_sample/test_180e.wav'

# dir for backward beamformer
audiob_dir = '/media/nca/data/raw_data_multi/audio_val/1_f.wav'
# audiob_dir = '/home/zhr/remotedata/raw_data_multi/audio_test/800_f.wav'
# audiob_dir = 'ex_sample/test_0e.wav'

# dir for output audio
out_audio_dir = 'ex_sample/test180.wav'
in_audiof, _ = librosa.load(audiof_dir, sr=None)
# in_audiof = np.random.randn(in_audiof.shape[0]) * 1e-5 + in_audiof
in_audiob, _ = librosa.load(audiob_dir, sr=None)
# in_audiob = np.random.randn(in_audiob.shape[0]) * 1e-5 + in_audiob


# transform into frames of spectrograms
inf_stft = stft(in_audiof, NFFT, Overlap)  # forward
inb_stft = stft(in_audiob, NFFT, Overlap)  # backward

# forward: take log
inf_stft_amp = np.abs(inf_stft)
inf_stft_amp = np.maximum(np.abs(inf_stft_amp),
                          np.max(np.abs(inf_stft_amp)) / 10000)
inf_data = 20. * np.log10(inf_stft_amp / 1e-4)
inf_data = np.maximum(inf_data, 0)
inf_data = np.round(inf_data)
phase_data = inf_stft / inf_stft_amp

# backward: take log
inb_stft_amp = np.abs(inb_stft)
inb_stft_amp = np.maximum(np.abs(inb_stft_amp),
                          np.max(np.abs(inb_stft_amp)) / 10000)
inb_data = 20. * np.log10(inb_stft_amp / 1e-4)
inb_data = np.maximum(inb_data, 0)
inb_data = np.round(inb_data)

# print(np.mean(inb_data))
# print(np.var(inb_data))

# ipdb.set_trace()
data_len = inb_data.shape[0]
assert NEFF == inb_data.shape[1], 'Uncompatible image height'
out_len = data_len - N_IN + 1
out_audio = np.zeros(shape=[(out_len - 1) * NMOVE + NFFT])

# with tf.Graph().as_default():

'''construct the computational graph'''
# to be fed with forward spectrogram
imagesf = tf.placeholder(tf.float32, [N_IN, NEFF])

# to be fed with backward spectrogram
imagesb = tf.placeholder(tf.float32, [N_IN, NEFF])

SE_net = SENN.SE_NET('', 1, NEFF, N_IN, N_OUT, 1e-20)

inf_targets, inf_vads = SE_net.inference(
    imagesf, imagesb, is_train=False)

saver = tf.train.Saver(tf.all_variables())

with tf.Session() as sess:

    # restore the weights from a given dir
    saver.restore(sess, 'ckpt_2/model.ckpt-1531035')
    print("Model restored")
    i = 0
    while(i < out_len):
        if i % 100 == 0:
            print('frame num: %d' % (i))
        feed_in_data = np.concatenate(
            [inf_data[i:i + N_IN][:], inb_data[i:i + N_IN][:]])
        # data_mean = np.mean(feed_in_data)
        # data_mean = np.maximum(data_mean, 0.000001)
        # data_stdv = np.sqrt(np.var(feed_in_data))
        data_mean = 44
        data_stdv = 15.5
        feed_in_dataf = (inf_data[i:i + N_IN][:] - data_mean) / data_stdv
        feed_in_datab = (inb_data[i:i + N_IN][:] - data_mean) / data_stdv
        # ipdb.set_trace()
        # feed in the data and compute the inferred clean speech
        inf_frame, = sess.run(
            [inf_targets],
            feed_dict={imagesf: feed_in_dataf,
                       imagesb: feed_in_datab})

        # restore the time domain signal using overlap and add
        out_data = inf_frame * data_stdv + data_mean
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
librosa.output.write_wav(out_audio_dir, out_audio, 16000)
