'''
Transform the audio signals into spectrogram images(.png)
'''
from spectrogram import stft
import os
from PIL import Image
import numpy as np
import librosa
import ipdb
raw_data_dir = '/media/nca/data/raw_data_multi/audio_train'
out_data_dir = '/media/nca/data/raw_data_multi/train_bin_lstm'
tot_N = 4620
tot_s_count = 0


def transform(audio_data, save_image_path, nFFT=256, overlap=0.75):
    '''audio_data: signals to convert
    save_image_path: path to store the image file'''
    # spectrogram
    freq_data = stft(audio_data, nFFT, overlap)
    freq_data = np.maximum(np.abs(freq_data),
                           np.max(np.abs(freq_data)) / 10000)
    log_freq_data = 20. * np.log10(freq_data / 1e-4)
    N_samples = log_freq_data.shape[0]
    # log_freq_data = np.maximum(log_freq_data, max_m - 70)
    # print(np.max(np.max(log_freq_data)))
    # print(np.min(np.min(log_freq_data)))
    log_freq_data = np.round(log_freq_data)
    log_freq_data = np.transpose(log_freq_data)
    # ipdb.set_trace()

    assert np.max(np.max(log_freq_data)) < 256, 'spectrogram value too large'
    # save the image
    spec_imag = Image.fromarray(log_freq_data)
    spec_imag = spec_imag.convert('RGB')
    spec_imag.save(save_image_path)
    return N_samples


''' Audios to be transformed should have the following file name
number_f.wav: forward beamformer output audio
number_b.wav: backward beamformer output audio
number_r.wav: reference beamformer output audio
'''
for i in range(1, tot_N + 1):
    backfix_list = ['_f', '_b', '_r']
    for backfix in backfix_list:
        print(str(i) + backfix + '.wav')
        in_audio_path = os.path.join(raw_data_dir, str(i) + backfix + '.wav')
        out_png_path = os.path.join(out_data_dir, str(i) + backfix + '.png')
        audio, _ = librosa.load(in_audio_path, sr=None)
        # transform and save the image
        tot_s_count += transform(audio, out_png_path)
print('%d samples converted' % tot_s_count)
