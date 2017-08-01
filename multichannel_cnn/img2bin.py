'''
Transform spectrogram images into binary files for tensorflow
'''
import os
import numpy as np
from PIL import Image
import struct
import ipdb
from matplotlib import pyplot as plt

tot_N = 4620  # total number of files
num_in_points = 8  # input frames for the tensorflow model
num_out_points = 1  # output(enhanced) frames for the model
num_binfiles = 8  # number of generated binary files
tot_samples = 0  # total sample counter
data_dir = '/media/nca/data/raw_data_multi/train_bin_lstm'
out_dir = '/media/nca/data/raw_data_multi/train_bin_lstm'

''' file format number_f.png(forward beamformer output)
                number_b.png(backward beamformer output)
                number_r.png(reference signal for enhancement)'''

filelist = ['%d' % i for i in range(1, tot_N + 1)]
for i in range(num_binfiles):
    print('%d out of %d is converted' % (i, num_binfiles))
    file_dir = 'train%d.bin' % i
    file_path = os.path.join(out_dir, file_dir)
    with open(file_path, 'wb+') as fout:
        for file_prefix in filelist[int(
                tot_N / num_binfiles * i + 0.5):
                int(tot_N / num_binfiles * (i + 1) + 0.5)]:

            # feed forward beamformer data into binary file
            for_dir = os.path.join(data_dir, file_prefix + '_f.png')
            forfile = Image.open(
                for_dir).convert('L')
            for_data = np.asarray(forfile).transpose()

            # feed backward beamformer data into binary file
            back_dir = os.path.join(data_dir, file_prefix + '_b.png')
            backfile = Image.open(
                back_dir).convert('L')
            back_data = np.asarray(backfile).transpose()

            # feed reference data into binary file
            ref_dir = os.path.join(data_dir, file_prefix + '_r.png')
            reffile = Image.open(
                ref_dir).convert('L')
            ref_data = np.asarray(reffile).transpose()

            vad_info = ref_data > (np.max(ref_data) - 60)
            # ipdb.set_trace()
            total_length, nFFT = for_data.shape
            N_frames = total_length - num_in_points + 1
            tot_samples += N_frames

            for i in range(N_frames):
                for j in range(num_in_points):
                    for k in range(nFFT):
                        fout.write(struct.pack('B', for_data[i + j][k]))
                for j in range(num_in_points):
                    for k in range(nFFT):
                        fout.write(struct.pack('B', back_data[i + j][k]))
                for k in range(nFFT):
                        fout.write(struct.pack(
                            'B', ref_data[i + num_in_points - 1][k]))
                for k in range(nFFT):
                        fout.write(struct.pack(
                            'B', vad_info[i + num_in_points - 1][k]))


print('%d samples converted' % tot_samples)
