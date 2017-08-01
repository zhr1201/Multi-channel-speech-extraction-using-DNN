# Multi-channel-speech-extraction-using-DNN
A tensorflow implementation of my paper Combining beamforming and deep neural networks for multi-channel speech extraction
presented at InterNoise 2017 (not published yet) and a manuscript can be found at
https://github.com/zhr1201/Multi-channel-speech-extraction-using-DNN/blob/master/Manuscript-InterNoise2017-ZhouHaoran_0525.pdf
## Samples
  * array_0_180.wav five element array raw signal
  * test_0.wav forward beamformer output
  * test_180.wav backward beamformer output
  * ex_cnn_0.wav the sample enhanced using CNN
  * ex_lstm_0.wav the sample enhanced using LSTM
  Note that the test_0.wav and test_180.wav is enhanced using single channel enhancement and cut several secs in the begining
  so they are not aligned for the neural nets. However, you can use the 5 channel raw signal and design a forward and backward beamformer
  and then feed the net with the beamformer output to enhance the signal.

## Requirement
  * tensorflow r0.11
  * scipy
  * numpy
  * librosa
## File documentation
  ### CNN
    spectrogram.py: funtions to convert .wav files into spectrograms(.png).
    data_set_gen.py: transform the audio signals into spectrogram images(.png).
    img2bin.py: transform spectrogram images into binary files for tensorflow.
    SENN.py: define the structure of the network.
    SENN_input.py: define how to read the binary files into tensorflow queue.
    SENN_train.py: train the neural net.
    audio_val: use a forward beamformer and a backward beamformer ouput to infer clean speech using a trained model.
  ### LSTM
    SENN_input.py: read from .pkl files and provide batch data for training.
    ln_lstm.py: layer normalized lstm
    ln_lstm2.py: two parallele layer norm lstm cell blocks sharing the same weights
    model.py: define the neural net.
    train.py: train the net.
    audio_eval.py: use a forward beamformer and a backward beamformer ouput to infer clean speech using a trained model.

## Training procedure
  ### CNN
    1. Design your array and your foward and backward beamformer.
    2. Generate raw signal using image model(can be got free online) and generate samples for the nets.
       Name them like index+f.wav(forward beamformer output) index+b.wav (backward beamformer output) index+r.wav(reference signal)
    3. Use data_set_gen.py to transform them into specgtrograms and store them as png files.
    4. Use img2bin.py to transform the png files into binary files for the TF model.
    5. Run SENN_train.py to train the model.
    6. Use audio_eval.py to make clean spectrum inference given a forward beamformer wav and a backward beamformer wav using
       a trained model.
  ### LSTM
    1. Design your array and your foward and backward beamformer.
    2. Generate raw signal using image model(can be got free online) and generate samples for the nets.
       Name them like index+f.wav(forward beamformer output) index+b.wav (backward beamformer output) index+r.wav(reference signal)
    3. Dump the data into .pkl file format.
    4. Run SENN_train.py to train the model.
    6. Use audio_eval.py to make clean spectrum inference given a forward beamformer wav and a backward beamformer wav using
       a trained model.

## Some other things
  There are some little differences betweem the implementation and the paper, eg. the VAD inference loss is not considered and etc.

   
