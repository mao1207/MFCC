import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import os
from matplotlib import pyplot as plt
import numpy as np
import math

# Load original signal
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_rate, signal = scipy.io.wavfile.read(script_dir + '/OSR_us_000_0010_8k.wav') 
original_signal = signal[0:int(3.5 * sample_rate)]

# Pre-emphasis
pre_emphasis = 0.97
emphasized_signal = np.append(original_signal[0], original_signal[1:] - pre_emphasis * original_signal[:-1])

# frame division and add hamming window
frame_size = 0.025
frame_sample_num = int(frame_size * sample_rate)
frame_stride = 0.1
frame_stride_sample_num = int(frame_stride * sample_rate)
frame_num = math.ceil((len(emphasized_signal) - frame_sample_num) / frame_stride_sample_num)
padding_size = int(frame_num * frame_stride_sample_num + frame_sample_num - len(emphasized_signal))
padding_signal = np.append(emphasized_signal, np.zeros(padding_size))
frame_signal = []
for i in range(frame_num):
    frame_signal.append(padding_signal[frame_stride_sample_num * i: frame_stride_sample_num * i + frame_sample_num])

# STFT
NFFT = 512
output_point = NFFT / 2 + 1
mag_frames = np.absolute(np.fft.rfft(frame_signal, NFFT))  # Magnitude of the FFT
pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

# Mel-filter bank
low_mel = 0
high_mel = (1127 * np.log(1 + (sample_rate / 2) / 700))
filts_num = 40
mel_points = (700 * (np.exp(np.linspace(low_mel, high_mel, filts_num + 2) / 1127) - 1))
bin_prequency = np.floor((NFFT + 1) * mel_points / sample_rate)
H_mel = np.zeros((filts_num, int(np.floor(output_point))))
for i in range(filts_num):
    for j in range(int(bin_prequency[i]), int(bin_prequency[i + 1])):
        H_mel[i][j] = 2 * (j - bin_prequency[i]) / ((bin_prequency[i + 2] - bin_prequency[i]) * (bin_prequency[i + 1] - bin_prequency[i]))
    for j in range(int(bin_prequency[i + 1]), int(bin_prequency[i + 2] + 1)):
        H_mel[i][j] = 2 * (bin_prequency[i + 2] - j) / ((bin_prequency[i + 2] - bin_prequency[i]) * (bin_prequency[i + 1] - bin_prequency[i]))

# Log power
filter_frames = np.dot(pow_frames, H_mel.T)
filter_frames = np.where(filter_frames == 0, np.finfo(float).eps, filter_frames)
filter_frames = 20 * np.log10(filter_frames)
filter_frames -= (np.mean(filter_frames, axis=0) + 1e-8)

# DCT
mfcc = dct(filter_frames, type=2, axis=1, norm='ortho')[:, 1 : 13]
(frames_num, coeff_num) = mfcc.shape
lift = 1 + (22 / 2) * np.sin(np.pi * np.arange(coeff_num) / 22)
mfcc *= lift
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

plt.figure(figsize=(11,7), dpi=500)

plt.subplot(211)
plt.imshow(np.flipud(filter_frames.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,filter_frames.shape[1],0,filter_frames.shape[0]]) #画热力图
plt.title("MFCC")

plt.subplot(212)
plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,mfcc.shape[0],0,mfcc.shape[1]])#热力图
plt.title("MFCC")

plt.savefig('mfcc_04.png')
