import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import os
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

def compute_delta(vector):
    delta = np.zeros_like(vector)
    reg = LinearRegression()
    for i in range(vector.shape[0]):
        context_frames = np.arange(i - 4, i + 5)
        context_frames = np.clip(context_frames, 0, 9)
        context_frames = context_frames.reshape(-1, 1)
        reg.fit(context_frames, vector[context_frames].squeeze())
        slope = reg.coef_
        delta[i] = slope.squeeze()
    return delta

class sound_wave():
    def __init__(self, wave_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.sample_rate, self.signal = scipy.io.wavfile.read(script_dir + wave_file) 
        self.NFFT = 512
        self.output_point = self.NFFT / 2 + 1
        self.low_mel = 0
        self.high_mel = (1127 * np.log(1 + (self.sample_rate / 2) / 700))
        self.filts_num = 40
        self.H_mel = np.zeros((self.filts_num, int(np.floor(self.output_point))))
        self.original_signal = self.signal[0:int(3.5 * self.sample_rate)]
        self.emphasized_signal = np.empty(0)
        self.frame_signal = []
        self.pow_frames = []
        self.filter_frames = []
        self.mfcc = []
        self.delta_1 = np.empty(0)
        self.energy_1 = np.empty(0)
        self.delta_2 = np.empty(0)
        self.energy_2 = np.empty(0)
        self.features = np.empty(0)


    def pre_emphasis(self):
        pre_emphasis = 0.97
        self.emphasized_signal = np.append(self.original_signal[0], self.original_signal[1:] - pre_emphasis * self.original_signal[:-1])

    def frame_division(self):
        frame_size = 0.025
        frame_sample_num = int(frame_size * self.sample_rate)
        frame_stride = 0.1
        frame_stride_sample_num = int(frame_stride * self.sample_rate)
        frame_num = math.ceil((len(self.emphasized_signal) - frame_sample_num) / frame_stride_sample_num)
        padding_size = int(frame_num * frame_stride_sample_num + frame_sample_num - len(self.emphasized_signal))
        padding_signal = np.append(self.emphasized_signal, np.zeros(padding_size))
        for i in range(frame_num):
            self.frame_signal.append(padding_signal[frame_stride_sample_num * i: frame_stride_sample_num * i + frame_sample_num])

    def hamming_window(self):
        for i in range(len(self.frame_signal)):
            self.frame_signal[i] *= np.hamming(len(self.frame_signal[i]))
    
    def SIFT(self):
        mag_frames = np.absolute(np.fft.rfft(self.frame_signal, self.NFFT))  # Magnitude of the FFT
        self.pow_frames = (1.0 / self.NFFT) * (mag_frames ** 2)

    def Mel_Filter_Bank(self):
        mel_points = (700 * (np.exp(np.linspace(self.low_mel, self.high_mel, self.filts_num + 2) / 1127) - 1))
        bin_prequency = np.floor((self.NFFT + 1) * mel_points / self.sample_rate)
        for i in range(self.filts_num):
            for j in range(int(bin_prequency[i]), int(bin_prequency[i + 1])):
                self.H_mel[i][j] = 2 * (j - bin_prequency[i]) / ((bin_prequency[i + 2] - bin_prequency[i]) * (bin_prequency[i + 1] - bin_prequency[i]))
            for j in range(int(bin_prequency[i + 1]), int(bin_prequency[i + 2] + 1)):
                self.H_mel[i][j] = 2 * (bin_prequency[i + 2] - j) / ((bin_prequency[i + 2] - bin_prequency[i]) * (bin_prequency[i + 1] - bin_prequency[i]))

    def log_power(self):
        self.filter_frames = np.dot(self.pow_frames, self.H_mel.T)
        self.filter_frames = np.where(self.filter_frames == 0, np.finfo(float).eps, self.filter_frames)
        self.filter_frames = 20 * np.log10(self.filter_frames)
        self.filter_frames -= (np.mean(self.filter_frames, axis=0) + 1e-8)

    def DCT(self):
        self.mfcc = dct(self.filter_frames, type=2, axis=1, norm='ortho')[:, 1 : 13]
        (frames_num, coeff_num) = self.mfcc.shape
        lift = 1 + (22 / 2) * np.sin(np.pi * np.arange(coeff_num) / 22)
        self.mfcc *= lift
        self.mfcc -= (np.mean(self.mfcc, axis=0) + 1e-8)

    def hstack_features(self):
        mfcc_delta_1 = compute_delta(self.mfcc)
        mfcc_delta_2 = compute_delta(mfcc_delta_1)
        energy = np.sum(self.pow_frames, axis=1, keepdims=True)
        energy = np.log(energy)
        energy_delta_1 = compute_delta(energy)
        energy_delta_2 = compute_delta(energy_delta_1)
        self.features = np.hstack([self.mfcc, energy, mfcc_delta_1, energy_delta_1, mfcc_delta_2, energy_delta_2])
        
    def normalization(self):
        self.features -= np.mean(self.features, axis=0)
        self.features /= np.std(self.features, axis=0)
        

    def print_picture(self):
        plt.figure(figsize=(11,7), dpi=500)
        plt.subplot(211)
        plt.imshow(np.flipud(self.filter_frames.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,self.filter_frames.shape[1],0,self.filter_frames.shape[0]]) #画热力图
        plt.title("MFCC")

        plt.subplot(212)
        plt.imshow(np.flipud(self.mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0,self.mfcc.shape[0],0,self.mfcc.shape[1]])#热力图
        plt.title("MFCC")

        plt.savefig('mfcc_04.png')

    def get_features(self):
        self.pre_emphasis()
        self.frame_division()
        self.hamming_window()
        self.SIFT()
        self.Mel_Filter_Bank()
        self.log_power()
        self.DCT()
        self.print_picture()
        self.hstack_features()
        self.normalization()
        return self.features


sound = sound_wave('/OSR_us_000_0040_8k.wav')
features = sound.get_features()
print(features.shape)
