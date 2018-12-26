#!/usr/bin/python

from __future__ import division
import numpy as np
import math
from scipy.special import *
from numpy.matlib import repmat
from scipy.signal import lfilter
#from scikits.audiolab import Sndfile, Format
import argparse
import sys

"""
Code taken from:
https://github.com/braindead/mmse-port/blob/master/mmse.py?fbclid=IwAR0EmWzbweF6ihEOHvlAW9utklrgRfeNJQXAGnBm59CkR2A0BaX4V0YMmVg
"""


np.seterr('ignore')

def MMSESTSA(signal, fs, IS=0.25, W=1024, NoiseMargin=3, saved_params=None):
    SP = 0.4
    wnd = np.hamming(W)

    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y, axis=0)
    YPhase = np.angle(Y[0:int(np.fix(len(Y)/2))+1,:])
    Y = np.abs(Y[0:int(np.fix(len(Y)/2))+1,:])
    numberOfFrames = Y.shape[1]

    NoiseLength = 9
    NoiseCounter = 0
    alpha = 0.99

    NIS = int(np.fix(((IS * fs - W) / (SP * W) + 1)))
    N = np.mean(Y[:,0:NIS].T).T
    LambdaD = np.mean((Y[:,0:NIS].T) ** 2).T

    if saved_params != None:
        NIS = 0
        N = saved_params['N']
        LambdaD = saved_params['LambdaD']
        NoiseCounter = saved_params['NoiseCounter']

    G = np.ones(N.shape)
    Gamma = G

    Gamma1p5 = math.gamma(1.5)
    X = np.zeros(Y.shape)

    for i in range(numberOfFrames):
        Y_i = Y[:,i]

        if i < NIS:
            SpeechFlag = 0
            NoiseCounter = 100
        else:
            SpeechFlag, NoiseCounter = vad(Y_i, N, NoiseCounter, NoiseMargin)

        if SpeechFlag == 0:
            N = (NoiseLength * N + Y_i) / (NoiseLength + 1)
            LambdaD = (NoiseLength * LambdaD + (Y_i ** 2)) / (1 + NoiseLength)

        gammaNew = (Y_i ** 2) / LambdaD
        xi = alpha * (G ** 2) * Gamma + (1 - alpha) * np.maximum(gammaNew - 1, 0)

        Gamma = gammaNew
        nu = Gamma * xi / (1 + xi)

        # log MMSE algo
        #G = (xi/(1 + xi)) * np.exp(0.5 * expn(1, nu))

        # MMSE STSA algo
        G = (Gamma1p5 * np.sqrt(nu)) / Gamma * np.exp(-1 * nu / 2) * ((1 + nu) * bessel(0, nu / 2) + nu * bessel(1, nu / 2))
        Indx = np.isnan(G) | np.isinf(G)
        G[Indx] = xi[Indx] / (1 + xi[Indx])

        X[:,i] = G * Y_i

    output = OverlapAdd2(X, YPhase, W, SP * W)
    return output, {'N': N, 'LambdaD': LambdaD, 'NoiseCounter': NoiseCounter}

def OverlapAdd2(XNEW, yphase, windowLen, ShiftLen):
    FrameNum = XNEW.shape[1]
    Spec = XNEW * np.exp(1j * yphase)

    ShiftLen = int(np.fix(ShiftLen))

    if windowLen % 2:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:,]))))
    else:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:-1,:]))))

    sig = np.zeros(((FrameNum - 1) * ShiftLen + windowLen, 1)) 

    for i in range(FrameNum):
        start = i * ShiftLen
        spec = Spec[:,[i]]
        sig[start:start + windowLen] = sig[start:start + windowLen] + np.real(np.fft.ifft(spec, axis=0))

    return sig

def segment(signal, W, SP, Window):
    L = len(signal)
    SP = int(np.fix(W * SP))
    N = int(np.fix((L-W)/SP + 1))

    Window = Window.flatten(1)

    Index = (np.tile(np.arange(1,W+1), (N,1)) + np.tile(np.arange(0,N) * SP, (W,1)).T).T
    hw = np.tile(Window, (N, 1)).T
    Seg = signal[Index] * hw
    return Seg

def vad(signal, noise, NoiseCounter, NoiseMargin, Hangover = 8):
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist[SpectralDist < 0] = 0

    Dist = np.mean(SpectralDist)
    if (Dist < NoiseMargin):
        NoiseFlag = 1
        NoiseCounter = NoiseCounter + 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0

    if (NoiseCounter > Hangover):
        SpeechFlag=0
    else:
        SpeechFlag=1

    return SpeechFlag, NoiseCounter

def bessel(v, X):
    return ((1j**(-v))*jv(v,1j*X)).real

# main
"""
parser = argparse.ArgumentParser(description='Speech enhancement/noise reduction using Log MMSE STSA algorithm')
parser.add_argument('input_file', action='store', type=str, help='input file to clean')
parser.add_argument('output_file', action='store', type=str, help='output file to write (default: stdout)', default=sys.stdout)
parser.add_argument('-i, --initial-noise', action='store', type=float, dest='initial_noise', help='initial noise in ms (default: 0.1)', default=0.1)
parser.add_argument('-w, --window-size', action='store', type=int, dest='window_size', help='hamming window size (default: 1024)', default=1024)
parser.add_argument('-n, --noise-threshold', action='store', type=int, dest='noise_threshold', help='noise thresold (default: 3)', default=3)
args = parser.parse_args()

input_file = Sndfile(args.input_file, 'r')

fs = input_file.samplerate
num_frames = input_file.nframes

output_file = Sndfile(args.output_file, 'w', Format(type=input_file.file_format, encoding='pcm16', endianness=input_file.endianness), input_file.channels, fs)

chunk_size = int(np.fix(60*fs))
saved_params = None

frames_read = 0
while (frames_read < num_frames):
    frames = num_frames - frames_read if frames_read + chunk_size > num_frames else chunk_size
    signal = input_file.read_frames(frames)
    frames_read = frames_read + frames

    output, saved_params = MMSESTSA(signal, fs, args.initial_noise, args.window_size, args.noise_threshold, saved_params)

    output = np.array(output*np.iinfo(np.int16).max, dtype=np.int16)
    output_file.write_frames(output)

input_file.close()
output_file.close()
"""
