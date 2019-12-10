"""
Tests of the MFCC conversion in the data processor
"""

#from wavmdrnn import DataProcessor
# from wavmdrnn.mmse import MMSESTSA
import librosa
import numpy as np

num_mfcc = 50
SAMPLE_RATE = 22050

def decode_MFCCs(mfccs, sr=SAMPLE_RATE):
    """
    Reconstruction of wav time-signal from MFCCs. Code taken from:
    https://amyang.xyz/posts/Inverse-MFCC-to-WAV
    """
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    # replace deprecated librosa.filters.dct: (algorithm taken from librosa 0.6.1)
    dct_m = np.empty((n_mfcc, n_mel))
    dct_m[0, :] = 1.0 / np.sqrt(n_mel)
    samples = np.arange(1, 2*n_mel, 2) * np.pi / (2.0 * n_mel)
    for i in range(1, n_mfcc):
        dct_m[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_mel)
    # dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(sr, n_fft)
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dct_m.T, mfccs)))
    shape_inv_recon = librosa.istft(recon_stft).shape[0]
    #excitation = np.random.randn(self.wav_shape)
    excitation = np.random.randn(shape_inv_recon)
    E = librosa.stft(excitation)
    
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    #recon = librosa.istft(np.sqrt(recon_stft))
    return recon

def invlogamplitude(S):
    """
    librosa.logamplitude is actually 10_log10, so invert that.
    Follows from decode_MFCCs
    """
    return 10.0**(S/10.0)

def reduce_noise(signal):
    """
    MMSE-STSA
    """
    output, saved_params = MMSESTSA(signal, SAMPLE_RATE)
    return output[:,0]


sample, sr = librosa.load("wavfiles/voice.wav")
mfccs = librosa.feature.mfcc(sample, n_mfcc=num_mfcc)
output = decode_MFCCs(mfccs)
# output = reduce_noise(output)
librosa.output.write_wav("voice-out.wav", output, SAMPLE_RATE, norm=True)

# #inverse transform to get 1D-sequence, also force upper bound
#out_sequence = self.data_processor.denorm(out_sequence)
#out_sequence = self.data_processor.decode_MFCCs(out_sequence)
#out_sequence = self.data_processor.reduce_noise(out_sequence)
#print(out_sequence.shape, np.max(abs(out_sequence)))
#out_sequence = np.where(abs(out_sequence) > 1.0, 1.0, out_sequence)
#print(out_sequence.shape, np.max(abs(out_sequence)))
#librosa.output.write_wav(self.base_dir + "results/{}.wav".format(self.name),out_sequence, self.data_processor.sr, norm=True)
