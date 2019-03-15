import librosa
import librosa.display
import numpy as np
import mmse
import sys
import glob
import os


class DataProcessor:

    def __init__(self, wav_dir, n_mfcc=20):
        """
        num_files is max: 3081
        """
        os.chdir(wav_dir)
        file_names = glob.glob("*.wav")
        self.num_wav_files = len(file_names)
        self.wav_generator = self._wav_generator(wav_dir)
        # data = self.concatenate_wavs(num_files)
        data = np.array([])
        for i in range(self.num_wav_files):
            sample_i, sr = next(self.wav_generator)
            data = np.concatenate([data, sample_i])
        self.sr = sr
        self.wav_shape = data.shape[0]
        # data = self.reduce_noise(data)
        # self.MFCCs = self.encode_wav(data, n_mfcc)
        self.MFCCs = librosa.feature.mfcc(data, n_mfcc=n_mfcc)
        """
        self.MFCCs
        self.normalization_version
        self.sr
        self.wav_shape
        self.min_MFCCs
        self.max_MFCCs
        self.mean_MFCCs
        self.std_MFCCs
        """

    def _wav_generator(self, wav_dir, wav_names):
        """
        Generate wav-sequences

        output:
            wav, sr: the 1D-wav sequence, sr is the sampling rate
        """
        for f in wav_names:
            file_path = wav_dir + "/" + f
            yield librosa.load(file_path)
            # librosa gives resource warnings, but that is their backend problem
            # yield librosa.load("{}{}{}".format(temp, i, ".wav"))


    def choose_data_model(self, normalization_version=2, data_version=1, num_time_steps=101,
        k=1, percentile_test=0.1):
        """
        Choose normalization, data_version and num_time_steps
        """
        self.normalization_version = normalization_version
        norm_MFFCs = None
        #if version is not included, then no normalization
        if(normalization_version==1):
            norm_MFCCs = self.normalize_MFCCs()
        elif(normalization_version==2):
            norm_MFCCs = self.normalize_MFCCs_v2()
        elif(normalization_version==3):
            norm_MFCCs = self.normalize_MFCCs_v3()
        else:
            norm_MFCCs = self.MFCCs

        if(data_version == 1):
            self.create_input_target_test_data(norm_MFCCs, k, num_time_steps, percentile_test)
        elif(data_version == 2):
            self.create_input_target_test_data_v2(norm_MFCCs, k, num_time_steps, percentile_test)
        else:
            print("Duh shit!? How many version yo need bruh?")
            print("Exiting, coz u dont listen.")
            sys.exit(1)

    def denorm(self, pred_norm_MFCCs):
        """
        Denormalize the chosen normalization
        """
        if(self.normalization_version==1):
            return self.denormalize_MFCCs(pred_norm_MFCCs)
        elif(self.normalization_version==2):
            return self.denormalize_MFCCs_v2(pred_norm_MFCCs)
        elif(self.normalization_version==3):
            return self.denormalize_MFCCs_v3(pred_norm_MFCCs)
        else:
            return pred_norm_MFCCs

    def encode_wav(self, data, n_mfcc):
        """
        Encode (and compress) wav-sequence to Mel-frequency cepstral coefficients (MFCCs)

        input:
            data: a 1D wav-sequence

        output: 2D-array with MFCCs
        """
        return librosa.feature.mfcc(data, n_mfcc=n_mfcc)

    def normalize_MFCCs(self):
        """
        normalize time-MFCCs to interval: [0, 1]
        """
        self.min_MFCCs = np.min(self.MFCCs)
        self.max_MFCCs = np.max(self.MFCCs - self.min_MFCCs)

        return (self.MFCCs - self.min_MFCCs) / self.max_MFCCs

    def denormalize_MFCCs(self, norm_MFCCs):
        """
        denormalize time-MFCCs based on above normalization (max, min)
        """
        return (norm_MFCCs * self.max_MFCCs) + self.min_MFCCs

    def normalize_MFCCs_v2(self):
        """
        Normalize such that the data is: N(0, 1) (standard normal distributed)
        """
        self.mean_MFCCs = np.mean(self.MFCCs)
        self.std_MFCCs = np.std(self.MFCCs)

        return (self.MFCCs - self.mean_MFCCs) / self.std_MFCCs

    def denormalize_MFCCs_v2(self, norm_MFCCs):
        """
        denormalize v2 normalization
        """
        return norm_MFCCs * self.std_MFCCs + self.mean_MFCCs

    def normalize_MFCCs_v3(self):
        """
        normalize time-MFCCs to interval: [-1, 1]
        """
        self.min_MFCCs = np.min(self.MFCCs)
        self.max_MFCCs = np.max(self.MFCCs - self.min_MFCCs)
        return 2*((self.MFCCs - self.min_MFCCs) / self.max_MFCCs) - 1

    def denormalize_MFCCs_v3(self, norm_MFCCs):
        """
        denormalize time-MFCCs based on above normalization (max, min)
        """

        return ((norm_MFCCs * self.max_MFCCs) + self.min_MFCCs)/2 - 1

    def invlogamplitude(self, S):
        """
        librosa.logamplitude is actually 10_log10, so invert that.

        Follows from decode_MFCCs
        """
        return 10.0**(S/10.0)

    def decode_MFCCs(self, mfccs):
        """
        Reconstruction of wav time-signal from MFCCs. Code taken from:
        https://amyang.xyz/posts/Inverse-MFCC-to-WAV
        """
        n_mfcc = mfccs.shape[0]
        n_mel = 128
        dctm = librosa.filters.dct(n_mfcc, n_mel)
        n_fft = 2048
        mel_basis = librosa.filters.mel(self.sr, n_fft)

        bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))

        recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, self.invlogamplitude(np.dot(dctm.T, mfccs)))

        shape_inv_recon = librosa.istft(recon_stft).shape[0]
        #excitation = np.random.randn(self.wav_shape)
        excitation = np.random.randn(shape_inv_recon)
        E = librosa.stft(excitation)

        recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
        #recon = librosa.istft(np.sqrt(recon_stft))
        return recon

    def create_input_target_test_data(self, MFCCs, k=1, num_time_steps=101,
                                      percentile_test=0.1):
        """
        Many-to-one, time-shifted-by-k

        input:
            MFCCs: 2D-array with mfccs (from librosa the matrix is of form: D[f, t])
            num_time_steps: how many time steps for an input to predict one time step output

        output:
            input_data: 3D input data, of the form (MFCC_sequence_i, time_i, MFCC_i) or 
                (sample, time_steps, feature)
            target_data: 2D target data of the form (MFCC_sequence_i, MFCC_i)
            validation_data: same as input_data and target_data (only part of the sequence, 1?)
        """

        #Change shape of MFCCs to D[t, f]
        MFCCs = np.swapaxes(MFCCs, 0, 1)
        input_data = []
        target_data = []
        input_data_test = []
        target_data_test = []
        num_iter = int((MFCCs.shape[0]-num_time_steps)/k)
        for i in range(num_iter):
            input_data.append(MFCCs[i*k:i*k+num_time_steps,:])
            target_data.append(MFCCs[i*k+num_time_steps,:])

        self.input_data = np.array(input_data)
        self.target_data = np.array(target_data)
        self.input_data_test = self.input_data[int(num_iter*(1-percentile_test)):]
        self.target_data_test = self.target_data[int(num_iter*(1-percentile_test)):]

    def create_input_target_test_data_v2(self, MFCCs, k=1, num_time_steps=101,
                                         percentile_test=0.1):
        """
        Many-to-many, one novel point, time-shifted-by-k
        """
        MFCCs = np.swapaxes(MFCCs, 0, 1)
        input_data = []
        target_data = []
        input_data_test = []
        target_data_test = []
        num_iter = int((MFCCs.shape[0]-num_time_steps)/k)
        for i in range(num_iter):
            input_data.append(MFCCs[i*k:i*k+num_time_steps-1,:])
            target_data.append(MFCCs[i*k+1:i*k+num_time_steps,:])

        self.input_data = np.array(input_data)
        self.target_data = np.array(target_data)

        self.input_data_test = self.input_data[int(num_iter*(1-percentile_test)):]
        self.target_data_test = self.target_data[int(num_iter*(1-percentile_test)):]

    def reduce_noise(self, signal):
        """
        MMSE-STSA
        """
        output, saved_params = mmse.MMSESTSA(signal, self.sr)
        return output[:,0]

if __name__ == "__main__":
    """
    n_mfcc = 35
    dp = DataProcessor(num_files=1, n_mfcc=n_mfcc)

    decoded_signal = dp.decode_MFCCs(dp.MFCCs)
    plt.figure(2)
    librosa.display.waveplot(decoded_signal, dp.sr)

    noised_reduced_reconstructed = dp.reduce_noise(decoded_signal)
    plt.figure(3)
    librosa.display.waveplot(noised_reduced_reconstructed, dp.sr)

    plt.show()

    librosa.output.write_wav('./test_n_mffcs/decompressed_{}.wav'.format(n_mfcc),
    decoded_signal, dp.sr, norm=True)
    librosa.output.write_wav('./test_n_mffcs/noise_reduced_decompressed_{}.wav'.format(n_mfcc),
    noised_reduced_reconstructed, dp.sr, norm=True)
    librosa.output.write_wav('./meisterfloh_file_1_compressed{}.wav'.format(n_mfcc),
    decoded_signal, dp.sr, norm=True)
    """
    print("Hello")
