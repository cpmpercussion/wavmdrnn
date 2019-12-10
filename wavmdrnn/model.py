"""
Audio-to-Audio Mixture Density Recurrent Neural Network
An MDRNN for generating digital audio signals.
Developed as part of IN5490 at the University of Oslo, 
Research Group for Robotics and Intelligent Systems, 2018.
"""
import librosa, librosa.display
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Bidirectional
from tensorflow.keras import regularizers, optimizers
import tensorflow.keras as keras
import tqdm
import sys
import mdn
import time
import datetime
from .callbacks import stats_callback


class Model:

    def __init__(self, data_processor, base_dir="./", model_version = 1, name="default"):
        self.data_processor = data_processor
        self.base_dir = base_dir
        self.name = name
        self.model_version = model_version
        self.stats_cb = stats_callback()
        self.stats_cb.Model = self

    def kSM(self, units=160, n_mixes=5):
        """
        Initialize k-Shifted Model
        """
        self.n_mixes = n_mixes
        self.OUTPUT_DIMS = self.data_processor.target_data.shape[1]
        self.model = Sequential()
        self.model.add(LSTM(units=units, return_sequences=True,  
            input_shape=self.data_processor.input_data.shape[1:]))
        self.model.add(LSTM(units=units, return_sequences=True))
        self.model.add(LSTM(units=units))
        self.model.add(mdn.MDN(self.OUTPUT_DIMS, self.n_mixes))
        self.model.compile(loss=mdn.get_mixture_loss_func(self.OUTPUT_DIMS, 
            self.n_mixes), optimizer='nadam')
        print(self.model.summary())
                

    def TDkSM(self, n_mixes=5, units=50, name="default2"):
        """
        Initialize Time-Distributed k-Shifted Model
        """
        self.n_mixes = n_mixes
        self.units = units
        self.OUTPUT_DIMS = self.data_processor.target_data.shape[2]
        self.model = Sequential()
        self.model.add(LSTM(units=self.units, return_sequences=True,
            input_shape=self.data_processor.input_data.shape[1:]))
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(LSTM(units=self.units, return_sequences=True))
        self.model.add(TimeDistributed(mdn.MDN(self.OUTPUT_DIMS, self.n_mixes)))
        self.model.compile(loss=mdn.get_mixture_loss_func(self.OUTPUT_DIMS,
            self.n_mixes), optimizer='nadam')
        print(self.model.summary())


    def train(self, epochs, batch_size=64, validation_split=0.15):
        """
        Train kSM or TDkSM
        """
        date_string = datetime.datetime.today().strftime('%Y%m%d-%H_%M_%S')
        filepath = self.base_dir + "wavmdrnn-E{epoch:02d}-VL{val_loss:.2f}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='min')
        terminateOnNaN = keras.callbacks.TerminateOnNaN()
        tboard = keras.callbacks.TensorBoard(log_dir=self.base_dir + 'logs/' + date_string + "wavmdrnn",
                                     batch_size=32,
                                     write_graph=True,
                                     update_freq='epoch')
        callbacks = [checkpoint, terminateOnNaN, tboard, self.stats_cb]
        history = self.model.fit(self.data_processor.input_data,
                                 self.data_processor.target_data,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 callbacks=callbacks)
        self.save()

    def predict_sequence(self, input_data_start=0, num_preds=800,
                         plot_stats=True, save_wav=True):
        """
        Predict new wav file, display and save it
        """
        #input_data_start = np.random.randint(self.data_processor.input_data.shape[0])
        new_sequence = np.array([self.data_processor.input_data[input_data_start, :, :]])
        out_sequence = new_sequence
        out_sequence = np.squeeze(out_sequence)
        pred_tot = []
        for i in tqdm.tqdm(range(num_preds)):
            pred = self.model.predict(new_sequence)

            if(self.model_version == 1):
                new_elem = mdn.sample_from_output(pred[0],
                    self.OUTPUT_DIMS, self.n_mixes)
                pred_tot.append(np.copy(pred))
            else:
                new_elem = mdn.sample_from_output(pred[0][-1],
                    self.OUTPUT_DIMS, self.n_mixes)
                pred_tot.append(np.copy(pred[0][-1]))

            new_sequence = np.concatenate((new_sequence[0,1:,:], new_elem))
            out_sequence = np.concatenate((out_sequence, new_elem))
            new_sequence = new_sequence.reshape(1, new_sequence.shape[0],
                new_sequence.shape[1])

        pred_tot = np.array(pred_tot)
        if(save_wav):
            self._inverse_and_plot_sequence(out_sequence)
        if(plot_stats):
            self._mixture_components(pred_tot, num_plots=1)
            self._mixture_components(self.stats_cb.pred_tot, num_plots=2)

        return pred_tot


    def _inverse_and_plot_sequence(self, out_sequence):
        """
        Inverse transform output data from prediction phase, save
        wav-file and save a plot of wav-file
        """
        #inverse transform to get 1D-sequence, also force upper bound
        out_sequence = np.swapaxes(out_sequence, 0, 1)
        out_sequence = self.data_processor.denorm(out_sequence)
        out_sequence = self.data_processor.decode_MFCCs(out_sequence)
        out_sequence = self.data_processor.reduce_noise(out_sequence)
        print(out_sequence.shape, np.max(abs(out_sequence)))
        out_sequence = np.where(abs(out_sequence) > 1.0, 1.0, out_sequence)
        print(out_sequence.shape, np.max(abs(out_sequence)))

        librosa.output.write_wav(self.base_dir + "results/{}.wav".format(self.name),
            out_sequence, self.data_processor.sr, norm=True)


    def _mixture_components(self, predictions, num_plots=1):
        """
        Statistics of mixture components
        """
        stats = []
        for i in range(len(predictions)):
            mus, sigs, pis = mdn.split_mixture_params(np.squeeze(np.array(predictions))[i],
                self.OUTPUT_DIMS, self.n_mixes)
            stats.append(np.zeros(12))
            stats[i][0] = np.mean(sigs)
            stats[i][1] = np.std(sigs)
            stats[i][2] = np.max(sigs)
            stats[i][3] = np.min(sigs)
            stats[i][4] = np.mean(mus)
            stats[i][5] = np.std(mus)
            stats[i][6] = np.max(mus)
            stats[i][7] = np.min(mus)
            stats[i][8] = np.mean(pis)
            stats[i][9] = np.std(pis)
            stats[i][10] = np.max(pis)
            stats[i][11] = np.min(pis)
        stats = np.array(stats)


    def save(self):
        """
        Save model weights
        """
        self.model.save_weights(self.base_dir + "models/{}.h5".format(self.name))
        print("Saved weights")


    def load(self):
        """
        Load model weights
        """
        self.model.load_weights(self.base_dir + "models/{}.h5".format(self.name))
        print("Loaded weights")
