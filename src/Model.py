#from DataProcessor import DataProcessor
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization, Bidirectional
from keras import regularizers, optimizers
import tqdm
import sys
import keras
import Callbacks
#sys.path.insert(0, "./../keras-mdn-layer/")
import mdn
import time

class Model:

    def __init__(self, data_processor, model_version = 1, name="default"):
        self.data_processor = data_processor
        self.name = name
        self.model_version = model_version
        self.stats_cb = Callbacks.stats_callback()
        self.stats_cb.Model = self


    def kSM(self, N_MIXES=5):
        """
        Initialize k-Shifted Model
        """
        self.N_MIXES = N_MIXES
        self.OUTPUT_DIMS = self.data_processor.target_data.shape[1]

        self.model = Sequential()
        self.model.add(LSTM(units=160, return_sequences=True,  
            input_shape=self.data_processor.input_data.shape[1:]))
        self.model.add(LSTM(units=160, return_sequences=True))
        self.model.add(LSTM(units=160))
        self.model.add(mdn.MDN(self.OUTPUT_DIMS, self.N_MIXES))
        self.model.compile(loss=mdn.get_mixture_loss_func(self.OUTPUT_DIMS, 
            self.N_MIXES), optimizer='nadam')


    def TDkSM(self, N_MIXES=5, name="default2"):
        """
        Initialize Time-Distributed k-Shifted Model
        """
        self.N_MIXES = N_MIXES
        self.OUTPUT_DIMS = self.data_processor.target_data.shape[2]

        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True,
            input_shape=self.data_processor.input_data.shape[1:]))
        #self.model.add(LSTM(units=150, kernel_regularizer=regularizers.l2(0.2), return_sequences=True))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(LSTM(units=50, return_sequences=True))
        #self.model.add(LSTM(units=100, return_sequences=True))
        #self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(mdn.MDN(self.OUTPUT_DIMS, self.N_MIXES)))

        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)
        #adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss=mdn.get_mixture_loss_func(self.OUTPUT_DIMS, 
            self.N_MIXES), optimizer='nadam')


    def train(self, epochs, batch_size=64, validation_split=0.15):
        """
        Train kSM or TDkSM
        """
        history = self.model.fit(self.data_processor.input_data, 
            self.data_processor.target_data, epochs=epochs, 
            batch_size=batch_size, validation_split=validation_split, callbacks=[self.stats_cb])

        self.save()
        fig = plt.figure(4)
        plt.subplot(2,1,1)
        plt.plot(history.history['loss'])
        plt.title('training loss')
        plt.subplot(2,1,2)
        plt.plot(history.history['val_loss'])
        plt.title('validation loss')
        fig.savefig("./../plots/{}_loss.png".format(self.name))
        print(self.model.summary())


    def predict_sequence(self, input_data_start=0, num_preds=800, plot_stats=True, 
        save_wav=True):
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
                    self.OUTPUT_DIMS, self.N_MIXES)
                pred_tot.append(np.copy(pred))
            else:
                new_elem = mdn.sample_from_output(pred[0][-1], 
                    self.OUTPUT_DIMS, self.N_MIXES)
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

        librosa.output.write_wav('./../results/{}.wav'.format(self.name), 
            out_sequence, self.data_processor.sr, norm=True)
        fig = plt.figure(3)
        librosa.display.waveplot(out_sequence, self.data_processor.sr)
        fig.savefig("./../plots/{}.png".format(self.name))


    def _mixture_components(self, predictions, num_plots=1):
        """
        Statistics of mixture components
        """
        stats = []
        for i in range(len(predictions)):
            mus, sigs, pis = mdn.split_mixture_params(np.squeeze(np.array(predictions))[i], 
                self.OUTPUT_DIMS, self.N_MIXES)
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

        fig = plt.figure(num_plots)
        x_label = ['mean', 'std', 'max', 'min']
        y_label = ['sigs', 'means', 'pis']
        x,y=0,0
        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.plot(np.linspace(0, len(stats)-1, len(stats)), stats[:,i])

            if(i>7):
                plt.xlabel(x_label[x])
                x += 1
            if((i+1)%4==1):
                plt.ylabel(y_label[y])
                y += 1

        if(num_plots==1):
            fig.savefig("./../plots/{}_stats.png".format(self.name))
        elif(num_plots==2):
            fig.savefig("./../plots/{}_stats.png".format(self.name+"_train"))


    def save(self):
        """
        Save model weights
        """
        self.model.save_weights("./../models/{}.h5".format(self.name))
        print("Saved weights")


    def load(self):
        """
        Load model weights
        """
        self.model.load_weights("./../models/{}.h5".format(self.name))
        print("Loaded weights")







