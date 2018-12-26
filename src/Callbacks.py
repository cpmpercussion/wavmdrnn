import numpy as np
import keras

class stats_callback(keras.callbacks.Callback):
	
	def __init__(self):
		self.Model = None
		self.pred_tot = None
		self.i = 0

	def on_epoch_end(self, epoch, logs={}):
		input_start = np.random.randint(self.Model.data_processor.input_data.shape[0])
		pred_tot = self.Model.predict_sequence(model_version=self.Model.model_version, 
			input_data_start=input_start, num_preds=10, create_wav=False)
		
		if(self.i==0):
			self.pred_tot = pred_tot

		self.pred_tot = np.concatenate((self.pred_tot, pred_tot))
		self.i += 1