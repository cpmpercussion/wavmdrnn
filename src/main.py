from DataProcessor import DataProcessor
from Model import Model
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras import regularizers
import tqdm

"""
TODO: 
1. Create Model save each %% epoch
2. Test loading and prediction/further train
3. Extend model selection to two separate methods, i.e a model selction and a train method
4. Extend main with choice of: start_train, more_train, inference
5. Save and restore train statistics? Such that more_train doesn't overwrite train statistics
6. Test temperature from MDN? I.e greedy select vs. creative
7. Fix plots

-- START: create github and save there
"""


model_and_data_version=2
name = "test"

data_processor = DataProcessor(start_file_num=1, num_files=2, n_mfcc=35)

data_processor.choose_data_model(normalization_version=2, 
	data_version=model_and_data_version, 
	num_time_steps=201, k=20, percentile_test=0)

model = Model(data_processor, num_epochs=2)

model.train_and_predict(model_version=model_and_data_version, 
		save_model=True, N_MIXES=5, 
		model_name="./../models/{}.json".format(name), 
		weights_name="./../models/{}.h5".format(name), 
		input_data_start=0, num_preds=50, name=name)

