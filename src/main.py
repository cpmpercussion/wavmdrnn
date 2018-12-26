from DataProcessor import DataProcessor
from Model import Model
import sys

"""
TODO: 
1. Create Model save each %% epoch
2. Test loading and prediction/further train
3. Extend model selection to two separate methods, i.e a model selction and a train method
4. Extend main with choice of: start_train, more_train, inference
5. Save and restore train statistics? Such that more_train doesn't overwrite train statistics
6. Test temperature from MDN? I.e greedy select vs. creative
7. Fix plots
8. Restore data to include test data, such that we can perform inference on test set

-- START: create github and save there
"""

try:
	name = sys.argv[2]
except IndexError:
	print("\n<start_train/more_train/predict> <name>\n")
	sys.exit(1)


#Choices: 1 => kSM, 2 => TDkSM
model_choice = 1

#Choices: 1 => [0, 1], 2 => N(0, 1), 3 => [-1, 1]
normalization_version = 2

num_files = 2
n_mfcc = 35
num_time_steps = 201
k = 1
percentile_test = 0
validation_split = 0.15
num_epochs = 2
N_MIXES = 5
batch_size = 64
input_data_start = 0
num_preds = 50

data_processor = DataProcessor(start_file_num=1, num_files=num_files, n_mfcc=n_mfcc)

data_processor.choose_data_model(normalization_version=normalization_version, 
	data_version=model_choice, 
	num_time_steps=num_time_steps, k=k, percentile_test=percentile_test)

model = Model(data_processor, model_version=model_choice, name=name)


if(model_choice == 1):
	model.kSM(N_MIXES=N_MIXES)
else:
	model.TDkSM(N_MIXES=N_MIXES)

if(sys.argv[1] == "start_train"):
	model.train(epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)
	model.predict_sequence(input_data_start=input_data_start, num_preds=num_preds, 
		plot_stats=True, save_wav=True)

elif(sys.argv[1] == "more_train"):
	model.load()
	model.train(epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)
	model.predict_sequence(input_data_start=input_data_start, num_preds=num_preds, 
		plot_stats=True, save_wav=True)

elif(sys.argv[1] == "predict"):
	model.load()
	model.predict_sequence(input_data_start=input_data_start, num_preds=num_preds, 
		plot_stats=True, save_wav=True)














