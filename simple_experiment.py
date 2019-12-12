from wavmdrnn.dataprocessor import DataProcessor
from wavmdrnn import Model
import tensorflow as tf
config = tf.ConfigProto()

print("Running simple wavmdrnn experiment.")
# Setup hyperparameters
model_choice = 2
#Choices: 1 => [0, 1], 2 => N(0, 1), 3 => [-1, 1]
normalization_version = 2
n_mfcc = 50
num_time_steps = 201
k = 1
percentile_test = 0
validation_split = 0.15
N_MIXES = 10
num_epochs = 100
batch_size = 64

print("Loading data...")
# ## Load the data
# - todo: should save the intermediary data file.
wav_dir = "./wavfiles"
data_processor = DataProcessor(wav_dir=wav_dir, n_mfcc=n_mfcc)
data_processor.choose_data_model(normalization_version=normalization_version, 
                                 data_version=model_choice, 
                                 num_time_steps=num_time_steps, 
                                 k=k, 
                                 percentile_test=percentile_test)
print("Loading model...")
# Construct the model
name = 'derp'
model = Model(data_processor, model_version=model_choice, name=name)

# model.kSM(n_mixes=N_MIXES)
model.TDkSM(units=256, n_mixes=N_MIXES)

print("Training...")
# Do some training.
model.train(epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

print("Predicting...")
# Do some predicting.
model.predict_sequence(input_data_start=5000, num_preds=1000, plot_stats=False, save_wav=True)
