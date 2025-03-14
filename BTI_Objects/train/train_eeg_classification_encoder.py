import os
import pickle
import sys

from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.regularizers import l2

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.eegclassifier import convolutional_encoder_model, convolutional_encoder_model_spectrogram_middle_latent, LSTM_Classifier, LSTM_VAE_encoder_model, LSTM_Encoder
from tensorflow.keras import backend as K

import argparse
import matplotlib.pyplot as plt
"""
Title: Training script for EEG Classification

Purpose:
    Testing  build and training of EEG encoder, Functional blocks for training
    classification model

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    Run the script as is, uses the online MNIST dataset to train the GAN

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""
#Argument parser 
parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset", default = "processed_dataset/filter_mne_car",required=False)
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training", default = "000thresh_AllStackLstm_All.pkl" , required=False)
parser.add_argument('--imageOrwindowed', type=str, help="spectrogram for image windowed for original", default = "LSTM_VAE" , required=False)
parser.add_argument('--model_name', type=str, help="Name of the model", default= "LSTM_all_stacked_signals", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/classifiers",required=False)
args = parser.parse_args()


print(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
print(os.getcwd())

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def save_model(model,name,path):
    check_path(path)
    filename = os.path.join(path,name)
    with open(filename,"wb") as f:
        pickle.dump(model,f)

    print("Model {} saved to {}".format(name,path))

def plot_history(history, filename='Loss_accuracy_history.png'):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot as an image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")



## Ensure main_dir is one file step back to allow access to all files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

indexes = [i for i, char in enumerate(args.dataset_pickle) if char == '_']
run_id = args.dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
model_name = args.model_name

dataset_dir_path = f"{args.root_dir}/{args.input_dir}"
dataset_file_path = f"{dataset_dir_path}/{args.dataset_pickle}"


output_dir_path = f"{args.root_dir}/{args.output_dir}/"
# output_file_path = f"{output_dir_path}/{run_id}_{args.model_name}"

model_save_dir = os.path.join(output_dir_path,"models",f"{run_id}",model_name)
print(f"Saving Models to {model_save_dir}")


print(f"** Reading data file {dataset_file_path}")
eeg_data = pickle.load(open(f"{dataset_file_path}", 'rb'), encoding='bytes')

# x_train, y_train, x_test, y_test = eeg_data['x_train'], eeg_data['y_train'], eeg_data['x_test'], eeg_data['y_test']
x_train, y_train, x_test, y_test = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['x_test_eeg'], eeg_data['y_test']

print(x_train.shape)
# use_only = 1000 #Test purposes
# x_train = x_train[:use_only]
# y_train = y_train[:use_only]
# x_test = x_test[:use_only]
# y_test = y_test[:use_only]


if args.imageOrwindowed == "spectrogram":
    encoder = convolutional_encoder_model_spectrogram_middle_latent(x_train.shape[1], x_train.shape[2]) # _expanded
    # encoder = convolutional_encoder_model_spectrogram_stacked(x_train.shape[2], x_train.shape[0], x_train.shape[1], 10) # _expanded
    batch_size, num_epochs = 128, 250 #128, 150
    adm = optimizers.Adam(learning_rate=0.001, beta_1=0.9, decay=1e-7)
    encoder.compile(loss='mae', optimizer=adm, metrics=['mse'])

    print(x_train[0][0])

elif args.imageOrwindowed == "LSTM":
    encoder = LSTM_Encoder(x_train.shape[1], x_train.shape[2], 256)
    batch_size, num_epochs = 128, 1000 #128, 150
    adm = optimizers.Adam(learning_rate=0.001, beta_1=0.9, decay=1e-7)
    encoder.compile(loss='mse', optimizer=adm, metrics=['mae'])

elif args.imageOrwindowed == "LSTM_VAE":
    encoder, e, d, mu, log_var = LSTM_VAE_encoder_model(x_train.shape[1], x_train.shape[2], 256)
    batch_size, num_epochs = 128, 1000 #128, 150
    adm = optimizers.Adam(learning_rate=1e-5, beta_1=0.9, decay=1e-7)

    encoder.compile(optimizer=adm)




if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# location for the trained model file
saved_model_file = os.path.join(model_save_dir, "eeg_encoder_adm5" + '_final' + '.h5')

# location for the intermediate model files
filepath = os.path.join(model_save_dir, "eeg_encoder_adm5" + "-model-improvement-{epoch:02d}-{val_loss:.2f}.h5")  #{epoch:02d}-{val_accuracy:.2f}

# call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
callback_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=False, save_best_only=True, mode='min')
variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

sgd = optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

encoder.summary()
history = encoder.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.25, callbacks=[callback_checkpoint], verbose=True)
save_model(history.history, f"history_{str(model_name)}_final.pkl",model_save_dir)
#encoder.load_weights(saved_model_file)
encoder.save(saved_model_file)


plot_history(history, filename=f'{model_save_dir}/Loss_accuracy_history.png')

test_loss = encoder.evaluate(x_test, x_test, batch_size=batch_size, verbose=False)
with open(f'{model_save_dir}/test_results.txt', "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")

print(f"Test loss: {test_loss}")