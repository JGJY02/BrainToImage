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
from models.dual_models.eegclassifier import convolutional_encoder_model_512_dual, convolutional_encoder_model_spectrogram, convolutional_encoder_model_spectrogram_stacked, LSTM_Classifier_dual_512, convolutional_encoder_model_128_dual
from models.transformer_classifier import EEGViT_raw

import argparse
import numpy as np
# Import transformer

"""
Title: Training script for EEG Classification

Purpose:
    Testing  build and training of EEG classifier, Functional blocks for training
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
parser.add_argument('--root_dir', type=str, help="Directory to the dataset - CNN_encoder / LSTM_encoder / Transformer", default = "processed_dataset/filter_mne_car/CNN_encoder",required=False)
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training xxxthresh_(channels)stack(model)_(dataset) 000thresh_AllSlidingCNN_All.pkl / 000thresh_AllStackLstm_All.pkl / 000thresh_AllStackTransformer_All", default = "000thresh_AllSlidingCNN_dual_28_All.pkl" , required=False)
parser.add_argument('--model_type', type=str, help="CNN_spectrogram, CNN_windowed, LSTM, VIT", default = "CNN_windowed" , required=False)
parser.add_argument('--model_name', type=str, help="Name of the model", default= "CNN_all_stacked_signals_dual_512_28_ori", required=False)
parser.add_argument('--latent_size', type=int, help="Size of the latent, 128 or 512", default = 512, required=False)
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

    print(f"Model {name} saved to {path}")

## Ensure main_dir is one file step back to allow access to all files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

indexes = [i for i, char in enumerate(args.dataset_pickle) if char == '_']
run_id = args.dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
model_name = args.model_name

dataset_dir_path = f"{args.root_dir}/{args.input_dir}"
dataset_file_path = f"{dataset_dir_path}/{args.dataset_pickle}"


output_dir_path = f"{args.output_dir}/"

model_save_dir = os.path.join(output_dir_path,args.input_dir,f"{run_id}",model_name)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print(f"Saving Models to {model_save_dir}")


print(f"** Reading data file {dataset_file_path} **")
eeg_data = pickle.load(open(f"{dataset_file_path}", 'rb'), encoding='bytes')
x_train, y_train, y_secondary_train, x_test, y_test, y_secondary_test = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']

print(f"** Now loading model {args.model_type}, with latent size: {args.latent_size}")
if args.model_type == "CNN_spectrogram":
    classifier = convolutional_encoder_model_spectrogram(x_train.shape[1], x_train.shape[2], 10) # _expanded
    batch_size, num_epochs = 128, 250 #128, 150

elif args.model_type == "LSTM":

    classifier = LSTM_Classifier_dual_512(x_train.shape[1], x_train.shape[2], 512, y_train.shape[1], y_secondary_train.shape[1])
    batch_size, num_epochs = 128, 100 #128, 150


elif args.model_type == "CNN_windowed":
    if args.latent_size == 128:
        classifier = convolutional_encoder_model_128_dual(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_secondary_train.shape[1])
    elif args.latent_size == 512:
        classifier = convolutional_encoder_model_512_dual(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_secondary_train.shape[1])    
    batch_size, num_epochs = 128, 150 #128, 150

elif args.model_type == "transformer":
    classifier = EEGViT_raw(x_train.shape[1], x_train.shape[2], 10)
    batch_size, num_epochs = 128, 150 #128, 150

else:
    raise ValueError(f"Error! Model type {args.model_type} does not exist!")

print(f"** Model {args.model_type}, with latent size: {args.latent_size} Successfuly loaded!")

# location for the trained model file
saved_model_file = os.path.join(model_save_dir, "eeg_classifier_adm5" + '_final' + '.h5')

# location for the intermediate model files
filepath = os.path.join(model_save_dir, "eeg_classifier_adm5" + "-model-improvement-{epoch:02d}.h5")  #{epoch:02d}-{val_accuracy:.2f}

# call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
callback_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=False, save_best_only=True, mode='max')
variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

adm = optimizers.Adam(learning_rate=1e-2, beta_1=0.9, decay=1e-6)

# classifier.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
classifier.compile(
    loss={
        'EEG_Class_Labels': 'categorical_crossentropy',
        'EEG_Class_type_Labels': 'categorical_crossentropy'
    },
    optimizer=adm,

    metrics={
        'EEG_Class_Labels': ['accuracy'],
        'EEG_Class_type_Labels': ['accuracy']
    }
)

# classifier.summary()
print("** Beginning to train model now!")
history = classifier.fit(x_train, [y_train, y_secondary_train], epochs=num_epochs, batch_size=batch_size, validation_split=0.25, callbacks=[callback_checkpoint], verbose=True)
save_model(history.history, f"history_{str(model_name)}_final.pkl",model_save_dir)
#classifier.load_weights(saved_model_file)
classifier.save(saved_model_file)
print("** Training and saving complete!")


accuracy = classifier.evaluate(x_test, [y_test, y_secondary_test], batch_size=batch_size, verbose=False)
print(f"Test results are: {accuracy}")