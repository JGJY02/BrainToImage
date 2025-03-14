import os
import pickle
import sys

from keras import optimizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.regularizers import l2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.eegclassifier import convolutional_encoder_model, convolutional_encoder_model_spectrogram_middle_latent, LSTM_Classifier, classification_model, LSTM_VAE_encoder_model
from keras.models import Model

import argparse

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
parser.add_argument('--root_dir', type=str, help="Directory to the dataset", default = "Datasets/MindBigData - The Visual MNIST of Brain Digits",required=False)
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "2022Data",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training", default = "800thresh_AllStackLstm_processed_train_MindBigData2022_MNIST_EP.pkl" , required=False)
parser.add_argument('--imageOrwindowed', type=str, help="spectrogram for image windowed for original", default = "LSTM_VAE" , required=False)
parser.add_argument('--encoder_name', type=str, help="Name of the model", default= "LSTM_VAE_Reworked_auto_encoder", required=False)
parser.add_argument('--encoder_dir', type=str, help="directory to the classifier", default= "MNIST_EP", required=False)

parser.add_argument('--model_name', type=str, help="Name of the model", default= "LSTM_Reworked_auto_encoder_classifier", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "MNIST_EP",required=False)
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
x_train, y_train, x_test, y_test = eeg_data['x_train'], eeg_data['y_train'], eeg_data['x_test'], eeg_data['y_test']

print(x_train.shape)
if args.imageOrwindowed == "spectrogram":
    encoder = convolutional_encoder_model_spectrogram_middle_latent(x_train.shape[1], x_train.shape[2]) # _expanded
    # classifier = convolutional_encoder_model_spectrogram_stacked(x_train.shape[2], x_train.shape[0], x_train.shape[1], 10) # _expanded
    batch_size, num_epochs = 128, 250 #128, 150

    print(x_train[0][0])

elif args.imageOrwindowed == "LSTM":
    encoder = LSTM_Classifier(x_train.shape[1], x_train.shape[2], 10)
    batch_size, num_epochs = 128, 250 #128, 150
    encoder_model_path = f"{args.root_dir}/{args.encoder_dir}/models/{run_id}/{args.encoder_name}/eeg_encoder_adm5_final.h5"
    print("Path to the encoder is : ", encoder_model_path)
    encoder.load_weights(encoder_model_path)
    layer_names = ['EEG_feature_FC128']
    encoder_outputs = [encoder.get_layer(layer_name).output for layer_name in layer_names]
    encoder_model = Model(inputs=encoder.input, outputs=encoder_outputs)

elif args.imageOrwindowed == "LSTM_VAE":
    encoder, e, d, mu, log_var = LSTM_VAE_encoder_model(x_train.shape[1], x_train.shape[2])
    batch_size, num_epochs = 128, 250 #128, 150
    encoder_model_path = f"{args.root_dir}/{args.encoder_dir}/models/{run_id}/{args.encoder_name}/eeg_encoder_adm5_final.h5" #eeg_encoder_adm5-model_best.h5
    print("Path to the encoder is : ", encoder_model_path)
    encoder.load_weights(encoder_model_path)
    

_, _, latent_representation = e.predict(x_train)  # data_for_encoding is your input sequence data
print(np.array(latent_representation).shape)
classifier = classification_model(128, 10)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# location for the trained model file
saved_model_file = os.path.join(model_save_dir, "eeg_classifier_adm5" + '_final' + '.h5')

# location for the intermediate model files
filepath = os.path.join(model_save_dir, "eeg_classifier_adm5" + "-model-improvement-{epoch:02d}-{val_accuracy:.2f}.h5")  #{epoch:02d}-{val_accuracy:.2f}

#Import Encoder
#classifier_optimizer = Adam(learning_rate=0.0001, decay=1e-6)
#classifier.compile(loss='categorical_crossentropy', optimizer=classifier_optimizer, metrics=['accuracy'])

## End of encoder import

# call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
callback_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=False, save_best_only=True, mode='max')
variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

sgd = optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adm = optimizers.Adam(learning_rate=1e-3, beta_1=0.9, decay=1e-6)

classifier.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
classifier.summary()

history = classifier.fit(latent_representation, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.25, callbacks=[callback_checkpoint], verbose=True)
save_model(history.history, f"history_{str(model_name)}_final.pkl",model_save_dir)
#classifier.load_weights(saved_model_file)
classifier.save(saved_model_file)

test_latent_representation = encoder_model.predict(x_test)  # data_for_encoding is your input sequence data
accuracy = classifier.evaluate(test_latent_representation, y_test, batch_size=batch_size, verbose=False)
print(accuracy)