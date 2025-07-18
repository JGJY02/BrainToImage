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
from models.eegclassifier import convolutional_encoder_model_512_dual, convolutional_encoder_model_spectrogram, convolutional_encoder_model_spectrogram_stacked, LSTM_Classifier_dual_512, convolutional_encoder_model_128_dual
from models.transformer_classifier import EEGViT_raw

import argparse
import numpy as np
# Import transformer
from transformers  import ViTImageProcessor
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from keras.models import Model, Sequential
import tensorflow as tf
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score 
from scikeras.wrappers import KerasModel

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from scikeras.wrappers import KerasClassifier

import pandas as pd

#Argument parser 
parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset - CNN_encoder / LSTM_encoder / Transformer", default = "processed_dataset/filter_mne_car/CNN_encoder",required=False)
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training xxxthresh_(channels)stack(model)_(dataset) 000thresh_AllSlidingCNN_All.pkl / 000thresh_AllStackLstm_All.pkl / 000thresh_AllStackTransformer_All", default = "000thresh_AllSlidingCNN_dual_28_All.pkl" , required=False)
parser.add_argument('--imageOrwindowed', type=str, help="spectrogram for image windowed for original", default = "windowed" , required=False)
parser.add_argument('--model_name', type=str, help="Name of the model", default= "CNN_all_stacked_signals_dual_512_28_ori", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/classifiers/crossVal",required=False)
parser.add_argument('--latent_size', type=int, help="Size of the latent, 128 or 512", default = 512, required=False)

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


output_dir_path = f"{args.output_dir}/"
# output_file_path = f"{output_dir_path}/{run_id}_{args.model_name}"

model_save_dir = os.path.join(output_dir_path,args.input_dir,f"{run_id}",model_name)
print(f"Saving Models to {model_save_dir}")


print(f"** Reading data file {dataset_file_path}")
eeg_data = pickle.load(open(f"{dataset_file_path}", 'rb'), encoding='bytes')

eeg_encoding_dim = args.latent_size
x_train, y_train, y_secondary_train, x_test, y_test, y_secondary_test = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']

X = np.vstack((x_train, x_test))
Y_primary = np.vstack((y_train, y_test))
Y_secondary = np.vstack((y_secondary_train, y_secondary_test))

def build_model(optimizer='adam', learning_rate=0.001, decay=0.0):
    if args.imageOrwindowed == "spectrogram":
        classifier = convolutional_encoder_model_spectrogram(x_train.shape[1], x_train.shape[2], 10) # _expanded
        # classifier = convolutional_encoder_model_spectrogram_stacked(x_train.shape[2], x_train.shape[0], x_train.shape[1], 10) # _expanded
        batch_size, num_epochs = 128, 250 #128, 150

        print(x_train[0][0])

    elif args.imageOrwindowed == "LSTM":

        classifier = LSTM_Classifier_dual_512(x_train.shape[1], x_train.shape[2], 512, y_train.shape[1], y_secondary_train.shape[1])
        batch_size, num_epochs = 128, 100 #128, 150


    elif args.imageOrwindowed == "windowed":
        if eeg_encoding_dim == 128:
            classifier = convolutional_encoder_model_128_dual(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_secondary_train.shape[1])
        elif eeg_encoding_dim == 512:
            classifier = convolutional_encoder_model_512_dual(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_secondary_train.shape[1])

    elif args.imageOrwindowed == "transformer":
        print(x_train.shape)
        classifier = EEGViT_raw(x_train.shape[1], x_train.shape[2], 10)
        batch_size, num_epochs = 128, 150 #128, 150

    # Create optimizer with specified parameters
    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, decay=decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    classifier.compile(
        loss={
            'EEG_Class_Labels': 'categorical_crossentropy',
            'EEG_Class_type_Labels': 'categorical_crossentropy'
        },
        optimizer=opt,

        metrics={
            'EEG_Class_Labels': ['accuracy'],
            'EEG_Class_type_Labels': ['accuracy']
        }
    )
    return classifier



clf = KerasClassifier(model=build_model, verbose=0)


param_dist = {
    "batch_size": [32, 64],
    "epochs": [50, 100, 150],
    'model__learning_rate': uniform(1e-4, 9e-3)  # learning rate from 0.0001 to 0.0099

}

random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=1,
    scoring=None  # custom scoring can be passed if needed
)

Y = {
    'EEG_Class_Labels': Y_primary,
    'EEG_Class_type_Labels': Y_secondary
}

random_search.fit(x_train, Y)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

## Save fold results
main_save_dir = os.path.join(output_dir_path, args.input_dir,f"{run_id}",model_name)
results_df = pd.DataFrame(random_search.cv_results_)
print(results_df.head())  # Show first few rows

# Optional: Save to CSV
results_df.to_csv(f"{main_save_dir}/random_search_results.csv", index=False)
