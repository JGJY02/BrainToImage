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
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)

#Arguments to configure in order to run the tests
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training xxxthresh_(channels)stack(model)_(dataset) 000thresh_AllSlidingCNN_All.pkl / 000thresh_AllStackLstm_All.pkl / 000thresh_AllStackTransformer_All", default = "000thresh_AllSlidingCNN_dual_28_All.pkl" , required=False)
parser.add_argument('--model_type', type=str, help="CNN_spectrogram, CNN_windowed, LSTM, VIT", default = "CNN_windowed" , required=False)
parser.add_argument('--root_dir', type=str, help="Directory to the dataset - CNN_encoder / LSTM_encoder / Transformer", default = "processed_dataset/filter_mne_car/CNN_Encoder",required=False)
parser.add_argument('--model_name', type=str, help="Name of the model LSTM_all_stacked_signals_dual_512_64_ori/ CNN_all_stacked_signals_dual_512_28_ori", default= "CNN_all_stacked_signals_dual_128_28_ori", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/classifiers/crossVal",required=False)
parser.add_argument('--latent_size', type=int, help="Size of the latent, 128 or 512", default = 128, required=False)
parser.add_argument('--num_of_folds', type=int, help="Number of folds", default = 5 , required=False)

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
eeg_encoding_dim = args.latent_size

indexes = [i for i, char in enumerate(args.dataset_pickle) if char == '_']
run_id = args.dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
model_name = args.model_name

#Define Directories to use
dataset_dir_path = f"{args.root_dir}/{args.input_dir}"
dataset_file_path = f"{dataset_dir_path}/{args.dataset_pickle}"
output_dir_path = f"{args.output_dir}/"
model_save_dir = os.path.join(output_dir_path,args.input_dir,f"{run_id}",model_name)
print(f"Saving Models to {model_save_dir}")

#Load relevant dataset and process it
print(f"** Reading data file {dataset_file_path}")
eeg_data = pickle.load(open(f"{dataset_file_path}", 'rb'), encoding='bytes')
x_train, y_train, y_secondary_train, x_test, y_test, y_secondary_test = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']

class_primary_labels = eeg_data['y_train'].shape[1]
label_dictionary = eeg_data['dictionary']

# Combine seperated dataset into one
X = np.vstack((x_train, x_test))
Y_primary = np.vstack((y_train, y_test))
Y_secondary = np.vstack((y_secondary_train, y_secondary_test))


#To stratify
Y = [f"{a}-{b}" for a, b in zip(Y_primary, Y_secondary)] # ensure each unique combination is evenly distributed

print(len(Y))

skf = StratifiedKFold(n_splits = args.num_of_folds)

previous_results = []
saved_indexes = {} # Save the indexes used in the combine dataset

#Train and test across the relevant folds
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    print(f"Current Fold is {i}")
    #Prep the dataset to be used along iwth the save directories
    main_save_dir = os.path.join(output_dir_path,args.input_dir,f"{run_id}",model_name)
    model_save_dir = os.path.join(main_save_dir, f"fold_{i}")
    check_path(model_save_dir)

    x_train = X[train_index]
    y_train = Y_primary[train_index]
    y_secondary_train = Y_secondary[train_index]
    
    x_test = X[test_index]
    y_test = Y_primary[test_index]
    y_secondary_test = Y_secondary[test_index]

    print("Test Primary Balance:", Counter(np.argmax(y_test, axis=1)))
    print("Test Secondary Balance: ", Counter(np.argmax(y_secondary_test, axis=1)))

    # Decide on the settings to be used
    if args.model_type == "CNN_spectrogram":
        classifier = convolutional_encoder_model_spectrogram(x_train.shape[1], x_train.shape[2], 10) # _expanded
        batch_size, num_epochs = 128, 250 #128, 150

        print(x_train[0][0])

    elif args.model_type == "LSTM":

        classifier = LSTM_Classifier_dual_512(x_train.shape[1], x_train.shape[2], 512, y_train.shape[1], y_secondary_train.shape[1])
        batch_size, num_epochs = 128, 100 #128, 150


    elif args.model_type == "CNN_windowed":
        if eeg_encoding_dim == 128:
            classifier = convolutional_encoder_model_128_dual(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_secondary_train.shape[1])
        elif eeg_encoding_dim == 512:
            classifier = convolutional_encoder_model_512_dual(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_secondary_train.shape[1])
        batch_size, num_epochs = 128, 250 #128, 150

    elif args.model_type == "VIT":
        classifier = EEGViT_raw(x_train.shape[1], x_train.shape[2], 10)
        batch_size, num_epochs = 128, 150 #128, 150

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # location for the trained model file
    saved_model_file = os.path.join(model_save_dir, "eeg_classifier_adm5" + '_final' + '.h5')

    # location for the intermediate model files
    filepath = os.path.join(model_save_dir, "eeg_classifier_adm5" + "-model-improvement-{epoch:02d}.h5")  #{epoch:02d}-{val_accuracy:.2f}

    # call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
    callback_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=False, save_best_only=True, mode='max')
    variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 2)

    sgd = optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adm = optimizers.Adam(learning_rate=1e-2, beta_1=0.9, decay=1e-6)

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

    history = classifier.fit(x_train, [y_train, y_secondary_train], validation_data=(x_test, [y_test, y_secondary_test]), epochs=num_epochs, batch_size=batch_size, verbose=1)
    #Save model
    save_model(history.history, f"history_{str(model_name)}_final.pkl",model_save_dir)
    classifier.save(saved_model_file)
    print(f"Fold {i} Complete!")

    ## Results for Fold
    layer_names = ['EEG_feature_BN2','EEG_Class_Labels','EEG_Class_type_Labels']
    encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
    encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

    dataset = tf.data.Dataset.from_tensor_slices(x_test)
    batch_size = 64
    dataset = dataset.batch(batch_size)

    encoded_labels = []
    encoded_type_labels = []

    for batch in tqdm(dataset):
        _, encodedLabel, encodedTypeLabels = encoder_model(batch, training=False)
        encoded_labels.append(encodedLabel)
        encoded_type_labels.append(encodedTypeLabels)
    
    encoded_labels = np.concatenate(encoded_labels, axis=0)
    encoded_type_labels = np.concatenate(encoded_type_labels, axis =0)
    to_labels = np.argmax(y_test,axis=1)  ## since eeg labels are in one-hot encoded format

    #predict for generated labels
    pred_primary_label_array = np.argmax(encoded_labels, axis = 1)
    true_primary_label_array = np.argmax(y_test, axis = 1)

    #For label type
    pred_secondary_label_array = np.argmax(encoded_type_labels, axis = 1)
    true_secondary_label_array = np.argmax(y_secondary_test, axis = 1)

    evaluation ={}
    text_to_save = []
    mean_accuracy_scores = []
    mean_accuracy_type_scores = []

    mean_precision_scores = []
    mean_recall_scores = []
    mean_F1_scores =[]

    mean_type_precision_scores = []
    mean_type_recall_scores = []
    mean_type_F1_scores =[]

    #Assess performance of the models per class label
    for lab in range(class_primary_labels):


        print("Current class label is : ", lab)
        matching_indices = np.where(to_labels == lab)

        conditioning_labels_array = pred_primary_label_array[matching_indices]
        conditioning_labels_type_array = pred_secondary_label_array[matching_indices]
        true_labels_array = true_primary_label_array[matching_indices]
        true_labels_type_array = true_secondary_label_array[matching_indices]


        true_positives = np.sum(true_labels_array == conditioning_labels_array)
        true_positives_type  = np.sum(true_labels_type_array == conditioning_labels_type_array) 


        #F1
        F1_value = f1_score(true_labels_array, conditioning_labels_array, average='macro')
        F1_type_value = f1_score(true_labels_type_array, conditioning_labels_type_array, average='macro')

        #Recall
        recall_value = recall_score(true_labels_array, conditioning_labels_array, average='macro')
        recall_type_value = recall_score(true_labels_type_array, conditioning_labels_type_array, average='macro')

        #Precision
        precision_value = precision_score(true_labels_array,conditioning_labels_array, average='macro')
        precision_type_value = precision_score(true_labels_type_array,conditioning_labels_type_array, average='macro')

        class_acc = true_positives / conditioning_labels_array.shape[0]
        class_type_acc = true_positives_type / conditioning_labels_type_array.shape[0]

        evaluation[lab] = {
            'primary_class_acc': class_acc, 'secondary_class_acc':class_type_acc,\
            'average_F1':F1_value,'average_recall':recall_value,'average_precision':precision_value,\
            'average_type_F1':F1_type_value,'average_type_recall':recall_type_value,'average_type_precision':precision_type_value}

        text_to_print = f"Class {lab} ({label_dictionary[lab]}): classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
                        mean F1 {evaluation[lab]['average_F1']:.2f}, mean recall {evaluation[lab]['average_recall']:.2f}, mean precision {evaluation[lab]['average_precision']:.2f} , \n \
                        mean type F1 {evaluation[lab]['average_type_F1']:.2f}, mean type recall {evaluation[lab]['average_type_recall']:.2f}, mean type precision {evaluation[lab]['average_type_precision']:.2f}"
        text_to_save.append(text_to_print)
        print(text_to_print)



        mean_accuracy_scores.append(class_acc)
        mean_accuracy_type_scores.append(class_type_acc)

        mean_F1_scores.append(evaluation[lab]['average_F1'])
        mean_recall_scores.append(evaluation[lab]['average_recall'])
        mean_precision_scores.append(evaluation[lab]['average_precision'])

        mean_type_F1_scores.append(evaluation[lab]['average_type_F1'])
        mean_type_recall_scores.append(evaluation[lab]['average_type_recall'])
        mean_type_precision_scores.append(evaluation[lab]['average_type_precision'])

    mean_evaluation = {'average_accuracy':np.mean(mean_accuracy_scores), 'average_type_accuracy':np.mean(mean_accuracy_type_scores), \
    'average_f1' : np.mean(mean_F1_scores), 'average_recall' : np.mean(mean_recall_scores), 'average_precision' : np.mean(mean_precision_scores), \
    'average_type_f1' : np.mean(mean_type_F1_scores), 'average_type_recall' : np.mean(mean_type_recall_scores), 'average_type_precision' : np.mean(mean_type_precision_scores), 
        }

    mean_text_to_print = f"Average Class Results: mean classification acc: {mean_evaluation['average_accuracy']:.1%} ,mean type classification acc: {mean_evaluation['average_type_accuracy']:.1%} \n \
    mean F1: {mean_evaluation['average_f1']:.2f}, mean recall: {mean_evaluation['average_recall']:.2f}, mean precision: {mean_evaluation['average_precision']:.2f} \n \
    mean type F1: {mean_evaluation['average_type_f1']:.2f}, mean type recall: {mean_evaluation['average_type_recall']:.2f}, mean type precision: {mean_evaluation['average_type_precision']:.2f} \n"
    
    print(mean_text_to_print)
    text_to_save.append(mean_text_to_print)


    ## Save fold results
    with open(f"{model_save_dir}/results.txt", "w") as file:
        file.write("\n".join(text_to_save) + "\n")
    
    with open(f"{model_save_dir}/fold_indices.txt", "w") as file:
        file.write(f"Dataset used {dataset_file_path}\n\n")
        file.write(f"Fold {i}\n\n")
        file.write(f"Train indices: {train_index.tolist()}\n\n")
        file.write(f"Val indices:   {test_index.tolist()}\n\n")

    saved_indexes[i] = {"train_idx": train_index, 'val_idx': test_index}

    previous_results.append([evaluation, mean_evaluation])

## Summarise folds
mean_per_class = np.zeros((10,8))
mean_fold = np.zeros((1,8))
for fold in range(args.num_of_folds):
    cur_perClass_dict, cur_mean_dict = previous_results[fold]
    
    rows = []
    for lab in range(class_primary_labels):
        vals = np.array(list(cur_perClass_dict[lab].values()))
        rows.append(vals)
    
    result = np.vstack(rows)

    mean_result = np.array(list(cur_mean_dict.values()))
    
    # print(result)
    # print(result.shape)
    mean_per_class = result + mean_per_class
    mean_fold = mean_result + mean_fold
    # print(mean_fold)
    # print(mean_fold.shape)

mean_per_class /= args.num_of_folds
mean_fold /= args.num_of_folds

text_to_save = []

for lab in range(class_primary_labels):
    class_acc, class_type_acc, F1, recall, precision, F1_type, recall_type, precision_type = mean_per_class[lab, :]
    
    text_to_print = f"Class {lab} ({label_dictionary[lab]}): classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
        mean F1 {F1:.2f}, mean recall {recall:.2f}, mean precision {precision:.2f} , \n \
        mean type F1 {F1_type:.2f}, mean type recall {recall_type:.2f}, mean type precision {precision_type:.2f}"
    
    text_to_save.append(text_to_print)
    print(text_to_print)

class_acc, class_type_acc, F1, recall, precision, F1_type, recall_type, precision_type = mean_fold[0,:]
    
mean_text_to_print = f"Average Class Results: classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
    mean F1 {F1:.2f}, mean recall {recall:.2f}, mean precision {precision:.2f} , \n \
    mean type F1 {F1_type:.2f}, mean type recall {recall_type:.2f}, mean type precision {precision_type:.2f}"

text_to_save.append(mean_text_to_print)
print(mean_text_to_print)  

saved_indexes["dataset_pickle"] = args.dataset_pickle
np.save(f"{main_save_dir}/saved_indexes.npy", saved_indexes)

## Save fold results
with open(f"{main_save_dir}/results.txt", "w") as file:
    file.write("\n".join(text_to_save) + "\n")