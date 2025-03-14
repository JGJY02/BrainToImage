import csv
import os
import pickle
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pywt
import seaborn as sns
from keras.utils import to_categorical
# from scipy.integrate import simps
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper_functions as hf


import argparse
import pywt
import matplotlib.pyplot as plt
import cv2


# warnings.filterwarnings('ignore')

#Argument parser 
parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="root directory", default = "processed_dataset",required=False)
parser.add_argument('--input_dir', type=str, help="input directory", default = "filtered_pickle",required=False)
parser.add_argument('--mne_dir', type=str, help="mne directory", default = "mne_epoch",required=False)

parser.add_argument('--dataset_pickle', type=str, help="sub-0x (where x 1-50)", default = "sub-01" , required=False)
parser.add_argument('--car_filter_percent', type=float, help="ratio to filter baseline 0.934", default = 0.75, required=False)
parser.add_argument('--filter_from_scratch', type = bool, help="Filter from scratch or load processed file", default = True)

parser.add_argument('--output_prefix', type=str, help="Name of the output file produced", default= "thresh_AllStackLstm", required=False)
parser.add_argument('--output_dir', type=str, help="output directory", default = "filter_mne_car",required=False)

args = parser.parse_args()

## File Directories
dataset_dir_path = f"{args.root_dir}/{args.input_dir}"
mne_dir_path = f"{args.root_dir}/{args.mne_dir}"

dataset_file = f"filtered_{args.dataset_pickle}_eeg_data.pkl"
mne_file = f"{args.dataset_pickle}_mne_epoch_rejection_idx.pkl"


dataset_file_path = f"{dataset_dir_path}/{dataset_file}"
mne_file_path = f"{mne_dir_path}/{mne_file}"


output_dir_path = f"{args.root_dir}/{args.output_dir}/{args.dataset_pickle}"

car_filter_percent_string = int(args.car_filter_percent * 1000)
car_file_path = f"{output_dir_path}/{args.dataset_pickle}_car_correlation_output.pkl"

output_prefix = f"{car_filter_percent_string}{args.output_prefix}"
output_file_path = f"{output_dir_path}/{output_prefix}_{args.dataset_pickle}"





## Car parameters
class_labels = [0,1,2,3,4,5,6,7,8,9]
label = 'object_class'

print(f"*** Processing files at car ratio of {args.car_filter_percent}")
if args.filter_from_scratch == True:

    print(f"*** Reading dataset file {dataset_file_path}") #filtered_{output_file}")
    df = pd.read_pickle(f"{dataset_file_path}") #filtered_{output_file}")
    df = df[df[label]!=-1]
    df.info()

    ## Key Setup
    list_of_Keys = list(df.keys())
    list_of_Keys = list_of_Keys[2:]
    # corr_keys_ = [f"{key}_corr" for key in keys_]
    corr_keys_all = [f"{key}_corr" for key in list_of_Keys]

    print("** Checking signals before MNE check")
    print(df[label].value_counts())
    print(df[label].value_counts().sum())

    ## get all passed eeg signals from MNE pipeline.
    print(f"*** Reading MNE file {mne_file_path}")
    epoched_indexs = pd.read_pickle(f"{mne_file_path}")

    df_pass = df.loc[epoched_indexs['passed']]
    print("** Checking signals passed MNE check")
    print(df_pass[label].value_counts())
    print(df_pass[label].value_counts().sum())

    df_reject = df.loc[epoched_indexs['reject']]
    print("** Checking signals failed MNE check")
    print(df_reject[label].value_counts())
    print(df_reject[label].value_counts().sum())

    ## sample if required. fraction = 1 to take all data
    fraction = 1
    sampled_indexes = df.groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
    sampled_df = df.loc[sampled_indexes]
    #sampled_df.info()
    df = None

    df_copy = sampled_df.copy()

    for key in list_of_Keys:
        df_copy[f"{key}_corr"] = pd.NA
    df_copy[f"erp"] = pd.NA

    for idx, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], desc="Processing...", unit="rows"):
        corr_data = row[list_of_Keys]
        arr = np.stack(corr_data.values)
        # Calculate ERP
        car = np.mean(arr, axis=0)
        # Baseline correction (using first 125 ms as baseline)
        baseline = np.mean(car[:16])
        car_corrected = car - baseline

        for key in list_of_Keys:
            # car_subtracted = corr_data[key] - car_corrected
            # correlation = np.corrcoef(car_subtracted, car_corrected)[0, 1]
            correlation = np.corrcoef(corr_data[key], car_corrected)[0, 1]
            row[f"{key}_corr"] = correlation
        #corr_data[label] = row[label]
        row['erp'] = car_corrected

        df_copy.loc[idx] = row

    df_copy.info()

    # df_copy['corr_mean_core'] = df_copy[corr_keys_].mean(axis=1)
    df_copy['corr_mean_all'] = df_copy[corr_keys_all].mean(axis=1)






    ## Save Car correlation calculation so that we we can filter out as need be later on
    os.makedirs(output_dir_path, exist_ok=True)

    print(f"writing {car_file_path}")
    with open(f"{car_file_path}", 'wb') as f:
        pickle.dump(df_copy, f)


df_copy = pd.read_pickle(car_file_path) #filtered_{output_file}")
fraction = 1
sampled_indexes = df_copy[df_copy['corr_mean_all'] > args.car_filter_percent].groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
sampled_df = df_copy.loc[sampled_indexes]

#sampled_df.info()
print(sampled_df[label].value_counts())
print(sampled_df[label].value_counts().sum())

scales = np.arange(0.4, 60, 0.233)
# split data to test train
WINDOW_SIZE = 30

feature_data = []
label_data = []
for class_label in class_labels:
    class_df = sampled_df[sampled_df[label]==class_label]
    
    for idx, row in tqdm(class_df.iterrows()):
        X = row[list_of_Keys]
        X = np.array(X.tolist(), dtype=np.float32)

        for i in range(X.shape[1] - WINDOW_SIZE + 1):
            w_data = X[:, i:i+WINDOW_SIZE]
            w_data = np.transpose(w_data, (1,0))
            # print(w_data.dtype)
            
            # print(w_data.shape)
            feature_data.append(w_data)
            label_data.append(to_categorical(int(class_label),num_classes=len(class_labels)))





train_data = np.array(feature_data)
labels = np.array(label_data).astype(np.uint8)
# train_data = np.array(feature_data)
# print(train_data.shape)
# print(train_data[0].shape)
# plt.imshow(train_data[0][2])
# plt.show()
# train_data = np.array(feature_data)
# labels = np.array(label_data).astype(np.uint8)

print(train_data.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, random_state=42)

print(f"The dimensions of each dataset is x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape} , y_test: {y_test.shape}")

print(f"*** Writing {output_file_path}")
data_out = {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test} #{'x_test':train_data,'y_test':labels}
with open(f"{output_file_path}", 'wb') as f:
    pickle.dump(data_out, f)
