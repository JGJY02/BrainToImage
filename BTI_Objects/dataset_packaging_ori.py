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

from PIL import Image

# warnings.filterwarnings('ignore')

#Argument parser 
parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="root directory", default = "processed_dataset",required=False)
parser.add_argument('--input_dir', type=str, help="input directory", default = "filtered_pickle",required=False)
parser.add_argument('--img_root_dir', type=str, help="image input directory", default = "raw_dataset/object_dataset",required=False)
parser.add_argument('--mne_dir', type=str, help="mne directory", default = "mne_epoch",required=False)

parser.add_argument('--dataset_pickle', type=str, help="sub-0x (where x 1-50)", default = "sub-01" , required=False)
parser.add_argument('--car_filter_percent', type=float, help="ratio to filter baseline 0.934", default = 0.00, required=False)
parser.add_argument('--filter_from_scratch', type = bool, help="Filter from scratch or load processed file", default = True)

parser.add_argument('--output_prefix', type=str, help="Name of the output file produced", default= "thresh_AllStackLstm_20", required=False)
parser.add_argument('--output_dir', type=str, help="output directory", default = "filter_mne_car",required=False)

args = parser.parse_args()

## File Directories
dataset_dir_path = f"{args.root_dir}/{args.input_dir}"
mne_dir_path = f"{args.root_dir}/{args.mne_dir}"

dataset_file = f"{args.dataset_pickle}_eeg_data.pkl"
mne_file = f"{args.dataset_pickle}_mne_epoch_rejection_idx.pkl"


dataset_file_path = f"{dataset_dir_path}/{dataset_file}"
mne_file_path = f"{mne_dir_path}/{mne_file}"


output_dir_path = f"{args.root_dir}/{args.output_dir}/{args.dataset_pickle}"

if args.car_filter_percent == 0:
    car_filter_percent_string = "000"
else:
    car_filter_percent_string = int(args.car_filter_percent * 1000)
car_file_path = f"{output_dir_path}/{args.dataset_pickle}_car_correlation_output.pkl"

output_prefix = f"{car_filter_percent_string}{args.output_prefix}"
output_file_path = f"{output_dir_path}/{output_prefix}_{args.dataset_pickle}.pkl"





## Car parameters


print(f"*** Processing files at car ratio of {args.car_filter_percent}")



df_copy = pd.read_pickle(car_file_path) #filtered_{output_file}")
print(df_copy.info())
list_of_keys = list(df_copy.keys())
list_of_keys = list_of_keys[2:65]
class_labels = list(dict.fromkeys(df_copy['object_class']))
softmax_labels = range(10) # range(len(class_labels))
softmax_dict = {i: item for i, item in enumerate(class_labels, start=0)}

print(list_of_keys[:10])
for i in softmax_labels:
    print(softmax_dict[i])

label = 'object_class'

fraction = 1
sampled_indexes = df_copy[df_copy['corr_mean_all'] > args.car_filter_percent].groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
sampled_df = df_copy.loc[sampled_indexes]

#sampled_df.info()
print(sampled_df[label].value_counts())
print(sampled_df[label].value_counts().sum())

scales = np.arange(0.4, 60, 0.233)
# split data to test train
WINDOW_SIZE = 10


img_size = (28,28)
obj_images = []
feature_data = []
label_data = []
# for class_label in softmax_labels:
#     class_df = sampled_df[sampled_df[label]== softmax_dict[class_label]]
    
#     for idx, row in tqdm(class_df.iterrows()):
#         X = row[list_of_keys]
#         X = np.array(X.tolist(), dtype=np.float32)

#         for i in range(X.shape[1] - WINDOW_SIZE + 1):
#             w_data = X[:, i:i+WINDOW_SIZE]
#             w_data = np.transpose(w_data, (1,0))
#             # print(w_data.dtype)
            
#             # print(w_data.shape)
#             feature_data.append(w_data)
#             label_data.append(to_categorical(int(class_label),num_classes=len(softmax_labels)))


for class_label in softmax_labels:
    class_df = sampled_df[sampled_df[label]== softmax_dict[class_label]]

    dir_to_extract_images = os.path.join(args.img_root_dir, softmax_dict[class_label])
    
    for idx, row in tqdm(class_df.iterrows()):
        X = row[list_of_keys]
        img_name =row['object_name']
        X = np.array(X.tolist(), dtype=np.float32)

        for i in range(X.shape[1] - WINDOW_SIZE + 1):
            w_data = X[:, i:i+WINDOW_SIZE]
            w_data = np.transpose(w_data, (1,0))
            # print(w_data.dtype)
            
            # print(w_data.shape)
            feature_data.append(w_data)
            label_data.append(to_categorical(int(class_label),num_classes=len(softmax_labels)))

            
            img_path = os.path.join(dir_to_extract_images, img_name)
            try:
                img = Image.open(img_path).resize(img_size)
                img_array = np.array(img)  # Normalize

                # print(softmax_dict[class_label], class_label)
                obj_images.append(img_array)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        







train_data_1 = np.array(feature_data)
train_data_2 = np.array(obj_images)
labels = np.array(label_data).astype(np.uint8)

# print(train_data.shape)
# print(train_data[0].shape)
# plt.imshow(train_data[0][2])
# plt.show()
# train_data = np.array(feature_data)
# labels = np.array(label_data).astype(np.uint8)

# print(train_data.shape)
# print(labels.shape)

x_train_eeg, x_test_eeg, x_train_img, x_test_img, y_train, y_test = train_test_split(train_data_1, train_data_2, labels, test_size=0.1, random_state=42)

print(f"The dimensions of each dataset is x_train_eeg: {x_train_eeg.shape}, x_test_eeg: {x_test_eeg.shape}, x_train_img: {x_train_img.shape}, x_test_eeg: {x_test_img.shape} , y_test: {y_test.shape} , y_train: {y_train.shape} ")

print(f"*** Writing {output_file_path}")
os.makedirs(output_dir_path, exist_ok=True)
data_out = {'x_train_eeg':x_train_eeg, 'x_test_eeg':x_test_eeg, 'x_train_img':x_train_img, 'x_test_img':x_test_img, 'y_train':y_train,'y_test':y_test, 'dictionary':softmax_dict} #{'x_test':train_data,'y_test':labels}
with open(f"{output_file_path}", 'wb') as f:
    pickle.dump(data_out, f)

# print(x_train[:10])
# print(y_train.shape)

#Extract corresponding images

# file path to images
# img_size = (28,28)
# obj_images = []
# obj_labels = []

# for class_label in softmax_labels:
#     dir_to_extract_images = os.path.join(args.img_root_dir, softmax_dict[class_label])
#     for img_name in os.listdir(dir_to_extract_images):
#         img_path = os.path.join(dir_to_extract_images, img_name)
#         try:
#             img = Image.open(img_path).resize(img_size)
#             img_array = np.array(img) / 255.0  # Normalize
#             obj_images.append(img_array)
#             obj_labels.append(class_label)
#         except Exception as e:
#             print(f"Error loading {img_path}: {e}")

# # Convert lists to NumPy arrays
# obj_images = np.array(obj_images)
# obj_labels = np.array(obj_labels)

# # Save to a pickle file
# x_obj_train, x_obj_test, y_obj_train, y_obj_test = train_test_split(obj_images, obj_labels, test_size=0.1, random_state=42)

# obj_data_out = {'x_train':x_obj_train,'x_test':x_obj_test,'y_train':y_obj_train,'y_test':y_obj_test} #{'x_test':train_data,'y_test':labels}
# object_dataset_file_path = os.path.join(output_dir_path, "object_img_dataset_unsorted.pkl")
# with open(object_dataset_file_path, "wb") as f:
#     pickle.dump(obj_data_out, f)

# print(x_obj_train.shape)
# print(y_obj_train.shape)
# print(labels)