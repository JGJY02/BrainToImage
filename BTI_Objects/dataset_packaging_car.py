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

parser.add_argument('--dataset_pickle', type=str, help="sub-0x (where x 1-50)", default = "All" , required=False)
parser.add_argument('--car_filter_percent', type=float, help="ratio to filter baseline 0.934", default = 0.00, required=False)
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

if args.car_filter_percent == 0:
    car_filter_percent_string = "000"
else:
    car_filter_percent_string = int(args.car_filter_percent * 1000)
car_file_path = f"{output_dir_path}/{args.dataset_pickle}_car_correlation_output.pkl"

output_prefix = f"{car_filter_percent_string}{args.output_prefix}"
output_file_path = f"{output_dir_path}/{output_prefix}_{args.dataset_pickle}.pkl"





## Car parameters


print(f"*** Processing files at car ratio of {args.car_filter_percent}")



files = [f for f in os.listdir(dataset_dir_path) if os.path.isfile(os.path.join(dataset_dir_path, f))]

df_copy = pd.read_pickle(f"{dataset_dir_path}/{files[0]}") #filtered_{output_file}")
print(df_copy.info())
list_of_keys = list(df_copy.keys())
list_of_keys = list_of_keys[2:65]
class_labels = list(dict.fromkeys(df_copy['object_class']))
softmax_labels = range(10) # range(len(class_labels))
softmax_dict = {i: item for i, item in enumerate(class_labels, start=0)} #Preset the dictionary

print(softmax_dict)

# print(list_of_keys[:10])
# for i in softmax_labels:
#     print(softmax_dict[i])

label = 'object_class'

fraction = 1


#sampled_df.info()

scales = np.arange(0.4, 60, 0.233)
WINDOW_SIZE = 30
img_size = (28,28)

obj_images_comp = []
feature_data_comp = []
label_data_comp = []
files = files


#Utilise own dataset creation method
splitPercent = 0.2



train_obj_images_comp = []
train_feature_data_comp = []
train_label_data_comp = []

test_obj_images_comp = []
test_feature_data_comp = []
test_label_data_comp = []


for file in files:
    print(f"On current file {file}")
    # obj_images = []
    # feature_data = []
    # label_data = []

    train_feature_array = []
    train_label_array = []
    train_img_array = []

    test_feature_array = []
    test_label_array = []
    test_img_array = []

    df = pd.read_pickle(f"{dataset_dir_path}/{file}") #filtered_{output_file}")
    # sampled_indexes = df_copy[df_copy['corr_mean_all'] > args.car_filter_percent].groupby(label).apply(lambda x: x.sample(frac=fraction)).index.get_level_values(1).tolist()
    # sampled_df = df_copy.loc[sampled_indexes]
    sampled_df = df_copy

    for class_label in tqdm(softmax_labels):
        class_df = sampled_df[sampled_df[label]== softmax_dict[class_label]]

        dir_to_extract_images = os.path.join(args.img_root_dir, softmax_dict[class_label])
        
        for idx, row in class_df.iterrows():
            X = row[list_of_keys]
            img_name =row['object_name']
            X = np.array(X.tolist(), dtype=np.float32)

            num_of_test_samples = int(len(range(X.shape[1] - WINDOW_SIZE + 1)) * splitPercent)
            test_sample_idx = np.random.choice(len(range(X.shape[1] - WINDOW_SIZE + 1)), size = num_of_test_samples, replace = False)


            for i in range(X.shape[1] - WINDOW_SIZE + 1):

 

                w_data = X[:, i:i+WINDOW_SIZE]
                w_data = np.transpose(w_data, (1,0))
                # print(w_data.dtype)
                
                # print(w_data.shape)
                # feature_data.append(w_data)
                # label_data.append(to_categorical(int(class_label),num_classes=len(softmax_labels)))

                
                # img_path = os.path.join(dir_to_extract_images, img_name)
                # try:
                #     img = Image.open(img_path).resize(img_size)
                #     img_array = np.array(img)  # Normalize

                #     obj_images.append(img_array)

                # except Exception as e:
                #     print(f"Error loading {img_path}: {e}")

                img_path = os.path.join(dir_to_extract_images, img_name)
                try:
                    img = Image.open(img_path).resize(img_size)
                    img_array = np.array(img)  # Normalize
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

                # print(i)
                # print(i not in test_sample_idx)

                if i not in test_sample_idx:
                    train_feature_array.append(w_data)
                    train_label_array.append(to_categorical(int(class_label),num_classes=len(softmax_labels)))
                    train_img_array.append(img_array)
                
                else:
                    test_feature_array.append(w_data)
                    test_label_array.append(to_categorical(int(class_label),num_classes=len(softmax_labels)))
                    test_img_array.append(img_array)
                

    
    # obj_images_comp.append(np.array(obj_images))
    # feature_data_comp.append(np.array(feature_data))
    # label_data_comp.append(np.array(label_data))

    train_obj_images_comp.append(np.array(train_img_array))
    train_feature_data_comp.append(np.array(train_feature_array))
    train_label_data_comp.append(np.array(train_label_array))

    test_obj_images_comp.append(np.array(test_img_array))
    test_feature_data_comp.append(np.array(test_feature_array))
    test_label_data_comp.append(np.array(test_label_array))


# obj_images_comp = np.vstack(obj_images_comp)
# feature_data_comp = np.vstack(feature_data_comp)
# label_data_comp = np.vstack(label_data_comp)

train_obj_images_comp = np.vstack(train_obj_images_comp)
train_feature_data_comp = np.vstack(train_feature_data_comp)
train_label_data_comp = np.vstack(train_label_data_comp)

test_obj_images_comp = np.vstack(test_obj_images_comp)
test_feature_data_comp = np.vstack(test_feature_data_comp)
test_label_data_comp = np.vstack(test_label_data_comp)

x_train_eeg = np.array(train_feature_data_comp)
x_train_img = np.array(train_obj_images_comp)
y_train = np.array(train_label_data_comp).astype(np.uint8)

x_test_eeg = np.array(test_feature_data_comp)
x_test_img = np.array(test_obj_images_comp)
y_test = np.array(test_label_data_comp).astype(np.uint8)

#Shuffle the arrays
train_shuffled_indices = np.random.permutation(x_train_eeg.shape[0])
test_shuffled_indices = np.random.permutation(x_test_eeg.shape[0])

x_train_eeg = x_train_eeg[train_shuffled_indices]
x_train_img = x_train_img[train_shuffled_indices]
y_train = y_train[train_shuffled_indices]

x_test_eeg = x_test_eeg[test_shuffled_indices]
x_test_img = x_test_img[test_shuffled_indices]
y_test = y_test[test_shuffled_indices]



# train_data_1 = np.array(feature_data_comp)
# train_data_2 = np.array(obj_images_comp)
# labels = np.array(label_data_comp).astype(np.uint8)

# x_train_eeg, x_test_eeg, x_train_img, x_test_img, y_train, y_test = train_test_split(train_data_1, train_data_2, labels, test_size=0.1, random_state=42)

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