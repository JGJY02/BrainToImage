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




# Load the dataset
dataset = pickle.load(open(f"{output_file_path}", 'rb'), encoding='bytes')

# Pick a random index
num_samples = len(dataset['x_test_img'])
num_grid = min(100, num_samples)  # Ensure we don't exceed available images


# Randomly select 100 images
indices = np.random.choice(num_samples, num_grid, replace=False)

image = dataset['x_test_img'][indices]
label = dataset['y_test'][indices]
print(label.shape)

label = np.argmax(label, axis = 1)



# Set up the grid
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle("10x10 Grid of Sample Images", fontsize=16)

# Plot each image
for i, ax in enumerate(axes.flat):
    ax.imshow(image[i].astype("uint8"))
    ax.set_title(f"{dataset['dictionary'][label[i]]}", fontsize=8)
    ax.axis("off")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
