"""
Title: mne_pipeline

Purpose:
    process fileter eeg signals using MNE to remove artifacts from eye movements, muscular effects

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""


import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from mne.preprocessing import ICA
from asrpy import ASR
import warnings

import argparse

warnings.filterwarnings('ignore')

#Argument parser 
parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="root directory", default = "processed_dataset",required=False)
parser.add_argument('--input_dir', type=str, help="input directory", default = "raw_pickle",required=False)
parser.add_argument('--dataset_pickle', type=str, help="sub-0x (where x 1-50)", default = "sub-01" , required=False)
parser.add_argument('--output_dir', type=str, help="output directory", default = "mne_epoch",required=False)

args = parser.parse_args()


## File Directories
dataset_dir_path = f"{args.root_dir}/{args.input_dir}"
dataset_file = f"{args.dataset_pickle}_eeg_data.pkl"
mne_dir_path = f"{args.root_dir}/{args.output_dir}"
mne_file = f"{args.dataset_pickle}_mne_epoch_rejection_idx.pkl"


## Filter parameters
# MNIST_MU sf = 220, 440 samples , MNIST_EP sf = 128, 256 samples , MNIST_IN sf = 128, 256 samples
sample_rate = 1000

# Define notch frequencies and widths
notch_freqs = [50] #, 60]  # Line noise frequencies (50 Hz and harmonics)
notch_widths = [1] #, 2]  # Notch widths (in Hz)

# Define filter parameters
lowcut = 0.4 # 0.4  # Low-cutoff frequency (Hz)
highcut = 60 # 110  # High-cutoff frequency (Hz)

# keys_ = ['T7','P7','T8','P8']
# keys_ext = ['EEGdata_T7','EEGdata_P7','EEGdata_T8','EEGdata_P8']

montage = mne.channels.make_standard_montage('brainproducts-RNP-BA-128')

print(f"{dataset_dir_path}/{dataset_file}")
## Data extraction from the dataset
data_pd = pd.read_pickle(f"{dataset_dir_path}/{dataset_file}")
list_of_Keys = list(data_pd.keys())
keys_to_import = list_of_Keys[2:] # Import all keys except for the label (We dont need to filter that)

all_data_array = data_pd[data_pd['object_class']!=-1]
print(all_data_array['object_class'].value_counts())
print(np.unique(all_data_array['object_class']))

## Extraction of keys from the dataset
processed_data = []
n_channels = len(keys_to_import)
ch_types = ['eeg'] * (n_channels) # - 1)

## MNE REJECTION
info = mne.create_info(ch_names=keys_to_import, sfreq=sample_rate, ch_types=ch_types)
passed_idx = []
rejected_idx = []
verbose = False
for index, row in tqdm(all_data_array.iterrows(), total=all_data_array.shape[0], desc="Processing...", unit="rows"):
    data = np.array(row[keys_to_import].values.tolist(), dtype=object)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, verbose=False)

    # Filter the raw signals and set the eeg reference
    raw.filter(l_freq=lowcut, h_freq=highcut,verbose=False)
    raw.set_eeg_reference(ref_channels='average',ch_type='eeg',projection=False,verbose=False)

    ## Create fixed length Epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=0.05, preload=True,verbose=False)
    epochs_clean = epochs.drop_bad(reject={'eeg': 100e-0},verbose=False)

    ## Append Accepted or reject arrays
    if epochs_clean:
        passed_idx.append(index)
    else:
        rejected_idx.append(index)

mne_epoch_rejection = {'passed':passed_idx,'reject':rejected_idx}

#Declare output path
output_file_path = os.path.join(mne_dir_path,mne_file)
os.makedirs(mne_dir_path, exist_ok=True)

with open(output_file_path, 'wb') as f:
    pickle.dump(mne_epoch_rejection, f)

print("Remaining epochs")
data_array = all_data_array.loc[passed_idx]
print(data_array['object_class'].value_counts())