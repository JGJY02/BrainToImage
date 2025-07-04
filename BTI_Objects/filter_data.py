import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch
import helper_functions as hf
import pickle
import os
import argparse

## Argument parser 
parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "processed_dataset/raw_pickle",required=False)
parser.add_argument('--dataset_pickle', type=str, help="sub-0x (where x 1-50)", default = "ALL" , required=False)
parser.add_argument('--output_dir', type=str, help="Directory to save processed file", default = "processed_dataset/filtered_pickle",required=False)

args = parser.parse_args()



print(os.getcwd())

## File directories
input_dir = args.input_dir
output_dir = args.output_dir
# pickle_name = f"sub-{args.dataset_pickle}"

## Filter parameters
label_field = 'digit_label'
sample_rate = 128  #Hz
# Define notch frequencies and widths
notch_freqs = [50] #, 60]  # Line noise frequencies (50 Hz and harmonics)
notch_widths = [1] #, 2]  # Notch widths (in Hz)
# Define butterworth filter parameters
butter_order = 5 # 4
lowcut = 0.4 # 0.4  # Low-cutoff frequency (Hz)
highcut = 60 # 110  # High-cutoff frequency (Hz)

if args.dataset_pickle == "ALL":
    files_to_process = [f"{i:02}" for i in range(40, 51)]
else:
    files_to_process = [args.file]

for file in files_to_process:
    ## Access filter data 
    data_pd = pd.read_pickle(f"{input_dir}/sub-{file}_eeg_data.pkl") #Change file_to_process to match output later on
    list_of_Keys = list(data_pd.keys())
    keys_to_import = list_of_Keys[2:] # Import all keys except for the label (We dont need to filter that)



    #filter_data.info()

    filter_data_copy = data_pd.copy()
    print(filter_data_copy.info())

    ## Filter the data
    for idx, row in tqdm(filter_data_copy.iterrows(), total=filter_data_copy.shape[0], desc="Processing rows", unit="row"):
        filtered = hf.apply_notch_filter(row[keys_to_import],sample_rate,notch_freqs=notch_freqs,notch_widths=notch_widths)
        filtered = hf.apply_butterworth_filter(filtered,sample_rate,lowcut,highcut,order=butter_order)
        filtered['object_class'] = row['object_class']
        filtered['object_name'] = row['object_name']
        filter_data_copy.loc[idx] = filtered

    print(filter_data_copy.info())



    #Declare output path
    output_dir = f"{output_dir}"
    output_file = f"filtered_sub-{file}_eeg_data.pkl"
    output_file_path = os.path.join(output_dir,output_file)
    os.makedirs(output_dir, exist_ok=True)

    ## Dump Output
    with open(output_file_path, 'wb') as f:
        pickle.dump(filter_data_copy, f)
  
