
import argparse
import mne
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--input_root_dir', type=str, help="Directory to the dataset", default = "raw_dataset/eeg_dataset",required=False)
parser.add_argument('--file', type=str, help="Directory to the dataset", default = "sub-11",required=False)
parser.add_argument('--output_root_dir', type=str, help="Directory to output", default = "processed_dataset",required=False)
args = parser.parse_args()

#Declare input path
input_dir = f"{args.input_root_dir}/{args.file}/eeg"
vhdr_file = f"{args.file}_task-rsvp_eeg.vhdr"
csv_file = f"{args.file}_task-rsvp_events.csv"
vhdr_path = os.path.join(input_dir, vhdr_file)
csv_path = os.path.join(input_dir,csv_file)

#Read Label file
processed_data = []
keys = ['object', 'stimname', 'time_stimon', 'time_stimoff']
for chunk in tqdm(pd.read_csv(csv_path, chunksize=10000, usecols=keys)):
    processed_data.append(chunk)

label_df = pd.concat(processed_data)

print(label_df.info())

#Read EEG file
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
print(raw.info)

data = raw.get_data()  # EEG signals (numpy array)
ch_names = raw.ch_names # Channel names
times = raw.times

print(data.shape)
print(times.shape)

print("Creating dataframe")



data_dict = {ch_names[i]: data[i,:] for i in range(63)}
data_df = pd.DataFrame(data_dict)

chunked_data = []
index_to_test = range(21911, len(label_df['time_stimon']))

for i in tqdm(range(len(label_df['time_stimon']))):

    if i == 0:
        start_time = int(np.round(label_df['time_stimon'][i] * 1000))
        print(start_time)

    new_row = {}
    start = int(np.round(label_df['time_stimon'][i] * 1000)) - start_time # starting time
    end = start + 50 #50 ms window
    for channel in ch_names:
        new_row[channel] = data_df[channel][start:end] * 1e6 ## convert volts to micro volts
    

    chunked_data.append(new_row)
    
processed_df = pd.DataFrame(chunked_data)
processed_df.insert(0, 'object_name', label_df['stimname'])
extracted = [s.split('_')[0] for s in label_df['stimname']]


processed_df.insert(0, 'object_class', extracted)



print(processed_df.info())
print(processed_df.shape)

# Declare output path
output_dir = f"{args.output_root_dir}/raw_pickle"
output_file = f"{args.file}_eeg_data.pkl"

output_file_path = os.path.join(output_dir,output_file)
os.makedirs(output_dir, exist_ok=True)

with open(output_file_path, "wb") as f:
    pickle.dump(processed_df, f)

# # View converted data
# df = pd.read_pickle(output_file_path)
# print(df['data'].shape)
# print(df['times'].shape)
# print(np.where(df['times'] == 61.653))
# print(len(df['ch_names']))
# print(df['sfreq'])


