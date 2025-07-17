from sklearn.model_selection import StratifiedKFold
import pickle
import os 
import argparse
import numpy as np

#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset", default = "datasets/processed_dataset/filter_mne_car/All",required=False)
parser.add_argument('--fold_indexes', type=str, help="Obtain indexes of fold", default = "trained_models/crossVal/Transformer_512_dual/saved_indexes.npy" , required=False)

parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "datasets/processed_dataset/filter_mne_car/crossVal",required=False)
parser.add_argument('--output_name', type=str, help="Directory to output", default = "Transformer_64_dual_2",required=False)
parser.add_argument('--num_of_folds', type=int, help="Number of folds", default = 5 , required=False)

args = parser.parse_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

## load the fold indexes

index_dictionary = np.load(args.fold_indexes, allow_pickle=True).item()
dataset_pickle = index_dictionary['dataset_pickle']

## load the Things-EEG trainig data
eeg_data_file = f"{args.root_dir}/{dataset_pickle}"
output_data_dir = f"{args.output_dir}/{args.output_name}"

print(f"Reading data file {eeg_data_file}")
eeg_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
x_train_eeg_data, y_primary_train_data, y_secondary_train_data, x_test_eeg_data, y_primary_test_data, y_secondary_test_data = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']
train_imgs, test_imgs = eeg_data['x_train_img'] , eeg_data['x_test_img']

label_dictionary = eeg_data['dictionary']
class_primary_labels = eeg_data['y_train'].shape[1]

X_eeg = np.vstack((x_train_eeg_data, x_test_eeg_data))
X_img = np.vstack((train_imgs, test_imgs))

Y_primary = np.vstack((y_primary_train_data, y_primary_test_data))
Y_secondary = np.vstack((y_secondary_train_data, y_secondary_test_data))

# X_eeg = X_eeg[:100]
# Y_primary = Y_primary[:100]
# Y_secondary = Y_secondary[:100]
# X_img = X_img[:100]

Y_eeg = [f"{a}-{b}" for a, b in zip(Y_primary, Y_secondary)]  # or tuple: list(zip(y1, y2))


num_of_class_labels = Y_primary.shape[1]
num_of_class_type_labels = Y_secondary.shape[1]
skf = StratifiedKFold(n_splits = args.num_of_folds)

for i in range(args.num_of_folds):

    #Clear resources after each fold
    train_index = index_dictionary[i]["train_idx"]
    test_index = index_dictionary[i]["val_idx"]

    #Clear resources after each fold
    print(f"Current Fold is {i}")
    dataset_save_folder = os.path.join(output_data_dir, f"fold{i}")
    dataset_save_path = os.path.join(dataset_save_folder, dataset_pickle)


    x_eeg_train = X_eeg[train_index]
    y_primary_train = Y_primary[train_index]
    y_secondary_train = Y_secondary[train_index]
    
    x_eeg_test = X_eeg[test_index]
    y_primary_test = Y_primary[test_index]
    y_secondary_test = Y_secondary[test_index]

    x_img_train = X_img[train_index]
    x_img_test = X_img[test_index]



    print(f"The dimensions of each dataset is x_train_eeg: {x_eeg_train.shape}, x_test_eeg: {x_eeg_test.shape}, x_train_img: {x_img_train.shape}, x_test_eeg: {x_img_test.shape} , y_test: {y_primary_test.shape} , y_train: {y_primary_train.shape} , y_secondary_train: {y_secondary_train.shape}, y_secondary_test: {y_secondary_test.shape}")

    print(f"*** Writing {dataset_save_folder}")
    os.makedirs(dataset_save_folder, exist_ok=True)
    data_out = {'x_train_eeg':x_eeg_train, 'x_test_eeg':x_eeg_test, 'x_train_img':x_img_train, 'x_test_img':x_img_test, 'y_train':y_primary_train, 'y_test':y_primary_test, 'y_secondary_train':y_secondary_train ,'y_secondary_test':y_secondary_test , 'dictionary':label_dictionary} #{'x_test':train_data,'y_test':labels}
    
    with open(f"{dataset_save_path}", 'wb') as f:
        pickle.dump(data_out, f)
    
    with open(f"{dataset_save_folder}/fold_indices.txt", "w") as file:
        file.write(f"Dataset used {eeg_data_file}\n\n")
        file.write(f"Fold {i}\n\n")
        file.write(f"Train indices: {train_index.tolist()}\n\n")
        file.write(f"Val indices:   {test_index.tolist()}\n\n")

np.save(f"{output_data_dir}/saved_indexes.npy", index_dictionary)
