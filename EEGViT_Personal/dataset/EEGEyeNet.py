from torch.utils.data import Dataset
import torch
import numpy as np
import pickle

class EEGEyeNetDataset(Dataset):
    def __init__(self, data_file,transpose = False):
        self.data_file = data_file
        print('loading data...')
        eeg_data = pickle.load(open(f"{self.data_file}", 'rb'), encoding='bytes')
        # print(eeg_data)
        # self.trainX = f['EEG']
        # self.trainY = f['labels']
        self.trainX = eeg_data['x_train_eeg']
        self.trainY = eeg_data['y_train']
        print(self.trainY)
        if transpose:
            self.trainX = np.transpose(self.trainX, (0,2,1))[:,np.newaxis,:,:]
        else:
            self.trainX = self.trainX[:,np.newaxis,:,:]

    def __getitem__(self, index):
        # Read a single sample of data from the data array
        X = torch.from_numpy(self.trainX[index]).float()
        y = torch.from_numpy(self.trainY[index]).float()
        # Return the tensor data
        return (X,y,index)

    def __len__(self):
        # Compute the number of samples in the data array
        return len(self.trainX)


class MultiClassDataset(Dataset):
    def __init__(self, x, y, y_secondary,transpose = False):

        self.x = x
        self.y = y
        self.y_secondary = y_secondary
        if transpose:
            self.x = np.transpose(self.x, (0,2,1))[:,np.newaxis,:,:]
        else:
            self.x = self.x[:,np.newaxis,:,:]

    def __getitem__(self, index):
        # Read a single sample of data from the data array
        x = torch.from_numpy(self.x[index]).float()
        y1 = torch.from_numpy(self.y[index]).float()
        y2 = torch.from_numpy(self.y_secondary[index]).float()
        # Return the tensor data
        return (x , y1, y2,index)

    def __len__(self):
        # Compute the number of samples in the data array
        return len(self.x)