
import pickle

import numpy as np
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import sys

main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

os.chdir(main_dir) #Jared Edition
print(main_dir)

from models.EEGViT_pretrained import EEGViT_pretrained_512


if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"


class MainClassWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        prim_out, _, _ = self.base_model(x)
        return prim_out # shape: [batch]

class subClassWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        _, _, sub_out = self.base_model(x)
        return sub_out # shape: [batch]


columns_to_look = ['EEG.Fp1',	'EEG.Fz',	'EEG.F3',	'EEG.F7'	,'EEG.FT9'	,'EEG.FC5'	,'EEG.FC1','EEG.C3','EEG.T7' ,'EEG.TP9'	,
                    'EEG.CP5'	,'EEG.CP1',	'EEG.Pz'	,'EEG.P3'	,'EEG.P7',	'EEG.O1',	'EEG.Oz',	'EEG.O2',	'EEG.P4', 'EEG.P8'	,
                    'EEG.TP10',	'EEG.CP6','EEG.CP2',	'EEG.C4'	,'EEG.T8',	'EEG.FT10',	'EEG.FC6', 'EEG.FC2',	'EEG.F4', 'EEG.F8',	
                    'EEG.Fp2', 'EEG.AF7', 'EEG.AF3', 'EEG.AFz', 'EEG.F1', 'EEG.F5', 'EEG.FT7', 'EEG.FC3', 'EEG.C1', 'EEG.C5', 
                    'EEG.TP7', 'EEG.CP3', 'EEG.P1', 'EEG.P5', 'EEG.PO7', 'EEG.PO3','EEG.POz', 'EEG.PO4', 'EEG.PO8', 'EEG.P6', 
                   'EEG.P2', 'EEG.CPz', 'EEG.CP4', 'EEG.TP8', 'EEG.C6', 'EEG.C2', 'EEG.FC4', 'EEG.FT8', 'EEG.F6', 'EEG.AF8', 
                   'EEG.AF4', 'EEG.F2', 'EEG.FCz']



batch_size = 1

model_pth = "trained_models/Transformer_512_dual_large/eeg_classifier_adm5_final.pth"



eeg_data = pickle.load(open("dataset/000thresh_AllStack_large_Transformer_dual_2_64.pkl", 'rb'), encoding='bytes')
base_model = EEGViT_pretrained_512(eeg_data['y_train'].shape[1], eeg_data['y_secondary_train'].shape[1])
base_model.load_state_dict(torch.load(model_pth))


Y_primary = np.argmax(eeg_data['y_train'], axis = 1)
Y_secondary = np.argmax(eeg_data['y_secondary_train'], axis = 1)

labels_to_get_gradients = [Y_primary, Y_secondary]

X = eeg_data['x_train_eeg']


for idx, Y in enumerate(labels_to_get_gradients):
    res_per_channel = []
    res_summarize_channel = []
    if idx == 0:
        name = "primary"
        model = MainClassWrapper(base_model=base_model)
    elif idx == 1:
        name = "secondary"
        model = subClassWrapper(base_model=base_model)
    
    model = model.to(device)
    model.eval()

    for label in tqdm(np.unique(Y), total=len(np.unique(Y))):
        print(f"\nCurrently working on label: {label}")
        total_attributions = []
        indices = np.where(Y == label)[0]
        Y_filtered = Y[indices]
        X_filtered = X[indices]
        X_filtered = X_filtered[:,np.newaxis,:,:]

        X_filtered = torch.tensor(X_filtered, dtype=torch.float32).to(device)
        Y_filtered = torch.tensor(Y_filtered, dtype=torch.long).to(device)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_filtered, Y_filtered),
                                                batch_size=batch_size,
                                                shuffle=True)
        
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader),position=0, leave=False):
            integrated_gradients = IntegratedGradients(model)
            attributions, _ = integrated_gradients.attribute(inputs, target=labels, return_convergence_delta=True)
            attributions = attributions.squeeze().cpu().numpy()
            total_attributions.append(np.abs(attributions))

            # if i == 100:
            #     break
        atotal_attributions_array = np.stack(total_attributions, axis=0)

        # print(attributions.shape)
        # print(len(total_attributions))
        mean_values = np.mean(atotal_attributions_array, axis=0)
        mean_values = mean_values.T
        # print(mean_values.shape)
        tmp_data = pd.DataFrame(mean_values)
        tmp_data.columns = columns_to_look
        tmp_data['number'] = label
        res_per_channel.append(tmp_data)

        mean_values_mean_values = np.mean(mean_values, axis=0)
        # print(mean_values_mean_values.shape)
        tmp_data_mean = pd.DataFrame([mean_values_mean_values])
        tmp_data_mean.columns = columns_to_look
        tmp_data_mean['number'] = label
        res_summarize_channel.append(tmp_data_mean)

        # plt.figure(figsize=(10, 5))
        # plt.imshow(mean_values.T, cmap='RdBu', aspect='auto')
        # plt.colorbar()
        # plt.title('Integrated Gradients {}'.format(label))
        # plt.xlabel('Time Step')
        # plt.ylabel('Feature Index')
        # plt.yticks(np.arange(len(columns_to_look)), columns_to_look)
        # plt.tight_layout()
        # plt.show()
        
    res_per_channel = pd.concat(res_per_channel)
    res_per_channel.to_csv(f"{name}_IG_res_per_channel.csv")

    res_summarize_channel = pd.concat(res_summarize_channel)
    res_summarize_channel.to_csv(f"{name}_IG_res_summarize_channel.csv")