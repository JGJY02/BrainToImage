from models.EEGViT_pretrained import EEGViT_pretrained
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from models.ViTBase_pretrained import ViTBase_pretrained
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset, singleClassDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import os
import pickle

'''
models: EEGViT_pretrained; EEGViT_raw; ViTBase; ViTBase_pretrained
'''
eeg_data = pickle.load(open("dataset/000thresh_AllStack_Transformer_dual_2.pkl", 'rb'), encoding='bytes')
model = EEGViT_pretrained(eeg_data['y_train'].shape[1])

save_path = "trained_models/Transformer_512_single"
os.makedirs(save_path, exist_ok=True)
batch_size = 64
n_epoch = 15
learning_rate = 1e-4

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

def accuracy(targets, output):
    targets_label = torch.argmax(targets, 1)
    predicted_label = torch.argmax(output, 1)

    correct = (targets_label == predicted_label).sum().item()
    
    return correct


def train(model, optimizer, scheduler = None):
    '''
        model: model to train
        optimizer: optimizer to update weights
        scheduler: scheduling learning rate, used when finetuning pretrained models
    '''
    torch.cuda.empty_cache()
    # indexes = range(len(EEGEyeNet))
    # train_indices, val_indices, test_indices = split(indexes,0.7,0.15,0.15)  # indices for the training set
    # print(train_indices.shape)
    print('create dataloader...')
    criterion = nn.CrossEntropyLoss()

    train =singleClassDataset(eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'])
    val = singleClassDataset(eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test'])
    test = singleClassDataset(eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test'])

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize lists to store losses
    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    test_losses = []
    test_accuracies = []
    print('training...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        epoch_val_acc = 0.0
        epoch_test_acc = 0.0
        correct = 0
        total = 0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader), desc="Training"):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # print(inputs.shape)
            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            
            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            epoch_train_acc += accuracy(targets, outputs)
            total += targets.size(0)

            # Print the loss and accuracy for the current batch

        epoch_train_loss /= len(train_loader)
        epoch_train_acc /= total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        print(f"Epoch {epoch}, Batch {i}, Loss: {epoch_train_loss}, acc: {epoch_train_acc:.2f}")

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total = 0
            for inputs, targets, index in val_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs, _ = model(inputs)
                # print(outputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()
                
                epoch_val_acc += accuracy(targets, outputs)
                total += targets.size(0)

            val_loss /= len(val_loader)
            epoch_val_acc /= total

            val_losses.append(val_loss)
            val_accuracies.append(epoch_val_acc)

            print(f"Epoch {epoch}, Val Loss: {val_loss}, Val acc: {epoch_val_acc:.2f}")

        with torch.no_grad():
            test_loss = 0.0
            total = 0

            for inputs, targets, index in test_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs, _ = model(inputs)

                loss = criterion(outputs.squeeze(), targets.squeeze())
                test_loss += loss.item()

                epoch_test_acc += accuracy(targets, outputs)
                total += targets.size(0)
                

            test_loss /= len(test_loader)
            epoch_test_acc /= total

            test_losses.append(test_loss)
            test_accuracies.append(epoch_test_acc)

            print(f"Epoch {epoch}, test Loss: {test_loss}, Test acc: {epoch_test_acc:.2f}")


        torch.save(model.state_dict(), f'{save_path}/eeg_classifier_adm5_{epoch+1}.pth')



        if scheduler is not None:
            scheduler.step()

    results_dict = {"train_loss": train_losses, "val_loss": val_losses, "test_loss": test_losses, "train_acc": train_accuracies, "val_acc": val_accuracies, "test_acc": test_accuracies}
    np.save(f'{save_path}/results.npy', results_dict)
    torch.save(model.state_dict(), f'{save_path}/eeg_classifier_adm5_final.pth')


if __name__ == "__main__":
    train(model,optimizer=optimizer, scheduler=scheduler)