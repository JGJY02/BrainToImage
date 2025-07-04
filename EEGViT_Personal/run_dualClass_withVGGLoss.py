from models.EEGViT_pretrained import EEGViT_pretrained, EEGViT_pretrained_512, EEGViT_pretrained_512_experimental
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset, MultiClassDataset, MultiClassImageDataset

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import os
import pickle

'''
models: EEGViT_pretrained; EEGViT_raw; ViTBase; ViTBase_pretrained
'''
class VGG19Latent512(nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19Latent512, self).__init__()

        # Load pretrained VGG19 and take layers up to relu5_4 (index 35)
        vgg = models.vgg19(pretrained=True).features
        for i, layer in enumerate(vgg):
            print(f"Layer {i}: {layer}")

        self.features_14_14 = nn.Sequential(*[vgg[i] for i in range(28)]) 
        self.features_7_7 = nn.Sequential(*[vgg[i] for i in range(28,37)])  # relu5_4 is at index 35
        self.features_14_14.to(device)
        self.features_14_14.eval()  # freeze for inference
        self.features_7_7.to(device)
        self.features_7_7.eval()  # freeze for inference       
        # Freeze parameters
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        """
        x: input image tensor of shape [B, 3, H, W], e.g., [B, 3, 224, 224]
        Returns:
            latent vector of shape [B, 512]
        """
        with torch.no_grad():
            feat_14_14 = self.features_14_14(x)
            feat_7_7 = self.features_7_7(feat_14_14)         # Output: [B, 512, 7, 7] for 224x224 input
        return feat_14_14, feat_7_7




eeg_data = pickle.load(open("dataset/000thresh_AllStack_large_Transformer_dual_2_64.pkl", 'rb'), encoding='bytes')
eeg_data_unseen = pickle.load(open("dataset/000thresh_AllStack_large_Transformer_dual_2_unseen.pkl", 'rb'), encoding='bytes')

model = EEGViT_pretrained_512_experimental(eeg_data['y_train'].shape[1], eeg_data['y_secondary_train'].shape[1])
# EEGEyeNet = MultiClassDataset('./dataset/000thresh_AllStack_Transformer_dual.pkl')
save_path = "trained_models/Transformer_512_dual_large"
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

    # train =MultiClassDataset(eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'])
    # val = MultiClassDataset(eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test'])
    # test = MultiClassDataset(eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test'])

    train =MultiClassImageDataset(eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_train_img'])
    val = MultiClassImageDataset(eeg_data_unseen['x_train_eeg'], eeg_data_unseen['y_train'], eeg_data_unseen['y_secondary_train'], eeg_data_unseen['x_train_img'])
    test = MultiClassImageDataset(eeg_data_unseen['x_test_eeg'], eeg_data_unseen['y_test'], eeg_data_unseen['y_secondary_test'], eeg_data_unseen['x_test_img'])

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
    
    vgg_model = VGG19Latent512(device=device)

    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize lists to store losses
    train_losses = []
    train_losses_1 = []
    train_losses_2 = []
    train_vgg_loss = []

    train_accuracies_1 = []
    train_accuracies_2 = []

    val_losses = []
    val_losses_1 = []
    val_losses_2 = []
    val_vgg_loss = []

    val_accuracies_1 = []
    val_accuracies_2 = []

    test_losses = []
    test_losses_1 = []
    test_losses_2 = []
    test_vgg_loss = []


    test_accuracies_1 = []
    test_accuracies_2 = []

    print('training...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss_total = 0.0
        epoch_train_loss_1 = 0.0
        epoch_train_loss_2 = 0.0
        epoch_train_vggloss = 0.0

        epoch_train_acc_1 = 0.0
        epoch_train_acc_2 = 0.0

        epoch_val_loss_total = 0.0
        epoch_val_loss_1 = 0.0
        epoch_val_loss_2 = 0.0
        epoch_val_vggloss = 0.0

        epoch_val_acc_1 = 0.0
        epoch_val_acc_2 = 0.0

        epoch_test_loss_total = 0.0
        epoch_test_loss_1 = 0.0
        epoch_test_loss_2 = 0.0
        epoch_test_vggloss = 0.0

        epoch_test_acc_1 = 0.0
        epoch_test_acc_2 = 0.0

        total = 0

        for i, (inputs, target1, target2, target_image,index) in tqdm(enumerate(train_loader), desc="Training"):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            target_image = target_image.to(device)

            # print(inputs.shape)
            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            output1, transformer_latent, output2, feature_map_14, feature_map_7 = model(inputs)
            vgg_14, vgg_7 = vgg_model(target_image)  # shape: [1, 512]

            # print(output1.shape)
            # print(target1.shape)
            # print(output2.shape)
            # print(target2.shape)
            vgg_loss_14 = torch.nn.functional.mse_loss(feature_map_14, vgg_14)  
            l1_loss_14 = torch.nn.functional.l1_loss(feature_map_14, vgg_14)  

            vgg_loss_7 = torch.nn.functional.mse_loss(feature_map_7, vgg_7)
            l1_loss_7 = torch.nn.functional.l1_loss(feature_map_14, vgg_14)  


            vgg_loss = vgg_loss_7 + vgg_loss_14 + l1_loss_7 + l1_loss_14

            loss1 = criterion(output1.squeeze(), target1.squeeze())
            loss2 = criterion(output2.squeeze(), target2.squeeze())

            if epoch <= 5:
                loss = vgg_loss
            else:
                loss = vgg_loss + loss1 + loss2

            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()

            epoch_train_loss_total += loss.item()
            epoch_train_loss_1 += loss1.item()
            epoch_train_loss_2 += loss2.item()
            epoch_train_vggloss += vgg_loss.item()


            epoch_train_acc_1 += accuracy(target1, output1)
            epoch_train_acc_2 += accuracy(target2, output2)

            total += target1.size(0)

            # Print the loss and accuracy for the current batch

        epoch_train_loss_total /= len(train_loader)
        epoch_train_loss_1 /= len(train_loader)
        epoch_train_loss_2 /= len(train_loader)
        epoch_train_vggloss /= len(train_loader)

        epoch_train_acc_1 /= total
        epoch_train_acc_2 /= total

        train_losses.append(epoch_train_loss_total)
        train_losses_1.append(epoch_train_loss_1)
        train_losses_2.append(epoch_train_loss_2)
        train_vgg_loss.append(epoch_train_vggloss)

        train_accuracies_1.append(epoch_train_acc_1)
        train_accuracies_2.append(epoch_train_acc_2)


        print(f"Epoch {epoch}, Train Results, Total Loss: {epoch_train_loss_total},Class Loss: {epoch_train_loss_1},Type Loss: {epoch_train_loss_2}, VGG Loss: {epoch_train_vggloss}  \
        Class acc: {epoch_train_acc_1:.2f}, Type acc: {epoch_train_acc_2:.2f}")

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total = 0
            for inputs, target1, target2, target_image, index in val_loader:
            # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device)
                target_image = target_image.to(device)

                # print(inputs.shape)
                # Compute the outputs and loss for the current batch
                optimizer.zero_grad()
                output1, transformer_latent, output2, feature_map_14,feature_map_7 = model(inputs)
                vgg_14, vgg_7 = vgg_model(target_image)  # shape: [1, 512]

                vgg_loss_14 = torch.nn.functional.mse_loss(feature_map_14, vgg_14)  
                l1_loss_14 = torch.nn.functional.l1_loss(feature_map_14, vgg_14)  

                vgg_loss_7 = torch.nn.functional.mse_loss(feature_map_7, vgg_7)
                l1_loss_7 = torch.nn.functional.l1_loss(feature_map_14, vgg_14)  


                vgg_loss = vgg_loss_7 + vgg_loss_14 + l1_loss_7 + l1_loss_14
                loss1 = criterion(output1.squeeze(), target1.squeeze())
                loss2 = criterion(output2.squeeze(), target2.squeeze())

                if epoch <= 5:
                    loss = vgg_loss
                else:
                    loss = vgg_loss + loss1 + loss2

                # Compute the gradients and update the parameters
                epoch_val_loss_total += loss.item()
                epoch_val_loss_1 += loss1.item()
                epoch_val_loss_2 += loss2.item()
                epoch_val_vggloss += vgg_loss.item()


                epoch_val_acc_1 += accuracy(target1, output1)
                epoch_val_acc_2 += accuracy(target2, output2)

                total += target1.size(0)

            # Print the loss and accuracy for the current batch

        epoch_val_loss_total /= len(val_loader)
        epoch_val_loss_1 /= len(val_loader)
        epoch_val_loss_2 /= len(val_loader)
        epoch_val_vggloss/= len(val_loader)

        epoch_val_acc_1 /= total
        epoch_val_acc_2 /= total

        val_losses.append(epoch_val_loss_total)
        val_losses_1.append(epoch_val_loss_1)
        val_losses_2.append(epoch_val_loss_2)
        val_vgg_loss.append(epoch_val_vggloss)

        val_accuracies_1.append(epoch_val_acc_1)
        val_accuracies_2.append(epoch_val_acc_2)


        print(f"Epoch {epoch}, Val Results, Total Loss: {epoch_val_loss_total},Class Loss: {epoch_val_loss_1},Type Loss: {epoch_val_loss_2}, VGG Loss: {epoch_val_vggloss}   \
        Class acc: {epoch_val_acc_1:.2f}, Type acc: {epoch_val_acc_2:.2f}")

        with torch.no_grad():
            test_loss = 0.0
            total = 0

            for inputs, target1, target2, target_image, index in test_loader:
            # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device)
                target_image = target_image.to(device)


                # print(inputs.shape)
                # Compute the outputs and loss for the current batch
                optimizer.zero_grad()
                output1, transformer_latent, output2, feature_map_14, feature_map_7 = model(inputs)
                vgg_14, vgg_7  = vgg_model(target_image)  # shape: [1, 512]

                vgg_loss_14 = torch.nn.functional.mse_loss(feature_map_14, vgg_14)  
                l1_loss_14 = torch.nn.functional.l1_loss(feature_map_14, vgg_14)  

                vgg_loss_7 = torch.nn.functional.mse_loss(feature_map_7, vgg_7)
                l1_loss_7 = torch.nn.functional.l1_loss(feature_map_14, vgg_14)  


                vgg_loss = vgg_loss_7 + vgg_loss_14 + l1_loss_14 + l1_loss_7
                loss1 = criterion(output1.squeeze(), target1.squeeze())
                loss2 = criterion(output2.squeeze(), target2.squeeze())

                if epoch <= 5:
                    loss = vgg_loss
                else:
                    loss = vgg_loss + loss1 + loss2

                # Compute the gradients and update the parameters
                epoch_test_loss_total += loss.item()
                epoch_test_loss_1 += loss1.item()
                epoch_test_loss_2 += loss2.item()
                epoch_test_vggloss += vgg_loss.item()


                epoch_test_acc_1 += accuracy(target1, output1)
                epoch_test_acc_2 += accuracy(target2, output2)

                total += target1.size(0)

            # Print the loss and accuracy for the current batch

        epoch_test_loss_total /= len(test_loader)
        epoch_test_loss_1 /= len(test_loader)
        epoch_test_loss_2 /= len(test_loader)
        epoch_test_vggloss /= len(test_loader)

        epoch_test_acc_1 /= total
        epoch_test_acc_2 /= total

        test_losses.append(epoch_test_loss_total)
        test_losses_1.append(epoch_test_loss_1)
        test_losses_2.append(epoch_test_loss_2)
        test_vgg_loss.append(epoch_test_vggloss)


        test_accuracies_1.append(epoch_test_acc_1)
        test_accuracies_2.append(epoch_test_acc_2)


        print(f"Epoch {epoch}, Test Results, Total Loss: {epoch_test_loss_total},Class Loss: {epoch_test_loss_1},Type Loss: {epoch_test_loss_2}, VGG Loss: {epoch_test_vggloss}  \
        Class acc: {epoch_test_acc_1:.2f}, Type acc: {epoch_test_acc_2:.2f}")

        torch.save(model.state_dict(), f'{save_path}/eeg_classifier_adm5_{epoch+1}.pth')



        if scheduler is not None:
            scheduler.step()

    results_dict = {"total_train_loss": train_losses, "total_val_loss": val_losses, "total_test_loss": test_losses,\
        "train_loss_class": train_losses_1, "val_loss_class": val_losses_1, "test_loss_class": test_losses_1, \
        "train_loss_type": train_losses_2, "val_loss_type": val_losses_2, "test_loss_type": test_losses_2,     \
        "train_vgg_loss": train_vgg_loss, "val_vgg_loss": val_vgg_loss, "test_vgg_loss": test_vgg_loss,     \
        "train_acc_class": test_accuracies_1, "val_acc_class": val_accuracies_1, "test_acc_class": test_accuracies_1, \
        "train_acc_type": test_accuracies_2, "val_acc_type": val_accuracies_2, "test_acc_type": test_accuracies_2 \
}
    np.save(f'{save_path}/results.npy', results_dict)
    torch.save(model.state_dict(), f'{save_path}/eeg_classifier_adm5_final.pth')


if __name__ == "__main__":
    train(model,optimizer=optimizer, scheduler=scheduler)