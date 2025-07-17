import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

from models.EEGViT_pretrained import EEGViT_pretrained, EEGViT_pretrained_512
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset, MultiClassDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from collections import Counter
import argparse
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score 

'''
models: EEGViT_pretrained; EEGViT_raw; ViTBase; ViTBase_pretrained
'''
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--num_of_folds', type=int, help="Number of folds", default = 5 , required=False)
parser.add_argument('--dataset_pickle', type=str, help="dataset used", default = "000thresh_AllStack_Transformer_dual_2_64.pkl" , required=False)

args = parser.parse_args()

dataset_used = f"dataset/{args.dataset_pickle}"
eeg_data = pickle.load(open(dataset_used, 'rb'), encoding='bytes')
save_path = f"trained_models/crossVal/Transformer_512_dual"

model = EEGViT_pretrained_512(eeg_data['y_train'].shape[1], eeg_data['y_secondary_train'].shape[1])

# EEGEyeNet = MultiClassDataset('./dataset/000thresh_AllStack_Transformer_dual.pkl')

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

    label_dictionary = eeg_data['dictionary']
    x_train, y_train, y_secondary_train, x_test, y_test, y_secondary_test = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']
    class_primary_labels = eeg_data['y_train'].shape[1]

    X = np.vstack((x_train, x_test))
    Y_primary = np.vstack((y_train, y_test))
    Y_secondary = np.vstack((y_secondary_train, y_secondary_test))
    
    Y = [f"{a}-{b}" for a, b in zip(Y_primary, Y_secondary)]  # for stratufucatuib if unique combinations

    # X = X[:100]
    # Y = Y[:100]

    skf = StratifiedKFold(n_splits = args.num_of_folds)
    previous_results = []

    saved_indexes = {}
    for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
        model_save_dir = os.path.join(save_path, f"fold{fold}")

        os.makedirs(model_save_dir, exist_ok=True)
        print('create dataloader...')
        criterion = nn.CrossEntropyLoss()

        x_train = X[train_index]
        y_train = Y_primary[train_index]
        y_secondary_train = Y_secondary[train_index]
        
        x_test = X[test_index]
        y_test = Y_primary[test_index]
        y_secondary_test = Y_secondary[test_index]

        train = MultiClassDataset(x_train, y_train, y_secondary_train)
        val = MultiClassDataset(x_test, y_test, y_secondary_test)
        test = MultiClassDataset(x_test, y_test, y_secondary_test)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
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

        model = model.to(device)
        criterion = criterion.to(device)

        # Initialize lists to store losses
        train_losses = []
        train_losses_1 = []
        train_losses_2 = []

        train_accuracies_1 = []
        train_accuracies_2 = []

        val_losses = []
        val_losses_1 = []
        val_losses_2 = []

        val_accuracies_1 = []
        val_accuracies_2 = []

        test_losses = []
        test_losses_1 = []
        test_losses_2 = []

        test_accuracies_1 = []
        test_accuracies_2 = []

        print('training...')
        # Train the model
        for epoch in range(n_epoch):
            model.train()
            epoch_train_loss_total = 0.0
            epoch_train_loss_1 = 0.0
            epoch_train_loss_2 = 0.0

            epoch_train_acc_1 = 0.0
            epoch_train_acc_2 = 0.0

            epoch_val_loss_total = 0.0
            epoch_val_loss_1 = 0.0
            epoch_val_loss_2 = 0.0

            epoch_val_acc_1 = 0.0
            epoch_val_acc_2 = 0.0

            epoch_test_loss_total = 0.0
            epoch_test_loss_1 = 0.0
            epoch_test_loss_2 = 0.0

            epoch_test_acc_1 = 0.0
            epoch_test_acc_2 = 0.0

            total = 0

            for i, (inputs, target1, target2, index) in tqdm(enumerate(train_loader), desc="Training"):
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device)

                # print(inputs.shape)
                # Compute the outputs and loss for the current batch
                optimizer.zero_grad()
                output1, _, output2 = model(inputs)

                # print(output1.shape)
                # print(target1.shape)
                # print(output2.shape)
                # print(target2.shape)

                loss1 = criterion(output1.squeeze(), target1.squeeze())
                loss2 = criterion(output2.squeeze(), target2.squeeze())

                
                loss = loss1 + loss2

                # Compute the gradients and update the parameters
                loss.backward()
                optimizer.step()
                epoch_train_loss_total += loss.item()
                epoch_train_loss_1 += loss1.item()
                epoch_train_loss_2 += loss2.item()


                epoch_train_acc_1 += accuracy(target1, output1)
                epoch_train_acc_2 += accuracy(target2, output2)

                total += target1.size(0)

                # Print the loss and accuracy for the current batch

            epoch_train_loss_total /= len(train_loader)
            epoch_train_loss_1 /= len(train_loader)
            epoch_train_loss_2 /= len(train_loader)

            epoch_train_acc_1 /= total
            epoch_train_acc_2 /= total

            train_losses.append(epoch_train_loss_total)
            train_losses_1.append(epoch_train_loss_1)
            train_losses_2.append(epoch_train_loss_2)

            train_accuracies_1.append(epoch_train_acc_1)
            train_accuracies_2.append(epoch_train_acc_2)


            print(f"Epoch {epoch}, Train Results, Total Loss: {epoch_train_loss_total},Class Loss: {epoch_train_loss_1},Type Loss: {epoch_train_loss_2}  \
            Class acc: {epoch_train_acc_1:.2f}, Type acc: {epoch_train_acc_2:.2f}")

            # Evaluate the model on the validation set
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                total = 0
                for inputs, target1, target2, index in val_loader:
                # Move the inputs and targets to the GPU (if available)
                    inputs = inputs.to(device)
                    target1 = target1.to(device)
                    target2 = target2.to(device)

                    # print(inputs.shape)
                    # Compute the outputs and loss for the current batch
                    optimizer.zero_grad()
                    output1, _, output2 = model(inputs)
                    loss1 = criterion(output1.squeeze(), target1.squeeze())
                    loss2 = criterion(output2.squeeze(), target2.squeeze())

                    loss = loss1 + loss2

                    # Compute the gradients and update the parameters
                    epoch_val_loss_total += loss.item()
                    epoch_val_loss_1 += loss1.item()
                    epoch_val_loss_2 += loss2.item()


                    epoch_val_acc_1 += accuracy(target1, output1)
                    epoch_val_acc_2 += accuracy(target2, output2)

                    total += target1.size(0)

                # Print the loss and accuracy for the current batch

            epoch_val_loss_total /= len(val_loader)
            epoch_val_loss_1 /= len(val_loader)
            epoch_val_loss_2 /= len(val_loader)

            epoch_val_acc_1 /= total
            epoch_val_acc_2 /= total

            val_losses.append(epoch_val_loss_total)
            val_losses_1.append(epoch_val_loss_1)
            val_losses_2.append(epoch_val_loss_2)

            val_accuracies_1.append(epoch_val_acc_1)
            val_accuracies_2.append(epoch_val_acc_2)


            print(f"Epoch {epoch}, Val Results, Total Loss: {epoch_val_loss_total},Class Loss: {epoch_val_loss_1},Type Loss: {epoch_val_loss_2}  \
            Class acc: {epoch_val_acc_1:.2f}, Type acc: {epoch_val_acc_2:.2f}")

            with torch.no_grad():
                test_loss = 0.0
                total = 0

                for inputs, target1, target2, index in test_loader:
                # Move the inputs and targets to the GPU (if available)
                    inputs = inputs.to(device)
                    target1 = target1.to(device)
                    target2 = target2.to(device)

                    # print(inputs.shape)
                    # Compute the outputs and loss for the current batch
                    optimizer.zero_grad()
                    output1, _, output2 = model(inputs)
                    loss1 = criterion(output1.squeeze(), target1.squeeze())
                    loss2 = criterion(output2.squeeze(), target2.squeeze())

                    loss = loss1 + loss2

                    # Compute the gradients and update the parameters
                    epoch_test_loss_total += loss.item()
                    epoch_test_loss_1 += loss1.item()
                    epoch_test_loss_2 += loss2.item()


                    epoch_test_acc_1 += accuracy(target1, output1)
                    epoch_test_acc_2 += accuracy(target2, output2)

                    total += target1.size(0)

                # Print the loss and accuracy for the current batch

            epoch_test_loss_total /= len(test_loader)
            epoch_test_loss_1 /= len(test_loader)
            epoch_test_loss_2 /= len(test_loader)

            epoch_test_acc_1 /= total
            epoch_test_acc_2 /= total

            test_losses.append(epoch_test_loss_total)
            test_losses_1.append(epoch_test_loss_1)
            test_losses_2.append(epoch_test_loss_2)

            test_accuracies_1.append(epoch_test_acc_1)
            test_accuracies_2.append(epoch_test_acc_2)


            print(f"Epoch {epoch}, Test Results, Total Loss: {epoch_test_loss_total},Class Loss: {epoch_test_loss_1},Type Loss: {epoch_test_loss_2}  \
            Class acc: {epoch_test_acc_1:.2f}, Type acc: {epoch_test_acc_2:.2f}")

            # torch.save(model.state_dict(), f'{save_path}/eeg_classifier_adm5_{epoch+1}.pth')



            if scheduler is not None:
                scheduler.step()


        results_dict = {"total_train_loss": train_losses, "total_val_loss": val_losses, "total_test_loss": test_losses, \
            "train_loss_class": train_losses_1, "val_loss_class": val_losses_1, "test_loss_class": test_losses_1, \
            "train_loss_type": train_losses_2, "val_loss_type": val_losses_2, "test_loss_type": test_losses_2,     \
            "train_acc_class": test_accuracies_1, "val_acc_class": val_accuracies_1, "test_acc_class": test_accuracies_1, \
            "train_acc_type": test_accuracies_2, "val_acc_type": val_accuracies_2, "test_acc_type": test_accuracies_2 \
    }
        np.save(f'{model_save_dir}/results.npy', results_dict)
        torch.save(model.state_dict(), f'{model_save_dir}/eeg_classifier_adm5_final.pth')
        print(f"Fold {fold} Complete!")

    ## Results for Fold


        # encoded_latents = []
        encoded_labels = []
        encoded_type_labels = []

        with torch.no_grad():
            val_loss = 0.0
            total = 0
            for inputs, target1, target2, index in val_loader:
            # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device)

                # print(inputs.shape)
                # Compute the outputs and loss for the current batch
                optimizer.zero_grad()
                output1, _, output2 = model(inputs)

                encoded_labels.append(output1.cpu())
                encoded_type_labels.append(output2.cpu())
        
        # encoded_latents = np.concatenate(encoded_latents, axis=0)
        encoded_labels = np.concatenate(encoded_labels, axis=0)
        encoded_type_labels = np.concatenate(encoded_type_labels, axis =0)
        to_labels = np.argmax(y_test,axis=1)  ## since eeg labels are in one-hot encoded format

        #predict for generated labels
        pred_primary_label_array = np.argmax(encoded_labels, axis = 1)
        true_primary_label_array = np.argmax(y_test, axis = 1)

        #For label type
        pred_secondary_label_array = np.argmax(encoded_type_labels, axis = 1)
        true_secondary_label_array = np.argmax(y_secondary_test, axis = 1)

        evaluation ={}
        text_to_save = []
        mean_accuracy_scores = []
        mean_accuracy_type_scores = []

        mean_precision_scores = []
        mean_recall_scores = []
        mean_F1_scores =[]

        mean_type_precision_scores = []
        mean_type_recall_scores = []
        mean_type_F1_scores =[]

        for lab in range(class_primary_labels):


            print("Current class label is : ", lab)
            matching_indices = np.where(to_labels == lab)

            conditioning_labels_array = pred_primary_label_array[matching_indices]
            conditioning_labels_type_array = pred_secondary_label_array[matching_indices]
            true_labels_array = true_primary_label_array[matching_indices]
            true_labels_type_array = true_secondary_label_array[matching_indices]


            true_positives = np.sum(true_labels_array == conditioning_labels_array)
            true_positives_type  = np.sum(true_labels_type_array == conditioning_labels_type_array) 


            #F1
            F1_value = f1_score(true_labels_array, conditioning_labels_array, average='macro')
            F1_type_value = f1_score(true_labels_type_array, conditioning_labels_type_array, average='macro')

            #Recall
            recall_value = recall_score(true_labels_array, conditioning_labels_array, average='macro')
            recall_type_value = recall_score(true_labels_type_array, conditioning_labels_type_array, average='macro')

            #Precision
            precision_value = precision_score(true_labels_array,conditioning_labels_array, average='macro')
            precision_type_value = precision_score(true_labels_type_array,conditioning_labels_type_array, average='macro')

            class_acc = true_positives / conditioning_labels_array.shape[0]
            class_type_acc = true_positives_type / conditioning_labels_type_array.shape[0]

            evaluation[lab] = {
                'primary_class_acc': class_acc, 'secondary_class_acc':class_type_acc,\
                'average_F1':F1_value,'average_recall':recall_value,'average_precision':precision_value,\
                'average_type_F1':F1_type_value,'average_type_recall':recall_type_value,'average_type_precision':precision_type_value}

            text_to_print = f"Class {lab} ({label_dictionary[lab]}): classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
                            mean F1 {evaluation[lab]['average_F1']:.2f}, mean recall {evaluation[lab]['average_recall']:.2f}, mean precision {evaluation[lab]['average_precision']:.2f} , \n \
                            mean type F1 {evaluation[lab]['average_type_F1']:.2f}, mean type recall {evaluation[lab]['average_type_recall']:.2f}, mean type precision {evaluation[lab]['average_type_precision']:.2f}"
            text_to_save.append(text_to_print)
            print(text_to_print)



            mean_accuracy_scores.append(class_acc)
            mean_accuracy_type_scores.append(class_type_acc)

            mean_F1_scores.append(evaluation[lab]['average_F1'])
            mean_recall_scores.append(evaluation[lab]['average_recall'])
            mean_precision_scores.append(evaluation[lab]['average_precision'])

            mean_type_F1_scores.append(evaluation[lab]['average_type_F1'])
            mean_type_recall_scores.append(evaluation[lab]['average_type_recall'])
            mean_type_precision_scores.append(evaluation[lab]['average_type_precision'])

        mean_evaluation = {'average_accuracy':np.mean(mean_accuracy_scores), 'average_type_accuracy':np.mean(mean_accuracy_type_scores), \
        'average_f1' : np.mean(mean_F1_scores), 'average_recall' : np.mean(mean_recall_scores), 'average_precision' : np.mean(mean_precision_scores), \
        'average_type_f1' : np.mean(mean_type_F1_scores), 'average_type_recall' : np.mean(mean_type_recall_scores), 'average_type_precision' : np.mean(mean_type_precision_scores), 
            }

        mean_text_to_print = f"Average Class Results: mean classification acc: {mean_evaluation['average_accuracy']:.1%} ,mean type classification acc: {mean_evaluation['average_type_accuracy']:.1%} \n \
        mean F1: {mean_evaluation['average_f1']:.2f}, mean recall: {mean_evaluation['average_recall']:.2f}, mean precision: {mean_evaluation['average_precision']:.2f} \n \
        mean type F1: {mean_evaluation['average_type_f1']:.2f}, mean type recall: {mean_evaluation['average_type_recall']:.2f}, mean type precision: {mean_evaluation['average_type_precision']:.2f} \n"
        
        print(mean_text_to_print)
        text_to_save.append(mean_text_to_print)

        ## Save fold results
        with open(f"{model_save_dir}/results.txt", "w") as file:
            file.write("\n".join(text_to_save) + "\n")
        
        with open(f"{model_save_dir}/fold_indices.txt", "w") as file:
            file.write(f"Dataset used: {dataset_used}\n\n")
            file.write(f"Fold {fold}\n\n")
            file.write(f"Train indices: {train_index.tolist()}\n\n")
            file.write(f"Val indices:   {test_index.tolist()}\n\n")
            
        saved_indexes[fold] = {"train_idx": train_index, 'val_idx': test_index}

        previous_results.append([evaluation, mean_evaluation])

    ## Summarise folds
    mean_per_class = np.zeros((10,8))
    mean_fold = np.zeros((1,8))
    for fold in range(args.num_of_folds):
        cur_perClass_dict, cur_mean_dict = previous_results[fold]
        
        rows = []
        for lab in range(class_primary_labels):
            vals = np.array(list(cur_perClass_dict[lab].values()))
            rows.append(vals)
        
        result = np.vstack(rows)

        mean_result = np.array(list(cur_mean_dict.values()))
        
        # print(result)
        # print(result.shape)
        mean_per_class = result + mean_per_class
        mean_fold = mean_result + mean_fold
        # print(mean_fold)
        # print(mean_fold.shape)

    mean_per_class /= args.num_of_folds
    mean_fold /= args.num_of_folds

    text_to_save = []

    for lab in range(class_primary_labels):
        class_acc, class_type_acc, F1, recall, precision, F1_type, recall_type, precision_type = mean_per_class[lab, :]
        
        text_to_print = f"Class {lab} ({label_dictionary[lab]}): classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
            mean F1 {F1:.2f}, mean recall {recall:.2f}, mean precision {precision:.2f} , \n \
            mean type F1 {F1_type:.2f}, mean type recall {recall_type:.2f}, mean type precision {precision_type:.2f}"
        
        text_to_save.append(text_to_print)
        print(text_to_print)

    class_acc, class_type_acc, F1, recall, precision, F1_type, recall_type, precision_type = mean_fold[0,:]
        
    mean_text_to_print = f"Average Class Results: classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
        mean F1 {F1:.2f}, mean recall {recall:.2f}, mean precision {precision:.2f} , \n \
        mean type F1 {F1_type:.2f}, mean type recall {recall_type:.2f}, mean type precision {precision_type:.2f}"

    text_to_save.append(mean_text_to_print)
    print(mean_text_to_print)  

    ## Save fold results
    with open(f"{save_path}/results.txt", "w") as file:
        file.write("\n".join(text_to_save) + "\n")

    saved_indexes["dataset_pickle"] = args.dataset_pickle
    np.save(f"{save_path}/saved_indexes.npy", saved_indexes)

    # print(saved_indexes["dataset_pickle"])
if __name__ == "__main__":
    train(model,optimizer=optimizer, scheduler=scheduler)