import os
import pickle
import random
import sys

import numpy as np
from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense,
                          DepthwiseConv2D, Dropout, Embedding, Flatten,
                          GlobalAveragePooling2D, Input, LeakyReLU,
                          MaxPooling1D, MaxPooling2D, Reshape, UpSampling2D,
                          ZeroPadding2D, multiply)
from keras.models import Model, Sequential
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.eegclassifier import convolutional_encoder_model, LSTM_Classifier
from models.eeggan import (build_discriminator, build_EEGgan, build_generator,
                           build_MoGCgenerator, build_MoGMgenerator,
                           combine_loss_metrics, sample_images, save_model)


import config
import tfutil
import dataset
import misc
import tensorflow as tf

import torch
from models.EEGViT_pretrained_dual import (EEGViT_pretrained_dual)
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.data import Dataset as TorchDatasetClass
from tqdm import tqdm

## new metrics
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score 
from skimage.metrics import peak_signal_noise_ratio as psnr

from image_similarity_measures.quality_metrics import fsim

import tensorflow as tf
import cv2
import torch
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms

import pandas as pd


tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
"""
Title: EEG ACGAN Evaluations of models

Purpose:
    Testing design and build of EEG ACGAN and classifier, Functional blocks for training
    ACGAN model. Call from training script.

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    Run the script as is, uses the online object dataset to train the GAN

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""

def convert_tf_to_torch(np_images):
    """
    Converts TensorFlow images to PyTorch tensors.
    
    Args:
    - tf_images (Tensor): TensorFlow images (N, H, W, C) in range [0,1] or [-1,1].
    
    Returns:
    - PyTorch tensor (N, 3, H, W) normalized to [0,1].
    """
    # Convert TensorFlow tensor to NumPy
    np_images = np.repeat(np_images, 3, axis=-1)  # Convert grayscale to RGB if needed
    np_images = np.transpose(np_images, (0, 3, 1, 2))  # Change shape to (N, C, H, W)
    
    # Convert NumPy to PyTorch tensor
    torch_images = torch.tensor(np_images, dtype=torch.uint8)
    return torch_images

os.environ.update(config.env)
tfutil.init_tf(config.tf_config)
#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition
print(os.getcwd())
eeg_latent_dim = 128
class_labels = [0,1,2,3,4,5,6,7,8,9]
type_labels = [0,1]
valid_threshold = 0.5



## load the eeg training data
eeg_dataset = f"{config.eeg_dataset_dir}/{config.eeg_dataset_pickle}"
print(f"Reading data file {eeg_dataset}")
eeg_data = pickle.load(open(eeg_dataset, 'rb'), encoding='bytes')
label_dictionary = eeg_data['dictionary']

if config.evaluation_type == "standard":

    signals = eeg_data['x_test_eeg']
    y_primary_test = eeg_data['y_test']
    y_secondary_test = eeg_data['y_secondary_test']
    x_img_test = eeg_data['x_test_img']
    to_labels = np.argmax(eeg_data['y_test'],axis=1)  ## since eeg labels are in one-hot encoded format


elif config.evaluation_type == "crossVal":
    x_train_eeg_data, y_primary_train_data, y_secondary_train_data, x_test_eeg_data, y_primary_test_data, y_secondary_test_data = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']
    train_imgs, test_imgs = eeg_data['x_train_img'] , eeg_data['x_test_img']

    # x_train_img = (np.array(train_imgs) - 127.5) / 127.5
    # x_test_img = (np.array(test_imgs) - 127.5) / 127.5

    X_img = np.vstack((train_imgs, test_imgs))
    X_eeg = np.vstack((x_train_eeg_data, x_test_eeg_data))
    Y_primary = np.vstack((y_primary_train_data, y_primary_test_data))
    Y_secondary = np.vstack((y_secondary_train_data, y_secondary_test_data))

    fold_indexes = os.path.join(config.eeg_dataset_idx_dir, config.model_eval,config.fold_indices)
    index_dictionary = np.load(fold_indexes,allow_pickle = True).item()
    train_index = index_dictionary[config.fold]["train_idx"]
    test_index = index_dictionary[config.fold]["val_idx"]

    signals = X_eeg[test_index]
    y_primary_test = Y_primary[test_index]
    y_secondary_test = Y_secondary[test_index]
    x_img_test = X_img[test_index]

    to_labels = np.argmax(y_primary_test,axis=1)  ## since eeg labels are in one-hot encoded format

    
for i in class_labels:
    total = np.sum(to_labels == i)
    print(f"For Class {i} we have {total} Instances")

#Mini batch size

# ## #############
# # Build EEG Gan
# ## #############
resume_run_id           = config.resume_run_id
resume_snapshot         = config.resume_snapshot


## #############
# EEG Classifier/Encoder
## #############

if config.TfOrTorch == "TF":
    classifier_model_path = f"{config.eval_classifier_dir}/eeg_classifier_adm5_final.h5"
    classifier = LSTM_Classifier(signals.shape[1], signals.shape[2], len(class_labels), 128)
    classifier.load_weights(classifier_model_path)
    layer_names = ['EEG_feature_BN2','EEG_Class_Labels']
    encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
    encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

    dataset = tf.data.Dataset.from_tensor_slices(signals)
    batch_size = 64
    dataset = dataset.batch(batch_size)

    encoded_latents = []
    encoded_labels = []

    for batch in tqdm(dataset):
        encodedEEG, encodedLabel = encoder_model(batch, training=False)
        encoded_latents.append(encodedEEG)
        encoded_labels.append(encodedLabel)

    # Combine results
    encoded_latents = np.concatenate(encoded_latents, axis=0)
    encoded_labels = np.concatenate(encoded_labels, axis=0)


elif config.TfOrTorch == "Torch":
    classifier_model_path = f"{config.eval_classifier_dir}/eeg_classifier_adm5_final.pth"

    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    encoder_model = EEGViT_pretrained_dual(len(class_labels), len(type_labels))
    encoder_model.load_state_dict(torch.load(classifier_model_path, map_location=device))
    encoder_model = encoder_model.to(device)
    encoder_model.eval() 

    # signals = np.transpose(signals, (0,2,1))[:,np.newaxis,:,:]
    signals = signals[:,np.newaxis,:,:]
    print(signals.shape)
    tensor_eeg  = torch.from_numpy(signals).to(device)
    batch_size = 64  # You can tune this depending on your hardware
    dataset = TensorDataset(tensor_eeg)
    loader = DataLoader(dataset, batch_size=batch_size)

    encoded_latents = []
    encoded_labels = []
    encoded_type_labels = []

    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch[0].to(device)  # move to GPU if needed
            encodedLabel, encodedEEG, encodedTypeLabel = encoder_model(inputs)
            encoded_latents.append(encodedEEG.cpu())  
            encoded_labels.append(encodedLabel.cpu())  
            encoded_type_labels.append(encodedTypeLabel.cpu())

    encoded_latents = torch.cat(encoded_latents, dim=0)
    encoded_labels = torch.cat(encoded_labels, dim=0)
    encoded_type_labels = torch.cat(encoded_type_labels, dim=0)


    encoded_latents = encoded_latents.numpy()
    encoded_labels = encoded_labels.numpy()
    encoded_type_labels = encoded_type_labels.numpy()

else:
    raise FileNotFoundError(f"{config.TfOrTorch} is not a valid implementation")
  

## #############################################################
# Make prediction on random selected eeg from eeg_data['x_test']
## #############################################################
history = {}
for i in class_labels:  ## outer loop per class
    print("Current class label is : ", i)
    ## get all EEG data for class i
    matching_indices = np.where(to_labels == i)
    true_images = x_img_test[matching_indices[0]]
    labels = y_primary_test[matching_indices[0]]

    encoded_eegs = encoded_latents[matching_indices[0]]
    conditioning_labels_raw = encoded_labels[matching_indices[0]]
    
    conditioning_labels_type = encoded_type_labels[matching_indices[0]]
    true_conditioning_labels_type = y_secondary_test[matching_indices[0]]

    with tf.Graph().as_default(), tfutil.create_session(config.tf_config).as_default():
        with tf.compat.v1.device('/gpu:0'):
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            generator, discriminator, Gs = misc.load_pkl(network_pkl)

        
        # minibatch_size = np.clip(8192 // encoded_eegs.shape[1], 4, 256)
        minibatch_size = 4
        # print("The encoded eegs shape is :", encoded_eegs.shape)
        # print("The calculated minibatch size is :", minibatch_size)
        generated_samples = Gs.run(encoded_eegs, conditioning_labels_raw, conditioning_labels_type, minibatch_size = minibatch_size)

        

        validitys, labels_pred, validitys_type, labels_type_pred  = discriminator.run(generated_samples, minibatch_size = minibatch_size)

            ## predict on GAN
        # true_images = np.pad(true_images, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0) #pad images as progressiveGAN did so as well
        true_images_test = np.transpose(true_images, (0, 3, 1, 2)) 
        true_images_test = (true_images_test/127.5) - 1

        validitys_true, labels_true_pred, validitys_type_true, labels_type_true_pred  = discriminator.run(true_images_test, minibatch_size = minibatch_size)
        generated_samples = np.transpose(generated_samples, (0, 2, 3, 1))

        generated_samples = generated_samples*127.5 + 127.5
        generated_samples = np.clip(generated_samples,0,255)
        generated_samples = generated_samples.astype(np.uint8)

        # print("labels predicted : ", labels_pred)
        # print("labels conditioning : ", conditioning_labels_raw)
        # print("True label : ", labels_true_pred)



    # print("Shape of object images : ", true_images.shape)
    # print("Shape of generated images : ",generated_samples.shape)

    # print(type(true_images))
    # print(type(generated_samples))

    # print(generated_samples.shape)
    # print(object_images.shape)


    ## collate results
    history[i] = {'generated':generated_samples,'true':true_images,'valid':validitys,'predicted':labels_pred,'conditioning':conditioning_labels_raw, 'true_labels':labels, \
        'predicted_type': labels_type_pred,'conditioning_type': conditioning_labels_type, 'true_type': true_conditioning_labels_type, \
        'labels_true_pred' : labels_true_pred, 'labels_type_true_pred': labels_type_true_pred}




def save_imgs(images, name, class_label, conditioning_labels, conditioning_type,  predicted_labels, pred_type, real_label, real_type, output_dir, label_dictionary):
    # Set up the grid dimensions (10x10)
    rows = 10
    cols = 10

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Loop through the axes and images to display them

    for k, ax in enumerate(axes.flat):
        if k < len(images):
            ax.imshow(images[k])
            #C conditioning
            #P predicted
            #R True
            ax.set_title(f"C: {int(conditioning_labels[k])}, C_type: {int(conditioning_type[k])},\n P: {int(predicted_labels[k])},  P_type: {int(pred_type[k])},\n R: {int(real_label[k])}, R_type: {int(real_type[k])}", fontsize=8)
            
            ax.axis('off')  # Hide the axes
        else:
            ax.axis('off')  # In case there are empty subplots

        
    # Save the grid of images to a file
    output_path = f'{output_dir}/{name}_class{class_label}_{label_dictionary[class_label]}.png'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_imgs_comparison(images, output_dir):
    # Set up the grid dimensions (10x10)
    rows = 10
    cols = 4

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(5, 10))

    # Loop through the axes and images to display them

    for i in range(rows):
        axes[i,0].imshow(images[i][0][0])
        axes[i,1].imshow(images[i][1][0])
        axes[i,2].imshow(images[i][0][1])
        axes[i,3].imshow(images[i][1][1])

        axes[i,0].axis('off')  # Hide the axes
        axes[i,1].axis('off')  # Hide the axes
        axes[i,2].axis('off')  # Hide the axes                
        axes[i,3].axis('off')  # Hide the axes
        
    # Save the grid of images to a file
    output_path = f'{output_dir}/ImageComparison.png'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def binarise(image, threshold=0):
    # Ensure the input is a numpy array
    image = np.array(image)
    # Reshape to (28, 28) if it's (28, 28, 1)
    if image.shape == (28, 28, 1):
        image = image.reshape(28, 28)
    # Rescale from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Convert to binary using the threshold
    binary_image = (image > threshold).astype(np.uint8)

    return binary_image

def compute_uiq(true, pred):
    true = true.astype(np.float32)
    pred = pred.astype(np.float32)

    values = []

    for i in range(true.shape[2]):

        true_cur = true[:,:,i]
        pred_cur = pred[:,:,i]
        # Mean values
        mu_x = np.mean(true_cur)
        mu_y = np.mean(pred_cur)

        # Variance and covariance
        sigma_x2 = np.var(true_cur)
        sigma_y2 = np.var(pred_cur)
        sigma_xy = np.cov(true_cur.flatten(), pred_cur.flatten())[0, 1]

        # Compute UIQ
        numerator = 4 * sigma_xy * mu_x * mu_y
        denominator = (sigma_x2 + sigma_y2) * (mu_x**2 + mu_y**2)

        UIQ_value = numerator / denominator if denominator != 0 else 1.0  # Avoid division by zero

        values.append(UIQ_value)

    return np.mean(values)


def compute_fsim(true, pred):
    true = true.astype(np.float32)
    pred = pred.astype(np.float32)

    values = []

    for i in range(true.shape[2]):
        true_cur = true[:,:,i]
        pred_cur = pred[:,:,i]
        fsim_value = fsim(true_cur,pred_cur)

        values.append(fsim_value)

    return np.mean(values)


def compute_ssim(true, pred):


    values = []

    for i in range(true.shape[2]):

        true_cur = true[:,:,i]
        pred_cur = pred[:,:,i]
        data_range = pred_cur.max() - pred_cur.min()

        ssim_value = ssim(true_cur,pred_cur, data_range = data_range)

        values.append(ssim_value)

    return np.mean(values)

def save_imgs_horizontal_vertical(images, output_dir):
    # Set up the grid dimensions (10x10)
    rows = 10
    cols = 2

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(1.28, 6.4))

    # Loop through the axes and images to display them

    for i in range(rows):
        axes[i,0].imshow(images[i][0][1])
        axes[i,1].imshow(images[i][1][1])

        axes[i,0].axis('off')  # Hide the axes
        axes[i,1].axis('off')  # Hide the axes
    # Save the grid of images to a file
    output_path = f'{output_dir}/ImageVertical.png'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

        # Set up the grid dimensions (10x10)
    rows = 2
    cols = 10

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(6.4, 1.28))

    # Loop through the axes and images to display them

    for i in range(cols):
        axes[0,i].imshow(images[i][0][1])
        axes[1,i].imshow(images[i][1][1])

        axes[0,i].axis('off')  # Hide the axes
        axes[1,i].axis('off')  # Hide the axes
    # Save the grid of images to a file
    output_path = f'{output_dir}/ImageHorizontal.png'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

        # Set up the grid dimensions (10x10)
    rows = 10
    cols = 2

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(1.28, 6.4))

    # Loop through the axes and images to display them

    for i in range(rows):
        axes[i,0].imshow(images[i][0][0])
        axes[i,1].imshow(images[i][1][0])

        axes[i,0].axis('off')  # Hide the axes
        axes[i,1].axis('off')  # Hide the axes
    # Save the grid of images to a file
    output_path = f'{output_dir}/ImageVertical_ori.png'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

        # Set up the grid dimensions (10x10)
    rows = 2
    cols = 10

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(6.4, 1.28))

    # Loop through the axes and images to display them

    for i in range(cols):
        axes[0,i].imshow(images[i][0][0])
        axes[1,i].imshow(images[i][1][0])

        axes[0,i].axis('off')  # Hide the axes
        axes[1,i].axis('off')  # Hide the axes
    # Save the grid of images to a file
    output_path = f'{output_dir}/ImageHorizontal_ori.png'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

inception = InceptionScore().to(device)  # normalize=True = expects [0, 1] images

comparison_imgs = []

all_generated_images = []
all_true_images = []

mean_ssim_scores = []
mean_rmse_scores = []
mean_psnr_scores =[]
mean_fsim_scores = []
mean_uiq_scores = []
mean_inception_scores = []
mean_inception_stds = []

mean_real_inception_scores = []
mean_real_inception_stds = []

mean_accuracy_scores = []
mean_accuracy_type_scores = []

mean_precision_scores = []
mean_recall_scores = []
mean_F1_scores =[]

mean_type_precision_scores = []
mean_type_recall_scores = []
mean_type_F1_scores =[]

num_of_sample_images = 10

sample_img_per_class = []
list_of_labels = []
text_to_save = []

evaluation ={}
output_dir = config.evalOutputDir

# Transform image for inception
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),        # InceptionV3 expects 299x299
    transforms.PILToTensor(),                # Converts to [0, 1] float tensor and rearranges to (C, H, W)
])




for i in class_labels:
    class_data = history[i]

    classes_added = []
    types_added = []


    ssim_scores = []
    rmse_scores = []
    psnr_scores =[]
    fsim_scores = []
    uiq_scores = []
    inception_scores = []

    precision_scores = []
    recall_scores = []
    F1_scores = []

    precision_type_scores = []
    recall_type_scores = []
    F1_type_scores = []

    true_positives = 0
    true_positives_type = 0

    #predict for generated labels
    conditioning_labels_array = np.argmax(class_data['conditioning'], axis = 1)
    predicted_labels_array = np.argmax(class_data['predicted'], axis = 1)
    true_labels_array = np.argmax(class_data['true_labels'], axis = 1)

    #For label type
    conditioning_labels_type_array = np.argmax(class_data['conditioning_type'], axis = 1)
    pred_labels_type_array = np.argmax(class_data['predicted_type'], axis = 1)
    true_labels_type_array = np.argmax(class_data['true_type'], axis = 1)

    #For Discriminator Prediction on Real images
    pred_real_labels_array = np.argmax(class_data['labels_true_pred'], axis = 1)
    pred_real_labels_type_array = np.argmax(class_data['labels_type_true_pred'], axis = 1)


    Index_of_relevant_classes = np.where(conditioning_labels_array == i)[0]
    sample_img_per_class.append(class_data['generated'][Index_of_relevant_classes[:num_of_sample_images]])

    labels_array = np.ones(num_of_sample_images)*i
    list_of_labels.append(labels_array)


    save_imgs(class_data['true'], "Real", i, true_labels_array, true_labels_type_array, \
        pred_real_labels_array, pred_real_labels_type_array, \
        true_labels_array, true_labels_type_array , output_dir, label_dictionary)

    save_imgs(class_data['generated'], "Generated", i ,conditioning_labels_array, conditioning_labels_type_array, \
        predicted_labels_array, pred_labels_type_array, \
        true_labels_array, true_labels_type_array, output_dir, label_dictionary)
    
    temp_hold_imgs = []
    sampling_imgs_taken = False
    last = 0
    for j in range(class_data['generated'].shape[0]):
        if i == np.argmax(class_data['conditioning'][j]):
            true_positives += 1

        if np.argmax(class_data['true_type'][j]) == np.argmax(class_data['conditioning_type'][j]):
            true_positives_type += 1

        y_true = class_data['true'][j][:,:,:]
        y_pred = class_data['generated'][j][:,:,:]

        all_true_images.append(y_true)
        all_generated_images.append(y_pred)

        if i not in classes_added:
            # print("Hello")
            # if j not in types_added:
                # print("In here")
                
            if  np.argmax(class_data['true_type'][j]) == last:
                last += 1

                # print("Now adding images")
                # print(y_true.dtype)
                temp_hold_imgs.append([y_true, y_pred])
                # print(temp_hold_imgs[0][0].dtype)
                types_added.append(j)
            if len(types_added) == y_secondary_test.shape[1] and not sampling_imgs_taken:

                classes_added.append(i)
                comparison_imgs.append(temp_hold_imgs)
                # print(len(temp_hold_imgs))
                sampling_imgs_taken = True
                
        # print(y_true.shape)
        # print(y_pred.shape)
        # print(y_pred)



        #SSIM score
        ssim_value = compute_ssim(y_true, y_pred)
        
        #RMSE
        rmse_value = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
        
        #PSNR value
        psnr_value = psnr(y_true, y_pred, data_range = 255)

        #FSIM
        fsim_value = fsim(y_true, y_pred)

        #UIQ 
        uiq_value = compute_uiq(y_true, y_pred)




        #Append all scores
        ssim_scores.append(ssim_value)
        rmse_scores.append(rmse_value)
        psnr_scores.append(psnr_value)
        fsim_scores.append(fsim_value)
        uiq_scores.append(uiq_value)

        # F1_scores.append(F1_value)
        # recall_scores.append(recall_value)
        # precision_scores.append(precision_value)

        # F1_type_scores.append(F1_type_value)
        # recall_type_scores.append(recall_type_value)
        # precision_type_scores.append(precision_type_value)

    #Inception
    # print(y_pred.shape)
    # print(type(y_pred))



    # Convert all images
    # Obtain inception for fake
    tensor_images_fake = torch.stack([transform(img) for img in class_data['generated']]).to(device)
    tensor_images_fake_dataset = TensorDataset(tensor_images_fake) 
    dataloader = DataLoader(tensor_images_fake_dataset, batch_size=32, shuffle=False)

    inception.reset()
    for (batch,) in tqdm(dataloader, desc="Computing Inception Score for genereated"):
        inception.update(batch.to(device))
    inception_value = inception.compute()
    inception_value = [tensor.cpu() for tensor in inception_value]


    #Obtain inception for real
    tensor_images_real = torch.stack([transform(img) for img in class_data['true']]).to(device)
    tensor_images_real_dataset = TensorDataset(tensor_images_real) 
    dataloader = DataLoader(tensor_images_real_dataset, batch_size=32, shuffle=False)

    inception.reset()
    for (batch,) in tqdm(dataloader, desc="Computing Inception Score for Real"):
        inception.update(batch.to(device))
    inception_value_real = inception.compute()
    inception_value_real = [tensor.cpu() for tensor in inception_value_real]
    
    # tensor_images = torch.stack([transform(img) for img in class_data['generated']])
    # inception_value = inception(tensor_images)

    # tensor_images_real = torch.stack([transform(img) for img in class_data['true']])
    # inception_value_real = inception(tensor_images_real)

    #F1
    F1_value = f1_score(true_labels_array, conditioning_labels_array, average='macro')
    F1_type_value = f1_score(true_labels_type_array, conditioning_labels_type_array, average='macro')


    #Recall
    recall_value = recall_score(true_labels_array, conditioning_labels_array, average='macro')
    recall_type_value = recall_score(true_labels_type_array, conditioning_labels_type_array, average='macro')

    #Precision
    precision_value = precision_score(true_labels_array,conditioning_labels_array, average='macro')
    precision_type_value = precision_score(true_labels_type_array,conditioning_labels_type_array, average='macro')
        


    evaluation[i] = {'average_ssim': np.mean(ssim_scores),'average_rmse':np.mean(rmse_scores),'average_psnr':np.mean(psnr_scores),'average_fsim':np.mean(fsim_scores),'average_uiq':np.mean(uiq_scores), 'average_inception':inception_value,\
        'average_real_inception': inception_value_real, \
        'average_F1':F1_value,'average_recall':recall_value,'average_precision':precision_value,\
        'average_type_F1':F1_type_value,'average_type_recall':recall_type_value,'average_type_precision':precision_type_value}

    class_acc = true_positives / class_data['generated'].shape[0]
    class_type_acc = true_positives_type / class_data['generated'].shape[0]
    text_to_print = f"Class {i} ({label_dictionary[i]}): mean ssim: {evaluation[i]['average_ssim']:.2f}, mean rmse: {evaluation[i]['average_rmse']:.2f}, mean psnr: {evaluation[i]['average_psnr']:.2f},  \
        mean fsim: {evaluation[i]['average_fsim']:.2f}, mean uiq: {evaluation[i]['average_uiq']:.2f}, Fake Inception: {evaluation[i]['average_inception'][0]:.3f} ~ inception std {evaluation[i]['average_inception'][1]:.3f}, Real Inception: {evaluation[i]['average_real_inception'][0]:.3f} ~ inception std {evaluation[i]['average_real_inception'][1]:.3f} ,classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
        mean F1 {evaluation[i]['average_F1']:.2f}, mean recall {evaluation[i]['average_recall']:.2f}, mean precision {evaluation[i]['average_precision']:.2f} , \n \
        mean type F1 {evaluation[i]['average_type_F1']:.2f}, mean type recall {evaluation[i]['average_type_recall']:.2f}, mean type precision {evaluation[i]['average_type_precision']:.2f} \n   "
    text_to_save.append(text_to_print)
    print(text_to_print)


    mean_ssim_scores.append(evaluation[i]['average_ssim'])
    mean_rmse_scores.append(evaluation[i]['average_rmse'])
    mean_psnr_scores.append(evaluation[i]['average_psnr'])
    mean_fsim_scores.append(evaluation[i]['average_fsim'])
    mean_uiq_scores.append(evaluation[i]['average_uiq'])
    mean_inception_scores.append(evaluation[i]['average_inception'][0])
    mean_inception_stds.append(evaluation[i]['average_inception'][1])
    mean_real_inception_scores.append(evaluation[i]['average_real_inception'][0])
    mean_real_inception_stds.append(evaluation[i]['average_real_inception'][1])

    mean_accuracy_scores.append(class_acc)
    mean_accuracy_type_scores.append(class_type_acc)

    mean_F1_scores.append(evaluation[i]['average_F1'])
    mean_recall_scores.append(evaluation[i]['average_recall'])
    mean_precision_scores.append(evaluation[i]['average_precision'])

    mean_type_F1_scores.append(evaluation[i]['average_type_F1'])
    mean_type_recall_scores.append(evaluation[i]['average_type_recall'])
    mean_type_precision_scores.append(evaluation[i]['average_type_precision'])

mean_evaluation = {'average_ssim':np.mean(mean_ssim_scores),'average_rmse':np.mean(mean_rmse_scores),'average_psnr':np.mean(mean_psnr_scores),'average_fsim':np.mean(mean_fsim_scores),'average_uiq':np.mean(mean_uiq_scores), 'average_inception':np.mean(mean_inception_scores), 'average_stds': np.mean(mean_inception_stds),\
    'average_real_inception':np.mean(mean_real_inception_scores), 'average_real_stds': np.mean(mean_real_inception_stds), \
    'average_accuracy':np.mean(mean_accuracy_scores), 'average_type_accuracy':np.mean(mean_accuracy_type_scores), \
    'average_f1' : np.mean(mean_F1_scores), 'average_recall' : np.mean(mean_recall_scores), 'average_precision' : np.mean(mean_precision_scores), \
    'average_type_f1' : np.mean(mean_type_F1_scores), 'average_type_recall' : np.mean(mean_type_recall_scores), 'average_type_precision' : np.mean(mean_type_precision_scores), 
        }


##Inception on whole dataset
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),        # InceptionV3 expects 299x299
    transforms.PILToTensor(),                # Converts to [0, 1] float tensor and rearranges to (C, H, W)
])



# Batch data to allow larger dataset processing
all_tensor_images = torch.stack([transform(img) for img in all_generated_images])
all_tensor_dataset = TensorDataset(all_tensor_images) 
dataloader = DataLoader(all_tensor_dataset, batch_size=32, shuffle=False)

inception.reset()
for (batch,) in tqdm(dataloader, desc="Computing Inception Score for genereated"):
    inception.update(batch.to(device))

inception_value = inception.compute()
inception_value = [tensor.cpu() for tensor in inception_value]

# Batch data to allow larger dataset processing
all_real_tensor_images = torch.stack([transform(img) for img in all_true_images])
all_real_tensor_dataset = TensorDataset(all_real_tensor_images) 
dataloader_real = DataLoader(all_real_tensor_dataset, batch_size=32, shuffle=False)

inception.reset()
for (batch,) in tqdm(dataloader_real, desc="Computing Inception Score for real"):
    inception.update(batch.to(device))

real_inception_value = inception.compute()
real_inception_value = [tensor.cpu() for tensor in real_inception_value]

mean_evaluation['overall_real_inception'] = real_inception_value[0].numpy()
mean_evaluation['overall_real_inception_std'] = real_inception_value[1].numpy()

mean_evaluation['overall_fake_inception'] = inception_value[0].numpy()
mean_evaluation['overall_fake_inception_std'] = inception_value[1].numpy()

# all_tensor_images = torch.stack([transform(img) for img in all_generated_images])
# inception_value = inception(all_tensor_images)

# all_real_tensor_images = torch.stack([transform(img) for img in all_true_images])
# real_inception_value = inception(all_real_tensor_images)

mean_text_to_print = f"Average Class Results: mean ssim: {mean_evaluation['average_ssim']:.2f}, mean rmse: {mean_evaluation['average_rmse']:.2f}, mean psnr: {mean_evaluation['average_psnr']:.2f}, \
    mean fsim: {mean_evaluation['average_fsim']:.2f}, mean uiq: {mean_evaluation['average_uiq']:.2f}, mean classification acc: {mean_evaluation['average_accuracy']:.1%} ,mean type classification acc: {mean_evaluation['average_type_accuracy']:.1%} \n \
    Fake mean inception: {mean_evaluation['average_inception']:.2f} ~ mean stds: {mean_evaluation['average_stds']:.2f} | Real mean inception: {mean_evaluation['average_real_inception']:.2f} ~ mean stds: {mean_evaluation['average_real_stds']:.2f} | Overall Fake inception: {mean_evaluation['overall_fake_inception']:.2f} ~ Overall stds: {mean_evaluation['overall_fake_inception_std']:.2f} | Overall Real inception: {mean_evaluation['overall_real_inception']:.2f} ~ Overall stds: {mean_evaluation['overall_real_inception_std']:.2f}\n \
    mean F1: {mean_evaluation['average_f1']:.2f}, mean recall: {mean_evaluation['average_recall']:.2f}, mean precision: {mean_evaluation['average_precision']:.2f} \n \
    mean type F1: {mean_evaluation['average_type_f1']:.2f}, mean type recall: {mean_evaluation['average_type_recall']:.2f}, mean type precision: {mean_evaluation['average_type_precision']:.2f} \n        "
print(mean_text_to_print)
text_to_save.append(mean_text_to_print)


save_imgs_comparison(comparison_imgs, output_dir)
save_imgs_horizontal_vertical(comparison_imgs, output_dir)

# stacked_images = np.stack(sample_img_per_class, axis=0)
# stacked_labels = np.stack(list_of_labels, axis = 0)

# stacked_images = stacked_images.reshape(-1, 28, 28, 3)
# stacked_labels = stacked_labels.reshape(-1)
# save_imgs(stacked_images, "Sampling image of each class", "all" ,stacked_labels, stacked_labels, output_dir)

with open(f"{output_dir}/results.txt", "w") as file:
    file.write("\n".join(text_to_save) + "\n")



# Create DataFrame
evaluation[(class_labels[-1]+1)] = mean_evaluation
df = pd.DataFrame.from_dict(evaluation, orient="index")

# Export to Excel
df.to_excel(f"{output_dir}/output.xlsx", index=True)

pass