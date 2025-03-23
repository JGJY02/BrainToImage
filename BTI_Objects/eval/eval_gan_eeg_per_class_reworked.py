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
from models.eeggan import (build_discriminator, build_EEGgan, build_MoGCgenerator, build_MoGMgenerator, build_generator)
from models.dcgan import (build_dc_discriminator, build_DCGgan, build_dc_generator)
from models.model_utils import (sample_images_eeg, save_model, combine_loss_metrics)
import argparse

## new metrics
from sklearn.metrics import mean_squared_error
from image_similarity_measures.quality_metrics import fsim
import tensorflow as tf
import cv2



"""
Title: EEG ACGAN Evaluations of models

Purpose:
    Testing design and build of EEG ACGAN and classifier, Functional blocks for training
    ACGAN model. Call from training script.

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    Run the script as is, uses the online MNIST dataset to train the GAN

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""

#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--eeg_dataset_dir', type=str, help="Directory to eeg datasets", default = "processed_dataset/filter_mne_car/LSTM_encoder/All" , required=False)
parser.add_argument('--eeg_dataset_file_path', type=str, help="specific eeg file to use LSTM: 000thresh_AllStackLstm_All.pkl / CNN: 000thresh_AllSlidingCNN_All.pkl", default = "000thresh_AllStackLstm_All.pkl" , required=False)

parser.add_argument('--model_dir', type=str, help="specific eeg file to use", default = "trained_models" , required=False)
parser.add_argument('--classifier_file_path', type=str, help="specific eeg file to use LSTM: All/000thresh/LSTM_all_stacked_signals / CNN: All/000thresh/CNN_all_stacked_signals", default = "All/000thresh/LSTM_all_stacked_signals" , required=False)
parser.add_argument('--GAN_type', type=str, help="DC or AC or Caps", default = "AC",required=False)
parser.add_argument('--classifierType', type = str, help = "CNN or LSTM", default = "LSTM")

parser.add_argument('--model_type', type=str, help="M,B,C", default= "B", required=False)
parser.add_argument('--gan_file_path', type=str, help="specific eeg file to use", default = "000thresh" , required=False)

parser.add_argument('--output_dir', type=str, help="Directory to save processed file", default = "results",required=False)
parser.add_argument('--output_name', type=str, help="Directory to save processed file", default = "LSTM_ACGAN",required=False)
args = parser.parse_args()

eeg_latent_dim = 128
class_labels = [0,1,2,3,4,5,6,7,8,9]
valid_threshold = 0.5


## load the eeg training data
dataset = "2022Data"
data_file = f"{args.eeg_dataset_dir}/{args.eeg_dataset_file_path}"
print(f"Reading data file {data_file}")
eeg_data = pickle.load(open(data_file, 'rb'), encoding='bytes')

to_labels = np.argmax(eeg_data['y_test'],axis=1)  ## since eeg labels are in one-hot encoded format

for i in range(10):
    matching_indices = np.where(to_labels == i)
    print("Length of index ", i, "is ", len(matching_indices[0]))
    

print(f" Loading Object images from {data_file}")
(x_test_img, y_test) = (eeg_data['x_test_img'], eeg_data['y_test'])

## #############
# Build EEG Gan
## #############
generator_type = args.model_type #C for concatenation M for Multiplication B for Basic
if args.GAN_type == "AC":
    if generator_type == "B":
        model_type = "Basic"
    else:
        model_type = f"MoG{generator_type}"

else: model_type = args.GAN_type

output_dir = f"{args.output_dir}/{args.output_name}/{args.gan_file_path}_{model_type}"

prefix = model_type
model_dir = args.model_dir

gan_dir = f"{model_dir}/GANs/{args.classifierType}_GAN/{args.GAN_type}/{args.gan_file_path}_{model_type}"

print(eeg_data['x_train_img'][0].shape)

if args.GAN_type == "AC":
    if prefix == "MoGC":
        generator = build_MoGCgenerator(eeg_latent_dim,eeg_data['x_train_img'][0].shape[2],len(class_labels))
    elif prefix == "MoGM":
        generator = build_MoGMgenerator(eeg_latent_dim,eeg_data['x_train_img'][0].shape[2],len(class_labels))
    elif prefix == "Basic":
        generator = build_generator(eeg_latent_dim,eeg_data['x_train_img'][0].shape[2],len(class_labels))
    generator.load_weights(f"{gan_dir}/{prefix}_EEGGan_generator_weights.h5")
    discriminator = build_discriminator((28,28,3),len(class_labels))

elif args.GAN_type == "DC":
    generator = build_dc_generator(eeg_latent_dim, eeg_data['x_train_img'][0].shape[2],len(class_labels))
    generator.load_weights(f"{gan_dir}/{prefix}_EEGGan_generator_weights.h5")

    discriminator = build_dc_discriminator((eeg_data['x_train_img'][0].shape[0], eeg_data['x_train_img'][0].shape[1],eeg_data['x_train_img'][0].shape[2]),len(class_labels))

#generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
#generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))


combined = build_EEGgan(eeg_latent_dim,len(class_labels),generator,discriminator)
combined.load_weights(f"{gan_dir}/{prefix}_EEGgan_combined_weights.h5")

## #############
# EEG Classifier/Encoder
## #############
classifier_dir = f"{model_dir}/classifiers/{args.classifier_file_path}"

if args.classifierType == "LSTM":
    classifier = LSTM_Classifier(eeg_data['x_train_eeg'].shape[1], eeg_data['x_train_eeg'].shape[2], len(class_labels))

elif args.classifierType == "CNN":
    classifier = convolutional_encoder_model(eeg_data['x_train_eeg'].shape[1], eeg_data['x_train_eeg'].shape[2], len(class_labels))

classifier_model_path = f"{classifier_dir}/eeg_classifier_adm5_final.h5"
classifier.load_weights(classifier_model_path)
# we need to classifier encoded laten space as input to the EEGGan model
layer_names = ['EEG_feature_BN2','EEG_Class_Labels']
encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

## #############################################################
# Make prediction on random selected eeg from eeg_data['x_test']
## #############################################################
history = {}
for i in class_labels:  ## outer loop per class
    ## get all EEG data for class i
    matching_indices = np.where(to_labels == i)
    eeg_samples = eeg_data['x_test_eeg'][matching_indices[0]]
    #gt_labels = np.full(eeg_samples.shape[0],i,dtype=int)
    ## get enough MNIST samples of class i to match eeg_samples
    # matching_indices = np.where(y_test == i)
    # matching_indices = np.random.choice(matching_indices[0],eeg_samples.shape[0],replace=False)

    true_images = x_test_img[matching_indices[0]]
    labels = y_test[matching_indices[0]]

    ## classify and enncode the EEG signals for input to GAN
    encoded_eegs, conditioning_labels_raw = encoder_model.predict(eeg_samples,batch_size=32)
    conditioning_labels = np.argmax(conditioning_labels_raw,axis=1)
    generated_samples = generator.predict([encoded_eegs, conditioning_labels],batch_size=32)
    ## predict on GAN
    generated_samples = generated_samples*127.5 + 127.5
    generated_samples = np.clip(generated_samples,0,255)
    generated_samples = generated_samples.astype(np.uint8)

    
    validitys, labels_pred = combined.predict([encoded_eegs, conditioning_labels],batch_size=32)



    ## collate results
    history[i] = {'generated':generated_samples,'true':true_images,'valid':validitys,'predicted':labels_pred,'conditioning':conditioning_labels_raw, 'true_labels':labels}



def save_imgs(images, name, class_label, conditioning_labels, predicted_labels, output_dir):
    # Set up the grid dimensions (10x10)
    rows = 10
    cols = 10

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Loop through the axes and images to display them

    for k, ax in enumerate(axes.flat):
        if k < len(images):
            ax.imshow(images[k])
            ax.set_title(f"C: {int(conditioning_labels[k])}, P: {int(predicted_labels[k])}", fontsize=8)
            ax.axis('off')  # Hide the axes
        else:
            ax.axis('off')  # In case there are empty subplots


    # Save the grid of images to a file
    output_path = f'{output_dir}/{name}_class{class_label}.png'
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


mean_ssim_scores = []
mean_rmse_scores = []
mean_psnr_scores =[]
mean_fsim_scores = []
mean_uiq_scores = []
mean_accuracy_scores = []

num_of_sample_images = 10

sample_img_per_class = []
list_of_labels = []
text_to_save = []

evaluation ={}
for i in class_labels:
    class_data = history[i]
    ssim_scores = []
    rmse_scores = []
    psnr_scores =[]
    fsim_scores = []
    uiq_scores = []



    true_positives = 0


    conditioning_labels_array = np.argmax(class_data['conditioning'], axis = 1)
    predicted_labels_array = np.argmax(class_data['predicted'], axis = 1)

    true_labels_array = np.argmax(class_data['true_labels'], axis = 1)


    Index_of_relevant_classes = np.where(conditioning_labels_array == i)[0]
    sample_img_per_class.append(class_data['generated'][Index_of_relevant_classes[:num_of_sample_images]])

    labels_array = np.ones(num_of_sample_images)*i
    list_of_labels.append(labels_array)


    save_imgs(class_data['true'], "Real", i, true_labels_array, true_labels_array, output_dir)
    save_imgs(class_data['generated'], "Generated", i ,conditioning_labels_array, predicted_labels_array, output_dir)

    for j in range(class_data['generated'].shape[0]):
        if i == np.argmax(class_data['predicted'][j]):
            true_positives += 1

        y_true = class_data['true'][j][:,:,:]
        y_pred = class_data['generated'][j][:,:,:]

        # print(y_true.shape)
        # print(y_pred.shape)
        # print(y_pred)

        #SSIM score
        ssim_value = compute_ssim(y_true, y_pred)
        
        #RMSE
        rmse_value = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
        
        #PSNR value
        psnr_value = tf.image.psnr(y_true, y_pred, 255)

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
        


    evaluation[i] = {'average_ssim':np.mean(ssim_scores),'average_rmse':np.mean(rmse_scores),'average_psnr':np.mean(psnr_scores),'average_fsim':np.mean(fsim_scores),'average_uiq':np.mean(uiq_scores)}
    class_acc = true_positives / class_data['generated'].shape[0]
    text_to_print = f"Class {i}: mean ssim: {evaluation[i]['average_ssim']:.2f}, mean rmse: {evaluation[i]['average_rmse']:.2f}, mean psnr: {evaluation[i]['average_psnr']:.2f}, mean fsim: {evaluation[i]['average_fsim']:.2f}, mean uiq: {evaluation[i]['average_uiq']:.2f},classification acc: {class_acc:.1%}"
    text_to_save.append(text_to_print)
    print(text_to_print)


mean_ssim_scores.append(evaluation[i]['average_ssim'])
mean_rmse_scores.append(evaluation[i]['average_rmse'])
mean_psnr_scores.append(evaluation[i]['average_psnr'])
mean_fsim_scores.append(evaluation[i]['average_fsim'])
mean_uiq_scores.append(evaluation[i]['average_uiq'])
mean_accuracy_scores.append(class_acc)




mean_evaluation = {'average_ssim':np.mean(ssim_scores),'average_rmse':np.mean(rmse_scores),'average_psnr':np.mean(psnr_scores),'average_fsim':np.mean(fsim_scores),'average_uiq':np.mean(uiq_scores), 'average_accuracy':np.mean(mean_accuracy_scores)}
mean_text_to_print = f"Average Class Results: mean ssim: {mean_evaluation['average_ssim']:.2f}, mean rmse: {mean_evaluation['average_rmse']:.2f}, mean psnr: {mean_evaluation['average_psnr']:.2f}, mean fsim: {mean_evaluation['average_fsim']:.2f}, mean uiq: {mean_evaluation['average_uiq']:.2f},mean classification acc: {mean_evaluation['average_accuracy']:.1%}"
print(mean_text_to_print)
text_to_save.append(mean_text_to_print)

stacked_images = np.stack(sample_img_per_class, axis=0)
stacked_labels = np.stack(list_of_labels, axis = 0)

stacked_images = stacked_images.reshape(-1, 28, 28, 3)
stacked_labels = stacked_labels.reshape(-1)
save_imgs(stacked_images, "Sampling image of each class", "all" ,stacked_labels, stacked_labels, output_dir)

with open(f"{output_dir}/results.txt", "w") as file:
    file.write("\n".join(text_to_save) + "\n")

pass