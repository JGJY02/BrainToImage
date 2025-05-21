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
from models.eegclassifier import convolutional_encoder_model_512_dual
from models.dual_models.eeggan import (build_discriminator, build_EEGgan, build_generator,
                           build_MoGCgenerator, build_MoGMgenerator,
                           combine_loss_metrics, sample_images, save_model)


import tensorflow as tf
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

import argparse


parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset - CNN_encoder / LSTM_encoder / Transformer", default = "processed_dataset/filter_mne_car/CNN_encoder/All",required=False)
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training xxxthresh_(channels)stack(model)_(dataset) 000thresh_AllSlidingCNN_All.pkl / 000thresh_AllStackLstm_All.pkl / 000thresh_AllStackTransformer_All", default = "000thresh_AllSlidingCNN_dual_28_ori_All.pkl" , required=False)
parser.add_argument('--imageOrwindowed', type=str, help="spectrogram for image windowed for original", default = "windowed" , required=False)
parser.add_argument('--model_name', type=str, help="Name of the model", default= "CNN_all_stacked_signals_dual_512_ori", required=False)
parser.add_argument('--classifier_path', type=str, help="Directory to output", default = "trained_models/classifiers/All/000thresh",required=False)

parser.add_argument('--gan_path', type=str, help="Directory to output", default = "trained_models/GANs/CNN_GAN/AC/000thresh_Basic_512",required=False)
parser.add_argument('--prefix', type=str, help="Basic MogM or MoGC", default = "Basic",required=False)


parser.add_argument('--output_path', type=str, help="Directory to output", default = "results/CNN_ACGAN_B_512_ori",required=False)

args = parser.parse_args()


#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition
print(os.getcwd())
eeg_latent_dim = 512
class_labels = [0,1,2,3,4,5,6,7,8,9]
valid_threshold = 0.5


## load the eeg training data
eeg_data_dir = args.root_dir
data_file = args.dataset_pickle
print(f"Reading data file {eeg_data_dir}/{data_file}")
eeg_dataset = f"{eeg_data_dir}/{data_file}"
print(f"Reading data file {eeg_dataset}")
eeg_data = pickle.load(open(eeg_dataset, 'rb'), encoding='bytes')
label_dictionary = eeg_data['dictionary']
signals = eeg_data['x_test_eeg']
(x_train, y_train, y_secondary_train) , (x_test, y_test, y_secondary_test) = (eeg_data['x_train_img'], eeg_data['y_train'], eeg_data['y_secondary_train']) , (eeg_data['x_test_img'], eeg_data['y_test'], eeg_data['y_secondary_test'])

num_of_class_labels = y_train.shape[1]
num_of_class_type_labels = y_secondary_train.shape[1]
to_labels = np.argmax(eeg_data['y_test'],axis=1)  ## since eeg labels are in one-hot encoded format

## #############
# Build EEG Gan
## #############
prefix = args.prefix
model_dir = f"{args.gan_path}/{prefix}_"

if prefix == "MoGC":
    generator = build_MoGCgenerator(eeg_latent_dim,x_train.shape[3],num_of_class_labels, num_of_class_type_labels)
elif prefix == "MoGM":
    generator = build_MoGMgenerator(eeg_latent_dim,x_train.shape[3],num_of_class_labels, num_of_class_type_labels)
elif prefix == "Basic":
    generator = build_generator(eeg_latent_dim,x_train.shape[3],num_of_class_labels, num_of_class_type_labels)

#generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
#generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))

generator.load_weights(f"{model_dir}EEGGan_generator_weights.h5")
print(x_train.shape)
discriminator = build_discriminator((x_train.shape[1],x_train.shape[2],x_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
combined = build_EEGgan(eeg_latent_dim,generator,discriminator)
combined.load_weights(f"{model_dir}EEGgan_combined_weights.h5")

## #############
# EEG Classifier/Encoder
## #############
classifier = convolutional_encoder_model_512_dual(eeg_data['x_train_eeg'].shape[1], eeg_data['x_train_eeg'].shape[2], num_of_class_labels, num_of_class_type_labels)
classifier_model_path = f"{args.classifier_path}/{args.model_name}/eeg_classifier_adm5_final.h5"
classifier.load_weights(classifier_model_path)
## #############################################################
# Make prediction on random selected eeg from eeg_data['x_test']
## #############################################################
layer_names = ['EEG_feature_BN2','EEG_Class_Labels','EEG_Class_type_Labels']
encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

dataset = tf.data.Dataset.from_tensor_slices(signals)
batch_size = 64
dataset = dataset.batch(batch_size)

encoded_latents = []
encoded_labels = []
encoded_type_labels = []

for batch in tqdm(dataset):
    encodedEEG, encodedLabel, encodedTypeLabels = encoder_model(batch, training=False)
    encoded_latents.append(encodedEEG)
    encoded_labels.append(encodedLabel)
    encoded_type_labels.append(encodedTypeLabels)

# Combine results
encoded_latents = np.concatenate(encoded_latents, axis=0)
encoded_labels = np.concatenate(encoded_labels, axis=0)
encoded_type_labels = np.concatenate(encoded_type_labels, axis =0)


history = {}
for i in class_labels:  ## outer loop per class
    print("Current class label is : ", i)
    ## get all EEG data for class i
    matching_indices = np.where(to_labels == i)
    true_images = eeg_data['x_test_img'][matching_indices[0]]
    labels = eeg_data['y_test'][matching_indices[0]]

    encoded_eegs = encoded_latents[matching_indices[0]]
    conditioning_labels_raw = encoded_labels[matching_indices[0]]
    
    conditioning_labels_type_raw = encoded_type_labels[matching_indices[0]]
    true_conditioning_labels_type = eeg_data['y_secondary_test'][matching_indices[0]]

        
    # minibatch_size = np.clip(8192 // encoded_eegs.shape[1], 4, 256)
    minibatch_size = 4
    # print("The encoded eegs shape is :", encoded_eegs.shape)
    # print("The calculated minibatch size is :", minibatch_size)
    conditioning_labels_argmax = np.argmax(conditioning_labels_raw, axis=1)
    conditioning_labels_type_argmax = np.argmax(conditioning_labels_type_raw, axis = 1)

    generated_samples = generator.predict([encoded_eegs, conditioning_labels_argmax, conditioning_labels_type_argmax],batch_size=32)
    ## predict on GAN
    validitys, labels_pred, labels_type_pred = combined.predict([encoded_eegs, conditioning_labels_argmax, conditioning_labels_type_argmax],batch_size=32)

        ## predict on GAN
    # true_images = np.pad(true_images, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0) #pad images as progressiveGAN did so as well
    # true_images_test = np.transpose(true_images, (0, 3, 1, 2)) 
    true_images_test = (true_images/127.5) - 1

    validitys_true, labels_true_pred, labels_type_true_pred  = discriminator.predict([true_images_test], batch_size=32)
    # generated_samples = np.transpose(generated_samples, (0, 2, 3, 1))

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
        'predicted_type': labels_type_pred,'conditioning_type': conditioning_labels_type_raw, 'true_type': true_conditioning_labels_type, \
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

inception = InceptionScore()

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
output_dir = args.output_path

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

        # print(y_true.shape)
        # print(y_pred.shape)
        # print(y_pred)


        if i not in classes_added:
            # print("Hello")
            # if j not in types_added:
                # print("In here")
                
            if  np.argmax(class_data['true_type'][j]) == last:
                last += 1

                # print("Now adding images")
                temp_hold_imgs.append([y_true, y_pred])
                types_added.append(j)
            if len(types_added) == eeg_data['y_secondary_test'].shape[1] and not sampling_imgs_taken:

                classes_added.append(i)
                comparison_imgs.append(temp_hold_imgs)
                # print(len(temp_hold_imgs))
                sampling_imgs_taken = True                          

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
    tensor_images = torch.stack([transform(img) for img in class_data['generated']])
    inception_value = inception(tensor_images)

    tensor_images_real = torch.stack([transform(img) for img in class_data['true']])
    inception_value_real = inception(tensor_images_real)
    
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
all_tensor_images = torch.stack([transform(img) for img in all_generated_images])
inception_value = inception(all_tensor_images)

all_real_tensor_images = torch.stack([transform(img) for img in all_true_images])
real_inception_value = inception(all_real_tensor_images)

mean_text_to_print = f"Average Class Results: mean ssim: {mean_evaluation['average_ssim']:.2f}, mean rmse: {mean_evaluation['average_rmse']:.2f}, mean psnr: {mean_evaluation['average_psnr']:.2f}, \
    mean fsim: {mean_evaluation['average_fsim']:.2f}, mean uiq: {mean_evaluation['average_uiq']:.2f}, mean classification acc: {mean_evaluation['average_accuracy']:.1%} ,mean type classification acc: {mean_evaluation['average_type_accuracy']:.1%} \n \
    Fake mean inception: {mean_evaluation['average_inception']:.2f} ~ mean stds: {mean_evaluation['average_stds']:.2f} | Real mean inception: {mean_evaluation['average_real_inception']:.2f} ~ mean stds: {mean_evaluation['average_real_stds']:.2f} | Overall Fake inception: {inception_value[0]:.2f} ~ Overall stds: {inception_value[1]:.2f} | Overall Real inception: {real_inception_value[0]:.2f} ~ Overall stds: {real_inception_value[1]:.2f}\n \
    mean F1: {mean_evaluation['average_f1']:.2f}, mean recall: {mean_evaluation['average_recall']:.2f}, mean precision: {mean_evaluation['average_precision']:.2f} \n \
    mean type F1: {mean_evaluation['average_type_f1']:.2f}, mean type recall: {mean_evaluation['average_type_recall']:.2f}, mean type precision: {mean_evaluation['average_type_precision']:.2f} \n        "
print(mean_text_to_print)
text_to_save.append(mean_text_to_print)

# stacked_images = np.stack(sample_img_per_class, axis=0)
# stacked_labels = np.stack(list_of_labels, axis = 0)

# stacked_images = stacked_images.reshape(-1, 28, 28, 3)
# stacked_labels = stacked_labels.reshape(-1)
# save_imgs(stacked_images, "Sampling image of each class", "all" ,stacked_labels, stacked_labels, output_dir)
save_imgs_comparison(comparison_imgs, output_dir)

with open(f"{output_dir}/results.txt", "w") as file:
    file.write("\n".join(text_to_save) + "\n")

pass