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
from models.eegclassifier import convolutional_encoder_model
from models.eeggan import (build_discriminator, build_EEGgan, build_generator,
                           build_MoGCgenerator, build_MoGMgenerator,
                           combine_loss_metrics, sample_images, save_model)
from utils.local_MNIST import (get_balanced_mnist_subset, load_local_mnist,
                               sample_image_by_label)


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
print(os.getcwd())
eeg_latent_dim = 128
class_labels = [0,1,2,3,4,5,6,7,8,9]
valid_threshold = 0.5

## load the MNIST trainig data
image_data_dir = "Datasets\MNIST_dataset"
print(f" Loading MNIST images from {image_data_dir}")
_ , (x_test, y_test) = load_local_mnist(image_data_dir,norm=-1,sparse=True)

## load the eeg training data
dataset = "2022Data"
eeg_data_dir = f"Datasets/MindBigData - The Visual MNIST of Brain Digits/{dataset}"
data_file = "data_4ch_epoch_filtered_324_0-85test_MindBigData2022_MNIST_EP.pkl"
print(f"Reading data file {eeg_data_dir}/{data_file}")
eeg_data = pickle.load(open(f"{eeg_data_dir}/{data_file}", 'rb'), encoding='bytes')
to_labels = np.argmax(eeg_data['y_test'],axis=1)  ## since eeg labels are in one-hot encoded format
## #############
# Build EEG Gan
## #############
prefix = "MoGM"
model_dir = f"./brain_to_Image/EEGgan/EEG_saved_model/{prefix}/{prefix}_"

if prefix == "MoGC":
    generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))
elif prefix == "MoGM":
    generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
elif prefix == "Basic":
    generator = build_generator(eeg_latent_dim,1,len(class_labels))

#generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
#generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))

generator.load_weights(f"{model_dir}EEGGan_generator_weights.h5")
discriminator = build_discriminator((28,28,1),len(class_labels))
combined = build_EEGgan(eeg_latent_dim,len(class_labels),generator,discriminator)
combined.load_weights(f"{model_dir}EEGgan_combined_weights.h5")

## #############
# EEG Classifier/Encoder
## #############
classifier = convolutional_encoder_model(eeg_data['x_train'].shape[1], eeg_data['x_train'].shape[2], len(class_labels))

classifier_model_path = f"Datasets\MindBigData - The Visual MNIST of Brain Digits\MNIST_EP\models\eeg_classifier_adm5_final.h5"
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
    eeg_samples = eeg_data['x_test'][matching_indices[0]]
    #gt_labels = np.full(eeg_samples.shape[0],i,dtype=int)
    ## get enough MNIST samples of class i to match eeg_samples
    matching_indices = np.where(y_test == i)
    matching_indices = np.random.choice(matching_indices[0],eeg_samples.shape[0],replace=False)
    mnist_images = x_test[matching_indices]
    labels = y_test[matching_indices]

    ## classify and enncode the EEG signals for input to GAN
    encoded_eegs, conditioning_labels = encoder_model.predict(eeg_samples,batch_size=32)
    conditioning_labels = np.argmax(conditioning_labels,axis=1)
    generated_samples = generator.predict([encoded_eegs, conditioning_labels],batch_size=32)
    ## predict on GAN
    validitys, labels = combined.predict([encoded_eegs, conditioning_labels],batch_size=32)

    ## collate results
    history[i] = {'generated':generated_samples,'mnist':mnist_images,'valid':validitys,'predicted':labels}



def save_imgs(images, name, class_label):
    # Set up the grid dimensions (10x10)
    rows = 10
    cols = 10

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Loop through the axes and images to display them
    for k, ax in enumerate(axes.flat):
        if k < len(images):
            ax.imshow(images[k], cmap='gray')
            ax.axis('off')  # Hide the axes
        else:
            ax.axis('off')  # In case there are empty subplots

    # Save the grid of images to a file
    output_path = f'C:/Brain_To_Image_ProgressiveGAN/results/OriginalEEGan/{name}_class{class_label}.png'
    if not os.path.exists(main_dir + "/results/OriginalEEGan"):
        os.makedirs(main_dir + "/results/OriginalEEGan")
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

def dice_score(mask1, mask2):
    """    Calculates the Dice score between two masks.
    The Dice score is a measure of similarity between two sets, and is defined as
    the ratio of twice the intersection of the masks to the sum of the two masks.
    Used to measure the similarity between two segmented regions.
    Args:
        mask1 (numpy.ndarray):
        mask2 (numpy.ndarray):
    Returns:
        float: The Dice score between the two masks, ranging from 0 (no overlap)
            to 1 (perfect overlap).
    """
    # Check that the masks have the same dimensions
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same dimensions.")

    # Calculate the intersection and union of the masks
    intersection = np.logical_and(mask1, mask2).sum() #cv.bitwise_and(mask1,mask2).sum()
    union = mask1.sum() + mask2.sum()

    # Calculate the Dice score
    dice_score = 2 * intersection / union if union > 0 else 0

    return dice_score

evaluation ={}
for i in class_labels:
    class_data = history[i]
    ds_scores = []
    ssim_scores = []
    true_positives = 0

    save_imgs(class_data['mnist'], "Real", i)
    save_imgs(class_data['generated'], "Generated", i)

    for j in range(class_data['generated'].shape[0]):
        if i == np.argmax(class_data['predicted'][j]):
            true_positives += 1
        ds = dice_score(binarise(class_data['mnist'][j][:,:,0],0.5),binarise(class_data['generated'][j][:,:,0],0.5))
        #print(f"Dice score {ds} for class {np.argmax(label[i])}")
        ds_scores.append(ds)
        data_range = class_data['generated'][j][:,:,0].max() - class_data['generated'][j][:,:,0].min()
        ssim_value = ssim(class_data['mnist'][j][:,:,0],class_data['generated'][j][:,:,0], data_range=data_range)
        #print(f"SSIM score {ssim_value}")
        ssim_scores.append(ssim_value)
    evaluation[i] = {'average_ds':np.mean(ds_scores),'average_ssim':np.mean(ssim_scores),'average_validity':np.mean(class_data['valid'])}
    class_acc = true_positives / class_data['generated'].shape[0]
    print(f"Class {i}: mean ds: {evaluation[i]['average_ds']:.2f}, mean ssim: {evaluation[i]['average_ssim']:.2f}, mean validity: {evaluation[i]['average_validity']:.2f}, classification acc: {class_acc:.1%}")

pass