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
valid_threshold = 0.5



## load the eeg training data
dataset_dir = "2022Data"
run_id = "934thresh_"
eeg_dataset = f"{config.eeg_dataset_dir}/{config.eeg_dataset_pickle}"
print(f"Reading data file {eeg_dataset}")
eeg_data = pickle.load(open(eeg_dataset, 'rb'), encoding='bytes')
to_labels = np.argmax(eeg_data['y_test'],axis=1)  ## since eeg labels are in one-hot encoded format

## load the object trainig data
print(f" Loading Object images from {eeg_dataset}")
(x_test, y_test) = (eeg_data['x_test_img'], eeg_data['y_test'])
print(eeg_data['x_train_img'].shape)

print(x_test.shape)

for i in class_labels:
    total = np.sum(to_labels == i)
    print(f"For Class {i} we have {total} Instances")
    # selected_indices = np.random.choice(cls_indices, 400, replace=False)  # Sample 400 indices
    # X_selected.append(X[selected_indices])
    # y_selected.append(y[selected_indices])

#Mini batch size

# ## #############
# # Build EEG Gan
# ## #############
# prefix = "MoGM"
# model_dir = f"./brain_to_Image/EEGgan/EEG_saved_model/{prefix}/{prefix}_"

# if prefix == "MoGC":
#     generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))
# elif prefix == "MoGM":
#     generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
# elif prefix == "Basic":
#     generator = build_generator(eeg_latent_dim,1,len(class_labels))

# #generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
# #generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))

# generator.load_weights(f"{model_dir}EEGGan_generator_weights.h5")
# discriminator = build_discriminator((28,28,1),len(class_labels))
# combined = build_EEGgan(eeg_latent_dim,len(class_labels),generator,discriminator)
# combined.load_weights(f"{model_dir}EEGgan_combined_weights.h5")
# resume_run_id           = os.path.join("results", "042-pgan-mnist-cond-preset-v2-1gpu-fp32-GRAPH-HIST")        # Run ID or network pkl to resume training from, None = start from scratch.
# resume_snapshot         = 10754        # Snapshot index to resume training from, None = autodetect.
resume_run_id           = os.path.join("results", "059-pgan-objects-cond-preset-v2-1gpu-fp32-GRAPH-HIST")        # Run ID or network pkl to resume training from, None = start from scratch.
resume_snapshot         = 12000 #2104 # 4247        # Snapshot index to resume training from, None = autodetect.


# #load generator to ekras model
# print(generator)
# layer_names = ['images_out']
# generator_outputs = [generator.get_layer(layer_name).output for layer_name in layer_names]
# generator_model = Model(inputs=generator.input, outputs=generator_outputs)

## #############
# EEG Classifier/Encoder
## #############
classifier = LSTM_Classifier(eeg_data['x_test_eeg'].shape[1], eeg_data['x_test_eeg'].shape[2], len(class_labels), 256)

indexes = [i for i, char in enumerate(config.eeg_dataset_pickle) if char == '_']
runid = config.eeg_dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id

classifier_model_path = f"{config.classifier_dir}/{runid}/{config.classifier_name}/eeg_classifier_adm5_final.h5"
classifier.load_weights(classifier_model_path)
# we need to classifier encoded laten space as input to the EEGGan model
layer_names = ['EEG_feature_BN2','EEG_Class_Labels']
encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

  

## #############################################################
# Make prediction on random selected eeg from eeg_data['x_test']
## #############################################################

history = {}
total_images_to_use = -1 #put -1 for all items
for i in class_labels:  ## outer loop per class
    print("Current class label is : ", i)
    ## get all EEG data for class i
    matching_indices = np.where(to_labels == i)
    eeg_samples = eeg_data['x_test_eeg'][matching_indices[0]]
    # eeg_classes = y_test[matching_indices]
    #gt_labels = np.full(eeg_samples.shape[0],i,dtype=int)
    ## get enough object samples of class i to match eeg_samples
    object_images = eeg_data['x_test_img'][matching_indices[0]][:total_images_to_use]

    

    ## classify and enncode the EEG signals for input to GAN
    encoded_eegs, conditioning_labels = encoder_model.predict(eeg_samples,batch_size=32)
    # conditioning_labels = np.argmax(conditioning_labels,axis=1)
    # print(conditioning_labels.shape)


    # # conditioning_labels = eeg_classes
    # empty_labels = np.zeros([conditioning_labels.shape[0], 0], dtype=np.float32)
    # forced_conditioned_labels = np.zeros([conditioning_labels.shape[0], 10], dtype=np.float32)

    # for j in range(conditioning_labels.shape[0]):
    #     forced_conditioned_labels[j][i] = 1
    
    # print("Our encoded eegs shape is : ", encoded_eegs.shape)
    encoded_eegs = encoded_eegs[:total_images_to_use]
    # forced_conditioned_labels = forced_conditioned_labels[:total_images_to_use]
    conditioning_labels = conditioning_labels[:total_images_to_use]
    # print(f"The conditioning labels are: {conditioning_labels}")
    # print(forced_conditioned_labels)
    # print(conditioning_labels.shape)

    # embedded_labels = embedding_model.predict(conditioned_labels)
    # embedded_labels = embedded_labels.squeeze(axis=1)
    # latent_representation = encoded_eegs * embedded_labels

    # print(embedded_labels.shape)
    # print(encoded_eegs.shape)

    # generated_samples = generator.predict([encoded_eegs, conditioning_labels],batch_size=32)
    with tf.Graph().as_default(), tfutil.create_session(config.tf_config).as_default():
        with tf.compat.v1.device('/gpu:0'):
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            generator, discriminator, Gs = misc.load_pkl(network_pkl)

        
        # minibatch_size = np.clip(8192 // encoded_eegs.shape[1], 4, 256)
        minibatch_size = 4
        # print("The encoded eegs shape is :", encoded_eegs.shape)
        # print("The calculated minibatch size is :", minibatch_size)
        generated_samples = Gs.run(encoded_eegs, conditioning_labels, minibatch_size = minibatch_size)
        generated_samples = generated_samples*127.5 + 127.5
        generated_samples = np.clip(generated_samples,0,255)
        generated_samples = generated_samples.astype(np.uint8)
        

        validitys, labels = discriminator.run(generated_samples, minibatch_size = minibatch_size)
        # print(validitys)
        # print(labels)

    ## predict on GAN
    object_images = np.pad(object_images, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0) #pad images as progressiveGAN did so as well
    generated_samples = np.transpose(generated_samples, (0, 2, 3, 1))

    print("Shape of object images : ", object_images.shape)
    print("Shape of generated images : ",generated_samples.shape)
    # print(generated_samples.shape)
    # print(object_images.shape)


    ## collate results
    history[i] = {'generated':generated_samples,'object':object_images,'valid':validitys,'predicted':labels, 'conditioning':conditioning_labels}





def binarise(image, threshold=0):
    # Ensure the input is a numpy array
    image = np.array(image)
    # Reshape to (28, 28) if it's (28, 28, 1)
    if image.shape == (28, 28, 1):
        image = image.reshape(28, 28)
    # Rescale from [-1, 1] to [0, 1]
    image = image/255
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

def save_imgs(images, name, class_label, conditioning_labels, predicted_labels):
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
    output_path = main_dir + f'/results/{run_id}{resume_snapshot}_ProgGAN/{name}_class_{class_label}.png'
    if not os.path.exists(main_dir + f"/results/{run_id}{resume_snapshot}_ProgGAN"):
        os.makedirs(main_dir + f"/results/{run_id}{resume_snapshot}_ProgGAN")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

evaluation ={}
sample_img_per_class = []
list_of_labels = []

mean_ds_scores = []
mean_ssim_scores = []
mean_validity_scores = []
mean_accuracy_scores = []
num_of_sample_images = 10
for i in class_labels:
    class_data = history[i]
    ds_scores = []
    ssim_scores = []
    true_positives = 0

    conditioning_labels_array = np.argmax(class_data['conditioning'], axis = 1)
    predicted_labels_array = np.argmax(class_data['predicted'], axis = 1)

    save_imgs(class_data['object'], "Real", i, conditioning_labels_array, predicted_labels_array)
    save_imgs(class_data['generated'], "Generated", i ,conditioning_labels_array, predicted_labels_array)
    
    Index_of_relevant_classes = np.where(conditioning_labels_array == i)[0]
    sample_img_per_class.append(class_data['generated'][Index_of_relevant_classes[:num_of_sample_images]])

    labels_array = np.ones(num_of_sample_images)*i
    list_of_labels.append(labels_array)

    # Update FID metric
    # fid = FrechetInceptionDistance(feature=2048)
    # real_images_torch = convert_tf_to_torch(class_data['object'])  # From TensorFlow model
    # fake_images_torch = convert_tf_to_torch(class_data['generated'])  # From GAN
    # fid.update(real_images_torch, real=True)
    # fid.update(fake_images_torch, real=False)
    # fid_score = fid.compute()

    
    for j in range(class_data['generated'].shape[0]):
        if i == np.argmax(class_data['predicted'][j]):
            true_positives += 1
        ds = dice_score(binarise(class_data['object'][j][:,:,0],0.5),binarise(class_data['generated'][j][:,:,0],0.5))
        #print(f"Dice score {ds} for class {np.argmax(label[i])}")
        ds_scores.append(ds)
        data_range = class_data['generated'][j][:,:,0].max() - class_data['generated'][j][:,:,0].min()
        ssim_value = ssim(class_data['object'][j][:,:,0],class_data['generated'][j][:,:,0], data_range=data_range)
        #print(f"SSIM score {ssim_value}")
        ssim_scores.append(ssim_value)
    evaluation[i] = {'average_ds':np.mean(ds_scores),'average_ssim':np.mean(ssim_scores),'average_validity':np.mean(class_data['valid'])}
    class_acc = true_positives / class_data['generated'].shape[0]
    print(f"Class {i}: mean ds: {evaluation[i]['average_ds']:.2f}, mean ssim: {evaluation[i]['average_ssim']:.2f}, mean validity: {evaluation[i]['average_validity']:.2f}, classification acc: {class_acc:.1%}")

    mean_ds_scores.append(evaluation[i]['average_ds'])
    mean_ssim_scores.append(evaluation[i]['average_ssim'])
    mean_validity_scores.append(evaluation[i]['average_validity'])
    mean_accuracy_scores.append(class_acc)

mean_evaluation = {'average_ds':np.mean(mean_ds_scores),'average_ssim':np.mean(mean_ssim_scores),'average_validity':np.mean(mean_validity_scores), 'average_accuracy':np.mean(mean_accuracy_scores)}
print(f"Average Class Results: mean ds: {mean_evaluation['average_ds']:.2f}, mean ssim: {mean_evaluation['average_ssim']:.2f}, mean validity: {mean_evaluation['average_validity']:.2f}, classification acc: {mean_evaluation['average_accuracy']:.1%}")
stacked_images = np.stack(sample_img_per_class, axis=0)
stacked_labels = np.stack(list_of_labels, axis = 0)

stacked_images = stacked_images.reshape(-1, 32, 32, 3)
stacked_labels = stacked_labels.reshape(-1)
save_imgs(stacked_images, "Sampling image of each class", "all" ,stacked_labels, stacked_labels)
pass