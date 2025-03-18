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

print(os.getcwd())
eeg_latent_dim = 128
class_labels = [0,1,2,3,4,5,6,7,8,9]
valid_threshold = 0.5

## load the MNIST trainig data
image_data_dir = "Datasets\MNIST_dataset"
print(f" Loading MNIST images from {image_data_dir}")
_ , (x_test, y_test) = load_local_mnist(image_data_dir,norm=-1,sparse=True)
## make a slection of 10 images from mnist testing set.
mnist_images = []
for i in range(10):
    matching_indices = np.where(y_test == i)
    chosen_index = np.random.choice(matching_indices[0])
    mnist_images.append(x_test[chosen_index])
## load the eeg training data
dataset = "MNIST_EP"
eeg_data_dir = f"Datasets/MindBigData MNIST of Brain Digits/{dataset}"
data_file = "data9032_train_MindBigData2022_MNIST_EP.pkl"
print(f"Reading data file {eeg_data_dir}/{data_file}")
eeg_data = pickle.load(open(f"{eeg_data_dir}/{data_file}", 'rb'), encoding='bytes')
### make a list of 10 EEG data from testing data set
###
eeg_samples = []
for i in range(10):
    to_labels = np.argmax(eeg_data['y_test'],axis=1)
    matching_indices = np.where(to_labels == i)
    chosen_index = np.random.choice(matching_indices[0])
    eeg_samples.append(eeg_data['x_test'][chosen_index])
eeg_samples = np.array(eeg_samples)
sampled_labels = class_labels
### Sample EEG to use as input to EEG Classifier/Encoder to provide latent space and conditioning label as generator input
### single sampling
# sample_indexs = np.random.choice(eeg_data['x_test'].shape[0], size=1, replace=False)
# eeg_samples = eeg_data['x_test'][sample_indexs]
# # The labels of the digits that the generator tries to create an
# # image representation of, this is for evalution of accuracy.
# sampled_labels = class_labels[np.argmax(eeg_data['y_test'][sample_indexs])]


## #############
# Build EEG Gan
## #############
prefix = "NoMoG"
model_dir = f"./Brain_to_Image/EEGgan/{prefix}_EEG_saved_model/{prefix}_"
generator = build_generator(eeg_latent_dim,1,len(class_labels))
#generator = build_MoGMgenerator(eeg_latent_dim,1,len(class_labels))
#generator = build_MoGCgenerator(eeg_latent_dim,1,len(class_labels))
generator.load_weights(f"{model_dir}EEGGan_generator_weights.h5")
discriminator = build_discriminator((28,28,1),len(class_labels))
combined = build_EEGgan(eeg_latent_dim,len(class_labels),generator,discriminator)
combined.load_weights(f"{model_dir}EEGgan_combined_weights.h5")
# we want to extract the Generator image created inside the Gan
# gan_layer_names = ["Generator","Discriminator"]
# EEGgan_outputs = [combined.get_layer(layer_name).output for layer_name in gan_layer_names]
# EEGgan_model = Model(inputs=combined.inputs, outputs=EEGgan_outputs)
# EEGgan_model.summary()
## #############
# EEG Classifier/Encoder
## #############
classifier = convolutional_encoder_model(eeg_data['x_train'].shape[1], eeg_data['x_train'].shape[2], len(class_labels))

classifier_model_path = f"Datasets\MindBigData MNIST of Brain Digits\MNIST_EP\models\eeg_classifier_adm_final.h5"
classifier.load_weights(classifier_model_path)
# we need to classifier encoded laten space as input to the EEGGan model
layer_names = ['EEG_feature_BN2','EEG_Class_Labels']
encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

## #############################################################
# Make prediction on random selected eeg from eeg_data['x_test']
## #############################################################

encoded_eeg, conditioning_label = encoder_model.predict(eeg_samples)
conditioning_label = np.argmax(conditioning_label,axis=1)
generated_sample = generator.predict([encoded_eeg, conditioning_label])
#prediced_image = np.expand_dims(generated_sample[0,:,:,0].numpy(), axis=-1)
validity, label = combined.predict([encoded_eeg, conditioning_label])
for i in range(10):
    print(f"Class label: {class_labels[i]} validity:{validity[i]}, predicted label: {np.argmax(label[i])}")
#predicted_label = class_labels[np.argmax(label)]
#predicted_validity = validity[0]

#mnist_image = sample_image_by_label(x_test, y_test, sampled_labels)

# filtered_mnist_labels = y_train[np.isin(y_train, sampled_labels)]
# filtered_mnist_idx = np.random.choice(np.where(filtered_mnist_labels)[0], size=1, replace=False)
# # Image labels. 0-9
# img_labels = filtered_mnist_labels[np.argmax(y_train[filtered_mnist_idx],axis=1)]
# # Images
# mnist_image = x_train[filtered_mnist_idx]

mnist_images = []
for i in range(10):
    matching_indices = np.where(y_test == i)
    chosen_index = np.random.choice(matching_indices[0])
    mnist_images.append(x_test[chosen_index])
mnist_images = np.array(mnist_images)

# fig, axs = plt.subplots(1,2)
# fig.suptitle("MNIST Image vs generated image from EEG signal",size=10)
# axs[0].imshow(mnist_image[:,:,0], cmap='gray')
# axs[1].imshow(prediced_image[:,:,0], cmap='gray')
# axs[0].axis('off')
# axs[1].axis('off')
# axs[0].set_title(f"MNIST image: {sampled_labels}",size=8)
# axs[1].set_title(f"Predicted image: {sampled_labels}, {predicted_label}, Conf :{predicted_validity}",size=8)
# fig.tight_layout()
# plt.show()

for i, sample in enumerate(generated_sample):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(generated_sample[i,:,:,0],'gray')
    ax.axis('off')
    fig.tight_layout()
    plt.savefig(f"./{prefix}/{prefix}_generated_{i}_sample.png",bbox_inches='tight')
    plt.show()

for i, sample in enumerate(mnist_images):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(mnist_images[i,:,:,0],'gray')
    ax.axis('off')
    fig.tight_layout()
    plt.savefig(f"./{prefix}/{prefix}_mnist_{i}_sample.png",bbox_inches='tight')
    plt.show()

## montage of samples
# fig, axs = plt.subplots(2, 5, figsize=(12, 8))
# fig.suptitle("MoGM GAN Generated Images")
# for i, ax in enumerate(axs.flat):
#     ax.imshow(generated_sample[i,:,:,0],'gray')
#     ax.set_title(f"{sampled_labels[i] == np.argmax(label[i])} : {validity[i][0]:.3f}:{np.argmax(label[i])}")
#     ax.axis('off')
# fig.tight_layout()
# plt.show()

# fig, axs = plt.subplots(2, 5, figsize=(12, 8))
# fig.suptitle("NMIST Image samples")
# for i, ax in enumerate(axs.flat):
#     ax.imshow(mnist_images[i,:,:,0],'gray')
#     ax.set_title(f"Sample vs predicted {sampled_labels[i] == np.argmax(label[i])}")
#     ax.axis('off')
# fig.tight_layout()
# plt.show()

exit()

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


for i in range(10):
    ds = dice_score(binarise(mnist_images[i][:,:,0],0.5),binarise(generated_sample[i][:,:,0],0.5))
    print(f"Dice score {ds} for class {np.argmax(label[i])}")
    data_range = generated_sample[i][:,:,0].max() - generated_sample[i][:,:,0].min()
    ssim_value = ssim(mnist_images[i][:,:,0],generated_sample[i][:,:,0], data_range=data_range)
    print(f"SSIM score {ssim_value}")

pass