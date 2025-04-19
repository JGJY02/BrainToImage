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
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.eeggan import (build_EEGgan, build_discriminator,
                            build_generator, combine_loss_metrics,
                            sample_images, save_model)


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


print(os.getcwd())
eeg_latent_dim = 128
class_labels = [0,1,2,3,4,5,6,7,8,9]
valid_threshold = 0.5

generator = build_generator(eeg_latent_dim,1,len(class_labels))
#generator.load_weights("./Brain_to_image/EEGgan/saved_model/generator_final_model.h5")
discriminator = build_discriminator((28,28,1),len(class_labels))
#discriminator.load_weights("./bigan/saved_model/")
combined = build_EEGgan(eeg_latent_dim,len(class_labels),generator,discriminator)
combined.load_weights(f"./Brain_to_image/EEGgan/saved_model/combined_final_model.h5")

eeg_space = np.random.normal(0, 1, (1, eeg_latent_dim))
label = np.random.randint(0, len(class_labels), (1, 1))

gen_image = generator([eeg_space, label])

validity, label = combined([eeg_space, label])
predicted_label = class_labels[np.argmax(label)]
predicted_validity = validity > valid_threshold


fig, axs = plt.subplots(1,1)
axs.imshow(gen_image[0,:,:,0], cmap='gray')
axs.axis('off')
axs.set_title(f"Predicted image: {predicted_label}, {validity[0]} {predicted_validity}")
plt.show()



pass