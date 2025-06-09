import os
import pickle
import sys

import numpy as np
from keras.initializers import RandomUniform
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D, ZeroPadding2D, multiply, concatenate)
from keras.models import Model, Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.mog_layer import MoGLayer

"""
Title: EEG ACGAN model design

Purpose:
    Testing design and build of EEG ACGAN, Functional blocks for training
    ACGAN model. Call from training script.

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    build ACGAN model

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""




def sample_images(epoch, generator, latent_dim, latent_space, labels, main_dir):
    r, c = 10, 10
    #noise = np.random.normal(0, 1, (r * c, latent_dim))
    #sampled_labels = np.array([num for _ in range(r) for num in range(c)])
    #gen_imgs = generator.predict([noise, sampled_labels])
    gen_imgs = generator.predict([latent_space, labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    
    dir_to_save = "./brain_to_image/EEGgan/EEG_images/"
    save_path = os.path.join(main_dir, dir_to_save)

    if not os.path.exists(save_path): #Jared Edition
        os.makedirs(save_path)


    fig.savefig(os.path.join(save_path, f"EEGGan_{epoch:.1f}.png"))
    plt.close()

def sample_images_eeg(epoch, gen_imgs, labels, main_dir, dir_to_save):
    r, c = 4, 8
    gen_imgs = (gen_imgs * 127.5) + 127.5
    gen_imgs = np.array(gen_imgs).astype(int)
    valid = labels[0] == labels[1]
    fig, axs = plt.subplots(r, c)
    fig.suptitle(f"Generated images for Epoch {epoch}",size=10)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            axs[i,j].set_title(f"{valid[cnt]} S:{labels[0][cnt]} P:{labels[1][cnt]}",size=5)
            cnt += 1

    save_path = os.path.join(main_dir, dir_to_save)

    if not os.path.exists(save_path): #Jared Edition
        os.makedirs(save_path)


    fig.savefig(os.path.join(save_path, f"EEGGan_{epoch:.1f}.png"))
    plt.close()


def save_model(model, model_type, cur_run, model_name, main_dir, output_dir):

    def save(model, model_name, main_dir, output_dir):
        path = f"{main_dir}/{output_dir}/{model_type}/hist"
        model_path = f"{path}/{model_name}.json"
        weights_path = f"{path}/{model_name}_weights.hdf5"
        model_file = f"{path}/{model_name}_model.hdf5"
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")
        options = {"file_arch": model_path,
                    "file_weight": weights_path,
                    "file_model": model_file}
        json_string = model.to_json()
        with open(options['file_arch'], 'w') as f:
            f.write(json_string)
        model.save_weights(options['file_weight'])
        #model.save(options["file_model"])

    save(model, model_name, main_dir, output_dir)
    #save(discriminator, "discriminator")

def combine_loss_metrics(d_loss_real, d_loss_fake):
    # Initialize a new dictionary to store the combined results
    d_loss_combined = {}

    # Iterate through the keys of the first dictionary (assumes both have the same keys)
    for key in d_loss_real:
        # Compute the average of the values for the corresponding key
        d_loss_combined[key] = 0.5 * np.add(d_loss_real[key], d_loss_fake[key])

    return d_loss_combined

