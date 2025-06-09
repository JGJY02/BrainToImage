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


#Generator model, with modulation and multiplication
def build_MoGMgenerator(latent_dim,num_channels,num_classes, num_classes_type,activation="relu",final_activation="tanh",verbose=False):

    model = Sequential([
        Dense(128 * 7 * 7, activation=activation,  name="Gen_Dense_1"),  # input_dim=latent_dim,
        Reshape((7, 7, 128), name="Reshape"),
        BatchNormalization(momentum=0.8, name="Gen_Block1_BN"),
        UpSampling2D(name="Gen_Block1_UpSample"),
        Conv2D(128, kernel_size=3, activation=activation, padding="same", name="Gen_Block1_Conv2D"),
        BatchNormalization(momentum=0.8, name="Gen_Block2_BN"),
        UpSampling2D(name="Gen_Block2_UpSample"),
        Conv2D(64, kernel_size=3, activation=activation, padding="same", name="Gen_Block2_Conv2D"),
        BatchNormalization(momentum=0.8, name="Gen_Block3_BN"),
        Conv2D(num_channels, kernel_size=3, activation=final_activation, padding='same', name="Gen_Block3_Conv2D")
    ], name="Generator_block")

    latent_space = Input(shape=(latent_dim,), name="Gen_Input_space")
    mog_layer = MoGLayer(kernel_initializer=RandomUniform(minval=-0.2, maxval=0.2),
                 bias_initializer=RandomUniform(minval=-1.0, maxval=1.0), kernel_regularizer=l2(0.01), name="Gen_MoG")(latent_space)
    label = Input(shape=(1,), dtype='int32', name="Gen_Input_label")
    label_type = Input(shape=(1,), dtype='int32', name="Gen_Input_label_type")

    label_embedding = Flatten(name="Gen_Flatten")(Embedding(num_classes, latent_dim, name="Gen_Embed")(label))
    label_embedding_type = Flatten(name="Gen_type_Flatten")(Embedding(num_classes_type, latent_dim, name="Gen_type_Embed")(label_type))


    model_input = multiply([mog_layer, label_embedding],name="Gen_Mul")
    model_input = multiply([model_input, label_embedding_type],name="Gen_Mul_Type")

    #model_input = concatenate([mog_layer, label_embedding],name="Gen_Mul")
    gen_img = model(model_input)

    final_model = Model([latent_space, label, label_type], gen_img, name="Generator")

    if verbose:
        #model.summary()
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

#Generator model, with modulation and concatination
def build_MoGCgenerator(latent_dim,num_channels,num_classes,num_classes_type ,activation="relu",final_activation="tanh",verbose=False):

    model = Sequential([
        Dense(128 * 7 * 7, activation=activation,  name="Gen_Dense_1"),  # input_dim=latent_dim,
        Reshape((7, 7, 128), name="Reshape"),
        BatchNormalization(momentum=0.8, name="Gen_Block1_BN"),
        UpSampling2D(name="Gen_Block1_UpSample"),
        Conv2D(128, kernel_size=3, activation=activation, padding="same", name="Gen_Block1_Conv2D"),
        BatchNormalization(momentum=0.8, name="Gen_Block2_BN"),
        UpSampling2D(name="Gen_Block2_UpSample"),
        Conv2D(64, kernel_size=3, activation=activation, padding="same", name="Gen_Block2_Conv2D"),
        BatchNormalization(momentum=0.8, name="Gen_Block3_BN"),
        Conv2D(num_channels, kernel_size=3, activation=final_activation, padding='same', name="Gen_Block3_Conv2D")
    ], name="Generator_block")

    latent_space = Input(shape=(latent_dim,), name="Gen_Input_space")
    mog_layer = MoGLayer(kernel_initializer=RandomUniform(minval=-0.2, maxval=0.2),
                 bias_initializer=RandomUniform(minval=-1.0, maxval=1.0), kernel_regularizer=l2(0.01), name="Gen_MoG")(latent_space)
    label = Input(shape=(1,), dtype='int32', name="Gen_Input_label")
    label_type = Input(shape=(1,), dtype='int32', name="Gen_Input_label_type")

    label_embedding = Flatten(name="Gen_Flatten")(Embedding(num_classes, latent_dim, name="Gen_Embed")(label))
    label_embedding_type = Flatten(name="Gen_type_Flatten")(Embedding(num_classes_type, latent_dim, name="Gen_type_Embed")(label_type))

    model_input = concatenate([mog_layer, label_embedding],name="Gen_Mul")
    model_input = concatenate([model_input, label_embedding_type],name="Gen_Mul_Type")

    gen_img = model(model_input)

    final_model = Model([latent_space, label, label_type], gen_img, name="Generator")

    if verbose:
        #model.summary()
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

#Generator model
def build_generator(latent_dim,num_channels,num_classes, num_classes_type,activation="relu",final_activation="tanh",verbose=False):

    model = Sequential([
        Dense(512 * 7 * 7, activation=activation, input_dim=latent_dim, name="Gen_Dense_1"),
        Reshape((7, 7, 512), name="Reshape"),
        BatchNormalization(momentum=0.8, name="Gen_Block1_BN"),
        UpSampling2D(name="Gen_Block1_UpSample"),
        Conv2D(128, kernel_size=3, activation=activation, padding="same", name="Gen_Block1_Conv2D"),
        BatchNormalization(momentum=0.8, name="Gen_Block2_BN"),
        UpSampling2D(name="Gen_Block2_UpSample"),
        Conv2D(64, kernel_size=3, activation=activation, padding="same", name="Gen_Block2_Conv2D"),
        BatchNormalization(momentum=0.8, name="Gen_Block3_BN"),
        Conv2D(num_channels, kernel_size=3, activation=final_activation, padding='same', name="Gen_Block3_Conv2D")
    ], name="Generator_block")

    latent_space = Input(shape=(latent_dim,), name="Gen_Input_space")
    label = Input(shape=(1,), dtype='int32', name="Gen_Input_label")
    label_type = Input(shape=(1,), dtype='int32', name="Gen_Input_label_type")

    label_embedding = Flatten(name="Gen_Flatten")(Embedding(num_classes, latent_dim, name="Gen_Embed")(label))
    label_embedding_type = Flatten(name="Gen_type_Flatten")(Embedding(num_classes_type, latent_dim, name="Gen_type_Embed")(label_type))


    model_input = multiply([latent_space, label_embedding],name="Gen_Mul")
    model_input_comp = multiply([model_input, label_embedding_type],name="Gen_Mul_Type")

    #model_input = concatenate([mog_layer, label_embedding],name="Gen_Mul")
    gen_img = model(model_input_comp)

    final_model = Model([latent_space, label, label_type], gen_img, name="Generator")
    if verbose:
        #model.summary()
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

#Discriminator model
def build_discriminator(img_shape,num_classes, num_classes_type,leaky_alpha=0.2,dropout=0.25,bn_momentum=0.8,verbose=False):

    model = Sequential([
        Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same", name="Dis_Block1_Conv2D"),
        LeakyReLU(alpha=leaky_alpha, name="Dis_Block1_LRelu"),
        Dropout(dropout, name="Dis_Block1_Dropout"),
        Conv2D(32, kernel_size=3, strides=2, padding="same", name="Dis_Block2_Conv2D"),
        ZeroPadding2D(padding=((0,1),(0,1)), name="Dis_Block2_ZeroPad"),
        LeakyReLU(alpha=leaky_alpha, name="Dis_Block2_LRelu"),
        Dropout(dropout, name="Dis_Block2_Dropout"),
        BatchNormalization(momentum=bn_momentum, name="Dis_Block2_BN"),
        Conv2D(64, kernel_size=3, strides=2, padding="same", name="Dis_Block3_Conv2D"),
        LeakyReLU(alpha=leaky_alpha, name="Dis_Block3_LRelu"),
        Dropout(dropout, name="Dis_Block3_Dropout"),
        BatchNormalization(momentum=bn_momentum, name="Dis_Block3_BN"),
        Conv2D(128, kernel_size=3, strides=1, padding="same", name="Dis_Block4_Conv2D"),
        LeakyReLU(alpha=leaky_alpha, name="Dis_Block4_LRelu"),
        Dropout(dropout, name="Dis_Block2_Drop"),
        Flatten(name="Dis_logits")
    ], name="Discriminator_block")

    input_img = Input(shape=img_shape, name="Dis_Input_Img")

    # Extract feature representation
    features = model(input_img)

    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid", name="Dis_Validity")(features)
    
    label = Dense(num_classes, activation="softmax", name="Dis_Class_Label")(features)
    label_type = Dense(num_classes_type, activation="softmax", name="Dis_Class_type_Label")(features)

    final_model = Model(input_img, [validity, label, label_type], name="Discriminator")

    if verbose:
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

# Complete GAN model
def build_EEGgan(latent_dim, gen, dis, verbose=False):

    latent_space = Input(shape=(latent_dim,), name="EEGGAN_Input_space")
    label = Input(shape=(1,), dtype=np.float32, name="EEGGAN_Input_label")
    label_type = Input(shape=(1,), dtype=np.float32, name="EEGGAN_Input_label_type")

    generator_image = gen(inputs=[latent_space, label, label_type])
    dis.trainable = False
    #gen_img = Input(shape=(28,28,1), name="EEGGAN_Gen_Image")
    validity, class_label, class_label_type = dis(inputs=[generator_image])
    final_model = Model(inputs=[latent_space,label, label_type], outputs=[validity,class_label, class_label_type] , name="EEGGAN")

    if verbose:
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

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
    valid = labels[0] == labels[1]
    fig, axs = plt.subplots(r, c)
    fig.suptitle(f"Generated images for Epoch {epoch}",size=10)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0])
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

if __name__ == '__main__':
    #gen = build_generator(128,1,10,verbose=True)
    #gen = build_MoGMgenerator(128,1,10,verbose=True)
    gen = build_MoGCgenerator(128,1,10,verbose=True)
    dis = build_discriminator((28,28,1),10,verbose=True)

    gan = build_EEGgan(128, 10, gen, dis, verbose=True)