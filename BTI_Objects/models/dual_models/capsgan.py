import os
import pickle
import sys

import numpy as np
from keras.initializers import RandomUniform
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D, ZeroPadding2D, multiply, concatenate, Conv2DTranspose, LeakyReLU, Add)
from keras.models import Model, Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.mog_layer import MoGLayer
from models.dual_models.capsnet import *
import tensorflow as tf

"""
Title: CAPSGan model design

Purpose:
    Testing design and build of CAPSGan, Functional blocks for training
    CAPSGan model. Call from training script.

Author: Jared
Date: 01/07/2024
Version: <Version number>

Usage:
    build ACGAN model

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""


#Generator model
def build_dccaps_generator(latent_dim,num_channels,num_classes, num_classes_type,activation="relu",final_activation="tanh",verbose=False):

    model = Sequential([
        Dense(128 * 8 * 8, input_dim=latent_dim, name="Gen_Dense_1"),
        Reshape((8, 8, 128), name="Reshape"),
        Conv2DTranspose(latent_dim, kernel_size=4,strides = 2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(256, kernel_size=4,strides = 2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(512, kernel_size=4,strides = 2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ], name="Generator_block")

    latent_space = Input(shape=(latent_dim,), name="Gen_Input_space")
    label = Input(shape=(1,), dtype='int32', name="Gen_Input_label")
    label_type = Input(shape=(1,), dtype='int32', name="Gen_Input_label_type")

    label_embedding = Flatten(name="Gen_Flatten")(Embedding(num_classes, latent_dim, name="Gen_Embed")(label))
    label_embedding_type = Flatten(name="Gen_Flatten_type")(Embedding(num_classes_type, latent_dim, name="Gen_Embed_type")(label_type))

    # label = Input(shape=(num_classes,), dtype=np.float32, name="Input_label")
    # label_embedding = Dense(latent_dim)(label)
    model_input = multiply([latent_space, label_embedding],name="Gen_Mul")
    model_input = multiply([model_input, label_embedding_type],name="Gen_Mul_type")

    gen_img = model(model_input)

    final_model = Model([latent_space, label, label_type], gen_img, name="Generator")

    if verbose:
        #model.summary()
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model




#Discriminator model
def build_caps_discriminator(input_shape, n_class, n_class_type, routings = 3):
    input_image = Input(shape=input_shape)
    # print("Shape of the input is ", input_shape)
    conv_output = Conv2D(256, activation='relu', kernel_size=9, strides=1, padding='valid', name="Dis_Block1_Conv2D")(input_image)
    primarycaps = PrimaryCap(conv_output, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    # print("shape of primarycaps is ", primarycaps.shape)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)
    # print("shape of the digitcaps is ,", digitcaps.shape)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = Input(shape=(n_class,))
    # y_type = Input(shape=(n_class_type,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    # masked_by_y_type = Mask()([digitcaps_type, y_type])  # The true label is used to mask the output of capsule layer. For training

    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction

    model = Sequential([
        Dense(512, activation='relu', input_dim=16*n_class),
        Dense(1024, activation='relu')
    ], name = 'decoder')
    # decoder = Sequential(name='decoder')
    # decoder.add(Dense(512, activation='relu', input_dim=16*n_class))
    # decoder.add(Dense(1024, activation='relu'))
    # decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
    # decoder.add(Reshape(target_shape=input_shape, name='out_recon'))

    # Determine validity and label of the image
    train_decoder = model(masked_by_y) #The masked input is causing an issue currently so we are using the prediction mask instead
    # print("Shaope of the train decoder is: ", train_decoder.shape)
    val_decoder = model(masked)
    # manipulate_decoder = decoder(masked_noised_y)


    validity = Dense(1, activation="sigmoid", name="Dis_Validity")(train_decoder)
    label = Dense(n_class, activation="softmax", name="Dis_Class_Label")(train_decoder)
    label_type = Dense(n_class_type, activation="softmax", name="Dis_Class_type_Label")(train_decoder)



    val_validity = Dense(1, activation="sigmoid", name="Dis_Validity")(val_decoder)
    val_label = Dense(n_class, activation="softmax", name="Dis_Class_Label")(val_decoder)
    val_label_type = Dense(n_class_type, activation="softmax", name="Dis_Class_type_Label")(val_decoder)

    # manipul_validity = Dense(1, activation="sigmoid", name="Dis_Validity")(manipulate_decoder)
    # manipul_label = Dense(n_class, activation="softmax", name="Dis_Class_Label")(val_decoder)
    
    # Models for training and evaluation (prediction)
    print(y.shape)
    train_model = Model([input_image, y], [validity, label, label_type], name="Discriminator")


    # print("Input of dis validity size", train_model.get_layer("Dis_Validity").input_shape)
    # print("Output of dis validity size",train_model.get_layer("Dis_Validity").output_shape) 

    eval_model = Model(input_image, [val_validity, val_label, val_label_type], name="Discriminator")

    # manipulate model
    noise = Input(shape=(n_class, 16))
    noised_digitcaps = Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = Model([input_image, y, noise], model(masked_noised_y), name="Discriminator")
    return train_model, eval_model, manipulate_model


# Complete GAN model
def build_capsGAN(latent_dim, num_classes, gen, dis, verbose=False):

    latent_space = Input(shape=(latent_dim,), name="EEG_CAPSGAN_Input_space")
    label = Input(shape=(1,), dtype=np.float32, name="EEG_CAPSGAN_Input_label")
    label_type = Input(shape=(1,), dtype=np.float32, name="EEG_CAPSGAN_Input_label_type")

    masking_label = Input(shape=(num_classes,), name="EEG_CAPSGAN_Input_Masking")


    generator_image = gen(inputs=[latent_space, label, label_type])
    dis.trainable = False
    #gen_img = Input(shape=(28,28,1), name="EEGGAN_Gen_Image")
    validity, class_label, class_label_type = dis(inputs=[generator_image, masking_label])
    final_model = Model(inputs=[latent_space, label, label_type, masking_label], outputs=[validity, class_label, class_label_type] , name="EEG_CAPSGAN")

    if verbose:
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

