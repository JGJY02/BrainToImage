import os
import pickle
import sys

import numpy as np
from keras.initializers import RandomUniform
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D, ZeroPadding2D, multiply, concatenate, Conv2DTranspose, LeakyReLU)
from keras.models import Model, Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.mog_layer import MoGLayer

"""
Title: DCGAN model design

Purpose:
    Testing design and build of DCGAN, Functional blocks for training
    DCGAN model. Call from training script.

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
def build_dc_generator(latent_dim,num_channels,num_classes, num_classes_type,activation="relu",final_activation="tanh",verbose=False):

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
def build_dc_discriminator(img_shape,num_classes,num_type_classes=2,leaky_alpha=0.2,dropout=0.25,bn_momentum=0.8,verbose=False):

    model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, input_shape=img_shape, padding="same", name="Dis_Block1_Conv2D"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ], name="Discriminator_block")

    input_img = Input(shape=img_shape, name="Dis_Input_Img")

    # Extract feature representation
    features = model(input_img)

    # Determine validity and label of the image
    validity = Dense(1, activation="sigmoid", name="Dis_Validity")(features)
    label = Dense(num_classes, activation="softmax", name="Dis_Class_Label")(features)
    label_type = Dense(num_type_classes, activation="softmax", name="Dis_Class_type_Label")(features)

    final_model = Model(input_img, [validity, label, label_type], name="Discriminator")

    if verbose:
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model

# Complete GAN model
def build_DCGgan(latent_dim, gen, dis, verbose=False):

    latent_space = Input(shape=(latent_dim,), name="EEGGAN_Input_space")
    label = Input(shape=(1,), dtype=np.float32, name="EEGGAN_Input_label")
    label_type = Input(shape=(1,), dtype=np.float32, name="EEGGAN_Input_label_type")

    generator_image = gen(inputs=[latent_space, label, label_type])
    dis.trainable = False
    #gen_img = Input(shape=(28,28,1), name="EEGGAN_Gen_Image")
    validity, class_label, class_label_type = dis(inputs=[generator_image])
    final_model = Model(inputs=[latent_space, label, label_type], outputs=[validity, class_label, class_label_type] , name="EEGGAN")

    if verbose:
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model


if __name__ == '__main__':
    #gen = build_generator(128,1,10,verbose=True)
    #gen = build_MoGMgenerator(128,1,10,verbose=True)
    gen = build_dc_generator(128,1,10,verbose=True)
    dis = build_dc_discriminator((28,28,1),10,verbose=True)

    gan = build_DCGgan(128, 10, gen, dis, verbose=True)