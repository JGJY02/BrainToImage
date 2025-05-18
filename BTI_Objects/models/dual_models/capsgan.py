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
from models.capsnet import *
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
def build_dccaps_generator(latent_dim,num_channels,num_classes,activation="relu",final_activation="tanh",verbose=False):

    model = Sequential([
        Dense(128 * 7 * 7, input_dim=latent_dim, name="Gen_Dense_1"), #Original is 8,8 but that gives a 64 image at the end to decrease to change size
        Reshape((7, 7, 128), name="Reshape"),
        Conv2DTranspose(latent_dim, kernel_size=4,strides = 2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(256, kernel_size=4,strides = 2, padding="same"),
        LeakyReLU(alpha=0.2),
        # Conv2DTranspose(512, kernel_size=4,strides = 2, padding="same"),
        # LeakyReLU(alpha=0.2),
        Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ], name="Generator_block")

    latent_space = Input(shape=(latent_dim,), name="Gen_Input_space")
    label = Input(shape=(1,), dtype='int32', name="Gen_Input_label")
    label_embedding = Flatten(name="Gen_Flatten")(Embedding(num_classes, latent_dim, name="Gen_Embed")(label))
    # label = Input(shape=(num_classes,), dtype=np.float32, name="Input_label")
    # label_embedding = Dense(latent_dim)(label)
    model_input = multiply([latent_space, label_embedding],name="Gen_Mul")
    gen_img = model(model_input)

    final_model = Model([latent_space, label], gen_img, name="Generator")

    if verbose:
        #model.summary()
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model



#Discriminator model
def build_caps_discriminator(input_shape, n_class, routings = 3):
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
    # print("n_class in put is, ", n_class)
    # print("y shape when initializing", y.shape)
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
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



    val_validity = Dense(1, activation="sigmoid", name="Dis_Validity")(val_decoder)
    val_label = Dense(n_class, activation="softmax", name="Dis_Class_Label")(val_decoder)

    # manipul_validity = Dense(1, activation="sigmoid", name="Dis_Validity")(manipulate_decoder)
    # manipul_label = Dense(n_class, activation="softmax", name="Dis_Class_Label")(val_decoder)
    
    # Models for training and evaluation (prediction)
    train_model = Model([input_image, y], [validity, label], name="Discriminator")


    # print("Input of dis validity size", train_model.get_layer("Dis_Validity").input_shape)
    # print("Output of dis validity size",train_model.get_layer("Dis_Validity").output_shape) 

    eval_model = Model(input_image, [val_validity, val_label], name="Discriminator")

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
    masking_label = Input(shape=(num_classes,), name="EEG_CAPSGAN_Input_Masking")

    generator_image = gen(inputs=[latent_space, label])
    dis.trainable = False
    #gen_img = Input(shape=(28,28,1), name="EEGGAN_Gen_Image")
    validity, class_label = dis(inputs=[generator_image, masking_label])
    final_model = Model(inputs=[latent_space, label, masking_label], outputs=[validity,class_label] , name="EEG_CAPSGAN")

    if verbose:
        final_model.summary(show_trainable=True,expand_nested=True)

    return final_model


# #Discriminator model

# ##Conv Layer
# def convlayer(img_shape, out_channels=256, kernel_size=9):
#     input_image = Input(shape = img_shape)
#     conv_output = Conv2D(out_channels, activation='relu', kernel_size=kernel_size, strides=1, name="Dis_Block1_Conv2D")(input_image)

#     return Model(input_image, conv_output, name="ConvLayer")

# ##Primary Caps
# def squash(input_tensor):
#     squared_norm = (input_tensor ** 2).sum(-1,keepdim=True)
#     output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * tf.sqrt(squared_norm)) 
#     return output_tensor

# def primaryCaps(convLayerShape, caps_size = 8, out_channels=32, kernel_size=9):
#     capsules = [Conv2D(out_channels, kernel_size=kernel_size, strides=2) for _ in range(caps_size)]
#     num_of_capsules=32*6*6 #figure out how to dynamically adjust this
#     input_image = Input(shape = convLayerShape)
#     conv_output = [capsule(input_image) for capsule in capsules]
#     conv_output = tf.stack(conv_output, dim=1)
#     conv_output = tf.reshape(conv_output, [convLayerShape[0], num_of_capsules, caps_size])
#     conv_output = squash(conv_output)

#     return Model(input_image, conv_output, name="primaryCaps")

# ## Digit Caps

# def digitCaps(convLayerShape, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
#     batch_size = convLayerShape[0]

#     x = tf.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

#     W = torch.cat([self.W] * batch_size, dim=0)
#     u_hat = torch.matmul(W, x)

#     b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
#     if USE_CUDA:
#         b_ij = b_ij.cuda()

#     num_iterations = 3
#     for iteration in range(num_iterations):
#         c_ij = F.softmax(b_ij)
#         c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
#         if USE_CUDA:
#             c_ij = c_ij.cuda()
#         s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
#         v_j = self.squash(s_j)
        
#         if iteration < num_iterations - 1:
#             a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
#             b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

#         return v_j.squeeze(1)


# def build_dc_discriminator(img_shape,num_classes,leaky_alpha=0.2,dropout=0.25,bn_momentum=0.8,verbose=False):

#     model = Sequential([
#         Conv2D(256, kernel_size=9, strides=1, input_shape=img_shape, name="Dis_Block1_Conv2D"),
#         ##Number of capsules
#         Conv2D(32, kernel_size=9, strides=2, padding=0, name="Dis_PrimaryCaps"),
#         LeakyReLU(alpha=0.2),
#         Conv2D(128, kernel_size=4, strides=2, padding="same"),
#         LeakyReLU(alpha=0.2),
#         Flatten(),
#         Dropout(0.2),
#         Dense(1, activation="sigmoid")
#     ], name="Discriminator_block")

#     input_img = Input(shape=img_shape, name="Dis_Input_Img")

#     # Extract feature representation
#     features = model(input_img)

#     # Determine validity and label of the image
#     validity = Dense(1, activation="sigmoid", name="Dis_Validity")(features)
#     label = Dense(num_classes, activation="softmax", name="Dis_Class_Label")(features)

#     final_model = Model(input_img, [validity, label], name="Discriminator")

#     if verbose:
#         final_model.summary(show_trainable=True,expand_nested=True)

#     return final_model

# # Complete GAN model
# def build_CapsGan(latent_dim, num_classes, gen, dis, verbose=False):

#     latent_space = Input(shape=(latent_dim,), name="EEGGAN_Input_space")
#     label = Input(shape=(1,), dtype=np.float32, name="EEGGAN_Input_label")
#     generator_image = gen(inputs=[latent_space, label])
#     dis.trainable = False
#     #gen_img = Input(shape=(28,28,1), name="EEGGAN_Gen_Image")
#     validity, class_label = dis(inputs=[generator_image])
#     final_model = Model(inputs=[latent_space,label], outputs=[validity,class_label] , name="EEGGAN")

#     if verbose:
#         final_model.summary(show_trainable=True,expand_nested=True)

#     return final_model


# if __name__ == '__main__':
#     #gen = build_generator(128,1,10,verbose=True)
#     #gen = build_MoGMgenerator(128,1,10,verbose=True)
#     gen = build_dc_generator(128,1,10,verbose=True)
#     dis = build_dc_discriminator((28,28,1),10,verbose=True)

#     gan = build_CapsGgan(128, 10, gen, dis, verbose=True)