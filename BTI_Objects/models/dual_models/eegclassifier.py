from keras.layers import (BatchNormalization, Conv2D, Conv3D,Dense,
                            Dropout, Flatten, MaxPooling2D, UpSampling2D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, LSTM, RepeatVector, TimeDistributed,Reshape, Input, Lambda)
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt
from keras import backend as K

"""
Title: EEG Classifier model design

Purpose:
    Testing design and build of EEG Classifier, Functional blocks for training
    CNN Classifier model. Call from training script.

Author: Tim Tanner
Date: 01/07/2024
Version: <Version number>

Usage:
    build CNN model

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""

def convolutional_encoder_model_512_dual(channels, observations, num_classes, num_types = 2, verbose=False):


    model = Sequential([
    BatchNormalization(input_shape=(channels, observations, 1), name="EEG_BN1"),
    Conv2D(128, (1, 4), activation='relu', padding='same', name="EEG_series_Conv2D"),
    Conv2D(64, (channels, 1), activation='relu',padding='same', name="EEG_channel_Conv2D"),
    MaxPooling2D((1, 2), name="EEG_feature_pool1"),
    Conv2D(64, (4, 25), activation='relu', data_format='channels_first', name="EEG_feature_Conv2D1"),
    MaxPooling2D((1, 2), name="EEG_feature_pool2"),
    Conv2D(128, (50, 2), activation='relu', name="EEG_feature_Conv2D2"),
    Flatten(name="EEG_feature_flatten"),

    ], name="EEG_Classifier")

    encoder_inputs = Input(shape=(channels, observations, 1))
    latent = model(encoder_inputs)
    latent = Dense(512, activation='relu', name="EEG_feature_FC512")(latent)  ## extract and use this as latent space for input to GAN
    latent = Dropout(0.1, name="EEG_feature_drop3")(latent)
    latent = BatchNormalization(name="EEG_feature_BN2")(latent)
    classification_main = Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")(latent)
    classification_type = Dense(num_types, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_type_Labels")(latent)
    full_model = Model(inputs=encoder_inputs, outputs=[classification_main, classification_type])


    return full_model

def convolutional_encoder_model_128_dual(channels, observations, num_classes, num_types = 2, verbose=False):


    model = Sequential([
    BatchNormalization(input_shape=(channels, observations, 1), name="EEG_BN1"),
    Conv2D(128, (1, 4), activation='relu', padding='same', name="EEG_series_Conv2D"),
    Conv2D(64, (channels, 1), activation='relu',padding='same', name="EEG_channel_Conv2D"),
    MaxPooling2D((1, 2), name="EEG_feature_pool1"),
    Conv2D(64, (4, 25), activation='relu', data_format='channels_first', name="EEG_feature_Conv2D1"),
    MaxPooling2D((1, 2), name="EEG_feature_pool2"),
    Conv2D(128, (50, 2), activation='relu', name="EEG_feature_Conv2D2"),
    Flatten(name="EEG_feature_flatten"),

    BatchNormalization(name="EEG_feature_BN1"),
    Dense(512, activation='relu', name="EEG_feature_FC512"),
    Dropout(0.1, name="EEG_feature_drop1"),
    Dense(256, activation='relu', name="EEG_feature_FC256"),
    Dropout(0.1, name="EEG_feature_drop2")
    ], name="EEG_Classifier")

    encoder_inputs = Input(shape=(channels, observations, 1))
    latent = model(encoder_inputs)
    latent =     Dense(128, activation='relu', name="EEG_feature_FC128")(latent)  ## extract and use this as latent space for input to GAN
    latent = Dropout(0.1, name="EEG_feature_drop3")(latent)
    latent = BatchNormalization(name="EEG_feature_BN2")(latent)
    classification_main = Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")(latent)
    classification_type = Dense(num_types, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_type_Labels")(latent)
    full_model = Model(inputs=encoder_inputs, outputs=[classification_main, classification_type])


    return full_model







## LSTM encoder model


def LSTM_Classifier_dual_512(timesteps, features, latent_size,  num_classes, num_types = 2,  verbose=False): 

    model = Sequential([
    
    LSTM(latent_size, activation='tanh', return_sequences=True, input_shape=(timesteps, features) ,name = "EEG_feature_LSTM_1"),
    LSTM(latent_size, activation='tanh', return_sequences=True ,name = "EEG_feature_LSTM_2"),
    LSTM(latent_size, activation='tanh', return_sequences=True ,name = "EEG_feature_LSTM_3"),    
    LSTM(latent_size, activation='tanh', return_sequences=False ,name = "EEG_feature_LSTM_4")    
    
    ], name="classifier")

    encoder_inputs = Input(shape=(timesteps, features))
    latent = model(encoder_inputs)
    latent = BatchNormalization(name="EEG_feature_BN2")(latent)
    classification_main = Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")(latent)
    classification_type = Dense(num_types, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_type_Labels")(latent)
    full_model = Model(inputs=encoder_inputs, outputs=[classification_main, classification_type])


    return full_model
