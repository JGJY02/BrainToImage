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
# define the CNN model for classification
def convolutional_encoder_model(channels, observations, num_classes, verbose=False):
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
    Dropout(0.1, name="EEG_feature_drop2"),
    Dense(128, activation='relu', name="EEG_feature_FC128"),   ## extract and use this as latent space for input to GAN
    Dropout(0.1, name="EEG_feature_drop3"),
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="EEG_Classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model

if __name__ == '__main__':
    classifier = convolutional_encoder_model(9, 32, 10, verbose=True)


## Testing other encoders
def convolutional_encoder_model_512(channels, observations, num_classes, verbose=False):
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
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="EEG_Classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model

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
def convolutional_encoder_model_expanded(channels, observations, num_classes, verbose=False):
    print(channels)
    print(observations)

    model = Sequential([
    BatchNormalization(input_shape=(channels, observations, 1), name="EEG_BN1"),
    Conv2D(128, (1, 4), activation='relu', padding='same', name="EEG_series_Conv2D_11"),
    Conv2D(64, (channels, 1), activation='relu',padding='same', name="EEG_channel_Conv2D_12"),
    MaxPooling2D((1, 2), name="EEG_feature_pool1"),
    Conv2D(32, (channels, 1), activation='relu',padding='same', name="EEG_channel_Conv2D_13"),
    MaxPooling2D((1, 2), name="EEG_feature_pool2"),
    Conv2D(32, (channels, 2), activation='relu',padding='same', data_format='channels_first', name="EEG_feature_Conv2D_21"),
    MaxPooling2D((1, 2), name="EEG_feature_pool3"),
    Conv2D(64, (channels, 1), activation='relu',padding='same', name="EEG_channel_Conv2D_23"),
    Conv2D(128, (1, 4), activation='relu', padding='same', name="EEG_series_Conv2D_24"),
    Flatten(name="EEG_feature_flatten"),
    BatchNormalization(name="EEG_feature_BN1"),
    Dense(512, activation='relu', name="EEG_feature_FC512"),
    Dropout(0.1, name="EEG_feature_drop1"),
    Dense(256, activation='relu', name="EEG_feature_FC256"),
    Dropout(0.1, name="EEG_feature_drop2"),
    Dense(128, activation='relu', name="EEG_feature_FC128"),   ## extract and use this as latent space for input to GAN
    Dropout(0.1, name="EEG_feature_drop3"),
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="EEG_Classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model

if __name__ == '__main__':
    classifier = convolutional_encoder_model(9, 32, 10, verbose=True)

def convolutional_encoder_model_spectrogram_stacked(channels, width, height, num_classes, verbose=False):

    model = Sequential([
    BatchNormalization(input_shape=(width, height, channels,1), name="EEG_BN1"),
    Conv3D(128, (3, 3, 1), activation='relu', padding='same', name="EEG_series_Conv3D_11"),
    Conv3D(64, (3, 3, 1), activation='relu',padding='same', name="EEG_channel_Conv3D_12"),
    MaxPooling3D((2, 2, 1), name="EEG_feature_pool1"),
    Conv3D(32, (3, 3, 1), activation='relu',padding='same', name="EEG_channel_Conv3D_13"),
    MaxPooling3D((3, 3, 1), name="EEG_feature_pool2"),
    Conv3D(32, (3, 3, 1), activation='relu',padding='same', name="EEG_feature_Conv3D_21"),
    Conv3DTranspose(32, (2, 2, 1), name="EEG_feature_pool3"),
    Conv3D(64, (3, 3, 1), activation='relu',padding='same', name="EEG_channel_Conv3D_22"),
    Conv3D(128, (3, 3, 1), activation='relu', padding='same', name="EEG_series_Conv3D_23"),
    Conv3DTranspose(128, (2, 2, 1), name="EEG_feature_pool4"),
    Flatten(name="EEG_feature_flatten"),
    BatchNormalization(name="EEG_feature_BN1"),
    Dense(512, activation='relu', name="EEG_feature_FC512"),
    Dropout(0.1, name="EEG_feature_drop1"),
    Dense(256, activation='relu', name="EEG_feature_FC256"),
    Dropout(0.1, name="EEG_feature_drop2"),
    Dense(128, activation='relu', name="EEG_feature_FC128"),   ## extract and use this as latent space for input to GAN
    Dropout(0.1, name="EEG_feature_drop3"),
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="EEG_Classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model

if __name__ == '__main__':
    classifier = convolutional_encoder_model(9, 32, 10, verbose=True)


# 128, 64 gets 65% for the convolution

def convolutional_encoder_model_spectrogram(width, height, num_classes, verbose=False): 

    model = Sequential([
    BatchNormalization(input_shape=(width, height,1), name="EEG_BN1"),

    Conv2D(64, (3, 3), activation='relu',  padding='same', name="EEG_series_Conv3D_111"),
    # Conv2D(128, (3, 3), activation='relu',  padding='same', name="EEG_series_Conv3D_112"),
    MaxPooling2D((2, 2), name="EEG_feature_pool1"),
    Conv2D(32, (3, 3), activation='relu',  padding='same', name="EEG_series_Conv3D_121"),
    # Conv2D(64, (3, 3), activation='relu', padding='same', name="EEG_channel_Conv3D_122"),
    MaxPooling2D((2, 2), name="EEG_feature_pool2"),

    # ## Bottle Neck
    Conv2D(256, (3, 3), activation='relu',  padding='valid', name="EEG_series_bottle_Conv3D_111"),
    Conv2D(256, (3, 3), activation='relu',  padding='valid', name="EEG_series_bottle_Conv3D_112"),

    ## End of bottle neck
    
    Conv2D(32, (3, 3), activation='relu', padding='same', name="EEG_channel_Conv3D_211"),
    # Conv2D(64, (3, 3), activation='relu', padding='same', name="EEG_channel_Conv3D_212"),
    Conv2DTranspose(32, (2, 2), name="EEG_feature_pool3"),
    Conv2D(64, (3, 3), activation='relu',  padding='same', name="EEG_series_Conv3D_221"),
    # Conv2D(128, (3, 3), activation='relu',  padding='same', name="EEG_series_Conv3D_222"),
    Conv2DTranspose(64, (2, 2), name="EEG_feature_pool4"),

    Flatten(name="EEG_feature_flatten"),
    BatchNormalization(name="EEG_feature_BN1"),
    Dense(512, activation='relu', name="EEG_feature_FC512"),
    Dropout(0.4, name="EEG_feature_drop1"),
    Dense(256, activation='relu', name="EEG_feature_FC256"),
    Dropout(0.4, name="EEG_feature_drop2"),
    Dense(128, activation='relu', name="EEG_feature_FC128"),   ## extract and use this as latent space for input to GAN
    Dropout(0.4, name="EEG_feature_drop3"),
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="EEG_Classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model

if __name__ == '__main__':
    classifier = convolutional_encoder_model(9, 32, 10, verbose=True)

## Latent representation in the middle
def convolutional_encoder_model_spectrogram_middle_latent(width, height, verbose=False): 

# Encoder
    model = Sequential([
        BatchNormalization(input_shape=(width, height,1), name="EEG_BN1"),

        Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu"),  
        Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu"),  
        Flatten(),                                                                # Flatten to 1D vector
        Dense(128, activation="relu", name="latent_vector"),                       # Latent vector (128)
        Dense(16 * 16 * 64, activation="relu"),            
        Reshape((16, 16, 64)),                             
        Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu"),  
        Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu"),  
        Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid") 
    ], name = "autoencoder")


    
    return model


## LSTM encoder model

def LSTM_Classifier(timesteps, features, num_classes, latent_size = 512, verbose=False): 

    model = Sequential([
    
    LSTM(latent_size*2, activation='tanh', return_sequences=True, input_shape=(timesteps, features) ,name = "EEG_feature_LSTM_1"),
    LSTM(latent_size, activation='tanh', return_sequences=False ,name = "EEG_feature_LSTM_2"),
    BatchNormalization(name="EEG_feature_BN2"),
    Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name="classifier")

    if verbose:
        model.summary(show_trainable=True,expand_nested=True)

    return model




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

## VAE encoder
def sampling(args):
    """Reparameterization trick: z = μ + σ * ε"""
    mu, log_var = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], K.int_shape(mu)[1]))  # Random noise
    return mu + K.exp(0.5 * log_var) * epsilon  # Reparameterization trick

def vae_loss(x, recon_x, mu, log_var):
    # Reconstruction Loss (Mean Squared Error)
    recon_loss = K.mean(K.square(x - recon_x))
    
    # KL Divergence Loss
    kl_loss = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
    
    return recon_loss + kl_loss

def vae_encoder(timesteps, features, latent_size, verbose=False): 
    encoder_inputs = Input(shape=(timesteps,features))
    x = LSTM(latent_size*2, activation='tanh', return_sequences=True, name = "EEG_channel_LSTM_1_1")(encoder_inputs)
    x = LSTM(latent_size, activation='tanh', return_sequences=False, name = "EEG_channel_LSTM_1_2")(x)

    mu = Dense(latent_size)(x)
    log_var = Dense(latent_size)(x)

    z = Lambda(sampling, output_shape=(latent_size,))([mu, log_var])
    return Model(encoder_inputs, [mu, log_var, z], name="Encoder")

def vae_decoder(timesteps, features,latent_size, verbose=False): 
    decoder_inputs = Input(shape=(latent_size,))
    repeated = RepeatVector(timesteps)(decoder_inputs)
    lstm_out = LSTM(latent_size, activation='tanh', return_sequences=True,name = "EEG_channel_LSTM_2_1")(repeated)
    lstm_out = LSTM(latent_size*2, activation='tanh', return_sequences=True, name = "EEG_channel_LSTM_2_2")(lstm_out)

    outputs = Dense(features)(lstm_out)

    return Model(decoder_inputs, outputs, name="Decoder")

def LSTM_VAE_encoder_model(timesteps, features, latent_size, verbose=False):
    # Build Encoder
    encoder = vae_encoder(timesteps, features, latent_size)
    
    # Build Decoder
    decoder = vae_decoder(timesteps, features, latent_size)
    
    # Connect Encoder and Decoder
    inputs = Input(shape=(timesteps, features))
    mu, log_var, z = encoder(inputs)
    recon_x = decoder(z)
    
    # Define VAE Model
    vae = Model(inputs, recon_x, name="LSTM-VAE")
    vae.add_loss(vae_loss(inputs, recon_x, mu, log_var))

    return vae, encoder, decoder, mu, log_var


## Simple LSTM Model

## Simple classification model

def classification_model(latent_vector_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_dim=latent_vector_shape),
        BatchNormalization(name="EEG_feature_BN1"),
        Dense(64, activation='relu'),
        BatchNormalization(name="EEG_feature_BN2"),




        Dense(num_classes, activation='softmax',kernel_regularizer=l2(0.015), name="EEG_Class_Labels")
    ], name = "Classifier")

    return model