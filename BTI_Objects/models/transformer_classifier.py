from keras.layers import (BatchNormalization, Conv2D, Conv3D,Dense,
                            Dropout, ZeroPadding2D, Flatten, MaxPooling2D, UpSampling2D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, LSTM, RepeatVector, TimeDistributed,Reshape, Input, Lambda)
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt
from keras import backend as K
import transformers
from transformers import TFViTForImageClassification, ViTImageProcessor

import tensorflow as tf
"""
Title: ViT model design

Purpose:
    ViT model taken from https://github.com/ruiqiRichard/EEGViT/tree/master

Author: Jared Goh
Date: 08/04/2025
Version: <Version number>

Usage:
   Build transformer Model

Notes:
    <Any additional notes or considerations>

Examples:
    <Example usage scenarios>
"""

def EEGViT_raw(channels, observations, num_classes, verbose=False):
    config = transformers.ViTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_channels=256,
        image_size=(129,14),
        patch_size=(8,1)
    )
    # vit_processor = ViTImageProcessor(
    # size={"height": 129, "width": 14}, 
    # do_resize=False,
    # do_normalize=False
    # )
    #Obtain the pretrained network
    model_name = "google/vit-base-patch16-224"
    config = transformers.ViTConfig.from_pretrained(model_name)
    config.update({'num_channels': 256})
    config.update({'image_size': (129,14)})
    config.update({'patch_size': (8,1)})

    vit_model = transformers.TFViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    vit_model.vit.embeddings.patch_embeddings.projection = Conv2D(filters = 768, kernel_size=(8, 1), strides=(8, 1),
                                                                groups=256)
    vit_model.pooler_act = Sequential([Dropout(0.1),
                                   Dense(num_classes, use_bias=True)])
    encoder_inputs = Input(shape=(channels, observations, 1))
    
    # out = Conv2D(256, (1, 36), stride=(1, 36), padding=(0, 2), bias=False)(encoder_inputs) # Original but modified to still get shape 256x63x14
    out = ZeroPadding2D(padding=((0, 0), (2, 2)))(encoder_inputs)  # (top, bottom), (left, right)
    out = Conv2D(256, (1, 21), strides=(1, 1), padding="valid", use_bias=False, name = "Formatting_Layer")(out) 
    out = tf.transpose(out, perm=[0, 3, 2, 1])
    out = BatchNormalization()(out)
    # out = vit_processor(out, return_tensors="tf")
    out = vit_model(out)

    model = Model(encoder_inputs, out.pooler_output, name="ViT_Transformer")

    return model


