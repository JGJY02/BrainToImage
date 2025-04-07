from keras.layers import (BatchNormalization, Conv2D, Conv3D,Dense,
                            Dropout, Flatten, MaxPooling2D, UpSampling2D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, LSTM, RepeatVector, TimeDistributed,Reshape, Input, Lambda)
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt
from keras import backend as K
from transformers import TFViTModel

"""
Title: ViT model design

Purpose:
    ViT model taken from https://github.com/ruiqiRichard/EEGViT/tree/master

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
        image_size=(129, 14),
        patch_size=(8, 1)
    )
    vit_model = TFViTModel(config)
    vit_model.embeddings.patch_embeddings.projection = Conv2d(filters = 768, kernel_size=(8, 1), strides=(8, 1),
                                                                groups=256)
    vit_model.pooler_act = Sequential([Dropout(p=0.1),
                                   aaDense(768, 2, bias=True)])
    encoder_inputs = Input(shape=(channels, observations, 1))
    out = Conv2d(256, (1, 36), stride=(1, 36), padding=(0, 2), bias=False)(encoder_inputs)
    out = BatchNormalization(out)
    out = vit_model(out)

    model = Model(encoder_inputs, [out.last_hidden_state , out.pooler_output], name="ViT_Transformer")



