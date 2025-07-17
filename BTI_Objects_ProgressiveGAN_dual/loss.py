# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                          Embedding, Flatten, Input, LeakyReLU, Reshape,
                          UpSampling2D, ZeroPadding2D, multiply, concatenate)
import config




def preprocess_image(image):
    # Normalize images to [0, 1] and apply ImageNet's mean and std normalization
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.transpose(image, perm=[0, 2, 3, 1])
    image = tf.image.resize(image, (64, 64))  # VGG expects 224x224 input size
    # image = tf.keras.applications.vgg16.preprocess_input(image)  # ImageNet preprocessing
    return image

def perceptual_loss_func(generated_image, real_image, vgg_model):
    # Preprocess images
    generated_image = preprocess_image(generated_image)
    real_image = preprocess_image(real_image)
    
    # Get feature maps from the VGG model
    gen_features = vgg_model(generated_image)
    real_features = vgg_model(real_image)
    
    # Compute perceptual loss (e.g., using L1 loss between feature maps)
    loss = 0
    loss = tf.reduce_mean(tf.square(gen_features - real_features)) 


    
    return loss


##
def is_tf_expression(x):
    return isinstance(x, tf.compat.v1.Tensor) or isinstance(x, tf.compat.v1.Variable) or isinstance(x, tf.compat.v1.Operation)

# def processSignals(eeg_signal, E):
#     # eeg_samples = tf.compat.v1.gather(eeg_signal, tf.compat.v1.random_uniform([minibatch_size], 0, eeg_signal.shape[0], dtype=tf.compat.v1.int32))
    
#     # latents = E.get_concrete_function(eeg_samples)
#     if config.TfOrTorch == "TF":
#         latents, labels = E(eeg_signal)

#     elif config.TfOrTorch == "Torch":
#         if torch.cuda.is_available():
#             gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
#             torch.cuda.set_device(gpu_id)
#             device = torch.device(f"cuda:{gpu_id}")
#         else:
#             device = torch.device("cpu")

#         # with tf.compat.v1.Session() as sess:
#         #     eeg_signal = sess.run(eeg_signal)
        
#         # eeg_signal = eeg_signal.numpy()
#         eeg_signal = np.transpose(eeg_signal, (0,2,1))[:,np.newaxis,:,:]
#         # print(eeg_signal.shape)
#         tensor_eeg  = torch.from_numpy(eeg_signal).to(device)
#         encoded_labels, encoded_eeg  = E(tensor_eeg)

#         encoded_eeg = encoded_eeg.detach().numpy()
#         predicted_labels = encoded_labels.detach().numpy()
#         latents = tf.convert_to_tensor(encoded_eeg)
#         labels = tf.convert_to_tensor(predicted_labels)

#     return latents, labels

##

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.compat.v1.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.compat.v1.cast(v, tf.compat.v1.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

# def G_wgan_acgan(G, D, opt, training_set, minibatch_size, 
#     cond_weight = 1.0): # Weight of the conditioning term.

#     latents = tf.compat.v1.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     labels = training_set.get_random_labels_tf(minibatch_size)
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
#     loss = -fake_scores_out

#     if D.output_shapes[1][1] > 0:
#         with tf.compat.v1.name_scope('LabelPenalty'):
#             label_penalty_fakes = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#         loss += label_penalty_fakes * cond_weight
#     return loss

def G_wgan_acgan(G, D, vgg_model, encoded_signals, encoded_labels, encoded_labels_type, opt, training_set,  minibatch_size, reals_gpu,
    cond_weight = 1): # Weight of the conditioning term.
    # latents, labels = processSignals(eeg_signal= eeg_signals, E=E)
    # print(cond_weight)
    latents = encoded_signals
    # print("Extracted signal Latent : ", latents.shape)
    # print("Extracted signal Labels : ", labels.shape)

    # print("Extracted signal Latent : ", latents.dtype)
    # print("Extracted signal Labels : ", labels.dtype)

    # latents = tf.compat.v1.random_normal([minibatch_size] + G.input_shapes[0][1:])
    # labels = training_set.get_random_labels_tf(minibatch_size)
    # print("random signal Latent : ",latents.shape)
    # print("random signal Labels : ",labels.shape)
    # print("random signal Latent : ", latents.dtype)
    # print("random signal Labels : ", labels.dtype)
    fake_images_out = G.get_output_for(latents, encoded_labels, encoded_labels_type, is_training=True)
    fake_scores_out, fake_labels_out, fake_scores_type_out, fake_labels_type_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out -fake_scores_type_out

    if D.output_shapes[1][1] > 0:
        with tf.compat.v1.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=encoded_labels, logits=fake_labels_out)
            label_penalty_fakes_type = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=encoded_labels_type, logits=fake_labels_type_out)
        loss += (label_penalty_fakes+label_penalty_fakes_type) * cond_weight
    
    with tf.compat.v1.name_scope('MatchingPenalty'):
        print("Hello")
        mse_loss = tf.reduce_mean(tf.square(reals_gpu - fake_images_out), axis=[1, 2, 3])
        mae_loss = tf.reduce_mean(tf.abs(reals_gpu - fake_images_out), axis=[1, 2, 3])
        # ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(reals_gpu, fake_images_out, max_val=1.0))
        # perceptual_loss = perceptual_loss_func(reals_gpu, fake_images_out, vgg_model)
        print("Loss obtained")

    loss += mse_loss + mae_loss

    #     perceptual_loss_val = perceptual_loss(reals_gpu, fake_images_out)

    # loss += perceptual_loss_val

    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

# def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
#     wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
#     wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
#     wgan_target     = 1.0,      # Target value for gradient magnitudes.
#     cond_weight     = 1.0):     # Weight of the conditioning terms.

#     latents = tf.compat.v1.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
#     fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
#     real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
#     fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
#     loss = fake_scores_out - real_scores_out

#     with tf.compat.v1.name_scope('GradientPenalty'):
#         mixing_factors = tf.compat.v1.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
#         mixed_images_out = tfutil.lerp(tf.compat.v1.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
#         mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
#         mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
#         mixed_loss = opt.apply_loss_scaling(tf.compat.v1.reduce_sum(mixed_scores_out))
#         mixed_grads = opt.undo_loss_scaling(fp32(tf.compat.v1.gradients(mixed_loss, [mixed_images_out])[0]))
#         mixed_norms = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(mixed_grads), axis=[1,2,3]))
#         mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
#         gradient_penalty = tf.compat.v1.square(mixed_norms - wgan_target)
#     loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

#     with tf.compat.v1.name_scope('EpsilonPenalty'):
#         epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.compat.v1.square(real_scores_out))
#     loss += epsilon_penalty * wgan_epsilon

#     if D.output_shapes[1][1] > 0:
#         with tf.compat.v1.name_scope('LabelPenalty'):
#             label_penalty_reals = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
#             label_penalty_fakes = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#             label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
#             label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
#         loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
#     return loss

def D_wgangp_acgan(G, D, encoded_signals, encoded_labels, encoded_labels_type, opt, training_set, minibatch_size, reals, labels, 
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1):     # Weight of the conditioning terms.

    # latents, labels_predicted = processSignals(eeg_signal=eeg_signals, E=E)
    
    latents = encoded_signals

    fake_images_out = G.get_output_for(latents, encoded_labels, encoded_labels_type, is_training=True)
    real_scores_out, real_labels_out, real_scores_type_out, real_labels_type_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out, fake_scores_type_out, fake_labels_type_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    real_scores_type_out = tfutil.autosummary('Loss/real_scores_type', real_scores_type_out)

    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    fake_scores_type_out = tfutil.autosummary('Loss/fake_scores_type', fake_scores_type_out)

    loss = (fake_scores_out+ fake_scores_type_out) - (real_scores_out + real_scores_type_out)

    with tf.compat.v1.name_scope('GradientPenalty'):
        mixing_factors = tf.compat.v1.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.compat.v1.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out, mixed_scores_type_out, mixed_labels_type_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out += mixed_scores_type_out
        
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.compat.v1.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.compat.v1.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.compat.v1.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.compat.v1.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.compat.v1.square(real_scores_out))
        epsilon_penalty_type = tfutil.autosummary('Loss/epsilon_penalty_type', tf.compat.v1.square(real_scores_type_out))

    loss += epsilon_penalty * wgan_epsilon
    loss += epsilon_penalty_type * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.compat.v1.name_scope('LabelPenalty'):
            label_penalty_reals = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=encoded_labels, logits=real_labels_out)
            label_penalty_fakes = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=encoded_labels, logits=fake_labels_out)

            label_penalty_reals_type = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=encoded_labels_type, logits=real_labels_type_out)
            label_penalty_fakes_type = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=encoded_labels_type, logits=fake_labels_type_out)

            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
            
            label_penalty_reals_type = tfutil.autosummary('Loss/label_penalty_reals_type', label_penalty_reals_type)
            label_penalty_fakes_type = tfutil.autosummary('Loss/label_penalty_fakes_type', label_penalty_fakes_type)

        loss += (label_penalty_reals + label_penalty_fakes + label_penalty_reals_type + label_penalty_fakes_type) * cond_weight
    return loss

#----------------------------------------------------------------------------
