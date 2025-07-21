import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.dual_models.eegclassifier import convolutional_encoder_model_128_dual, LSTM_Classifier_dual_512, convolutional_encoder_model_512_dual
from models.dual_models.eeggan import (build_discriminator, build_EEGgan, build_MoGCgenerator, build_MoGMgenerator, build_generator)

from models.dual_models.dcgan import (build_dc_discriminator, build_DCGgan, build_dc_generator)
from models.dual_models.capsgan import (build_caps_discriminator, build_capsGAN, build_dccaps_generator)
from models.EEGViT_pretrained import (EEGViT_pretrained)

from models.model_utils import (sample_images_eeg, save_model, combine_loss_metrics)


from utils.general_funcs_Jared import use_or_make_dir

import torch
import argparse

print(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
print(os.getcwd())

#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset", default = "processed_dataset/filter_mne_car",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training LSTM : 000thresh_AllStackLstm_64_dual_All_2.pkl / CNN 000thresh_AllSlidingCNN_dual_28_All.pkl / 000thresh_AllStackTransformer_All.pkl", default = "000thresh_AllStackLstm_64_dual_All_2.pkl" , required=False)

parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)

parser.add_argument('--classifier_path', type=str, help="directory to the classifier", default= "trained_models/classifiers", required=False)
parser.add_argument('--classifier_model', type=str, help="Name of the model", default= "eeg_classifier_adm5", required=False)
parser.add_argument('--GAN_type', type=str, help="DC or AC or CAPS", default = "CAPS",required=False)
parser.add_argument('--GAN_SubType', type=str, help="Only Relevant if using ACGAN model, Types available: M,B,C", default= "C", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/GANs",required=False)

parser.add_argument('--ClassifierImplementation', type = str, help = "TF or Torch", default = "TF")
parser.add_argument('--classifierType', type = str, help = "CNN or LSTM or Transformer", default = "LSTM")
parser.add_argument('--classifierName', type = str, help = "CNN_all_stacked_signals_dual_128 or CNN_all_stacked_signals_dual_512_28_ori or LSTM_all_stacked_signals_dual_512_64_ori or Transformer_all_stacked_signals", default = "LSTM_all_stacked_signals_dual_512_64_ori")

parser.add_argument('--datasetType', type = str, help = "CNN_encoder or LSTM_encoder or Transformer_encoder", default = "LSTM_encoder")

parser.add_argument('--latent_size', type=int, help="Size of the latent, 128 or 512", default = 512, required=False)
parser.add_argument('--batch_size', type=int, help="Batch size", default = 32,required=False)
parser.add_argument('--epochs', type=int, help="Number of epochs to run", default = 2000,required=False)
parser.add_argument('--save_interval', type=int, help="how many epochs before saving", default = 250,required=False)


args = parser.parse_args()

#Setup variables
batch_size = args.batch_size
epochs = args.epochs
save_interval = args.save_interval
eeg_encoding_dim = args.latent_size


class_labels = [0,1,2,3,4,5,6,7,8,9]

generator_type = args.GAN_SubType #C for concatenation M for Multiplication B for Basic
if args.GAN_type == "AC":
    if generator_type == "B":
        GAN_SubType = "Basic"
    else:
        GAN_SubType = f"MoG{generator_type}"

else: GAN_SubType = args.GAN_type


indexes = [i for i, char in enumerate(args.dataset_pickle) if char == '_']
run_id = args.dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
classifier_id = f"{run_id}_{args.epochs}_{args.classifierName}_{GAN_SubType}"

#Output save path name
model_save_path = f"{args.output_dir}/{args.classifierType}_GAN/{args.GAN_type}/{run_id}_{GAN_SubType}_{eeg_encoding_dim}"
model_save_path_imgs = f"{model_save_path}/imgs"

#Generate classifier path name depending if implementation is torch or TF
if args.ClassifierImplementation == "TF":
    classifier_model_path = f"{args.classifier_path}/{args.input_dir}/{run_id}/{args.classifierName}/eeg_classifier_adm5_final.h5"
elif args.ClassifierImplementation == "Torch":
    classifier_model_path = f"{args.classifier_path}/{args.input_dir}/{run_id}/{args.classifierName}/eeg_classifier_adm5_final.pth"
    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

else:
    raise FileNotFoundError(f"{args.ClassifierImplementation} is not a valid implementation")

print("Taking Classifier from : ", classifier_model_path)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

## EEG data
eeg_data_dir = f"{args.root_dir}/{args.datasetType}/{args.input_dir}"
eeg_data_file = f"{eeg_data_dir}/{args.dataset_pickle}"
print(f"** Reading data file {eeg_data_file}")
eeg_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
(x_imgs_train, y_primary_train, y_secondary_train) , (x_imgs_test, y_primary_test, y_secondary_test) = (eeg_data['x_train_img'], eeg_data['y_train'], eeg_data['y_secondary_train']) , (eeg_data['x_test_img'], eeg_data['y_test'], eeg_data['y_secondary_test'])
x_eeg_train, x_eeg_test = eeg_data['x_train_eeg'],  eeg_data['x_test_eeg']

x_imgs_train = (np.array(x_imgs_train) - 127.5) / 127.5
x_imgs_test = (np.array(x_imgs_test) - 127.5) / 127.5

num_of_class_labels = y_primary_train.shape[1]
num_of_class_type_labels = y_secondary_train.shape[1]

# dataset = "2022Data"

print(f"Reading data file {eeg_data_file}")
## ################
## Create GAN model
## ################

gan_optimizer = Adam(0.0002, 0.5, decay=1e-6)
discrim_losses = ['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']  #sparse_
gen_losses = ['mse']

# build discriminator sub model
print("Shape of training images is")
print((x_imgs_train.shape[1],x_imgs_train.shape[2],x_imgs_train.shape[3]))

print(f"** Training model for type: {args.GAN_type}")
if args.GAN_type == "AC":
    print(f"*** Training sub model for type: {args.GAN_SubType}")

    discriminator = build_discriminator((x_imgs_train.shape[1],x_imgs_train.shape[2],x_imgs_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
    discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    # build generator sub model
    if generator_type == "C":
        generator = build_MoGCgenerator(eeg_encoding_dim,x_imgs_train.shape[3],num_of_class_labels, num_of_class_type_labels)
    elif generator_type == "M":
        generator = build_MoGMgenerator(eeg_encoding_dim,x_imgs_train.shape[3],num_of_class_labels, num_of_class_type_labels)
    elif generator_type == "B":
        generator = build_generator(eeg_encoding_dim,    x_imgs_train.shape[3],num_of_class_labels, num_of_class_type_labels)

    generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])

elif args.GAN_type == "DC":
    discriminator = build_dc_discriminator((x_imgs_train.shape[1],x_imgs_train.shape[2],x_imgs_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
    discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    generator = build_dc_generator(eeg_encoding_dim, x_imgs_train.shape[3],num_of_class_labels, num_of_class_type_labels)
    generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])

elif args.GAN_type == "CAPS":

    discriminator, _, _ = build_caps_discriminator((x_imgs_train.shape[1],x_imgs_train.shape[2],x_imgs_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
    discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    generator = build_dccaps_generator(eeg_encoding_dim, x_imgs_train.shape[3], num_of_class_labels, num_of_class_type_labels)
    generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])

# prime generator.
noise = Input(shape=(eeg_encoding_dim,))
label = Input(shape=(1,), dtype=tf.int32)
label_type = Input(shape=(1,), dtype=tf.int32)

img = generator([noise, label, label_type])

# set discriminator used in combined model to none trainable.
discriminator.trainable = False

if args.GAN_type == "CAPS":
    masking_label = Input(shape=(num_of_class_labels,))
    valid_class, target_label, target_label_type = discriminator([img, masking_label])
    combined = build_capsGAN(eeg_encoding_dim, num_of_class_labels, generator, discriminator)
else:
    valid_class, target_label, target_label_type = discriminator(img)
    combined = build_EEGgan(eeg_encoding_dim, generator, discriminator)


# Create combined EEGGan model.

combined.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])

## #############
# EEG Classifier
## #############

if args.ClassifierImplementation == "TF":

    if args.classifierType == "LSTM":
        classifier = LSTM_Classifier_dual_512(x_eeg_train.shape[1],  x_eeg_train.shape[2], 512, num_of_class_labels, num_of_class_type_labels)

    elif args.classifierType == "CNN":
        if eeg_encoding_dim == 128:
            classifier = convolutional_encoder_model_128_dual(x_eeg_train.shape[1], x_eeg_train.shape[2], num_of_class_labels, num_of_class_type_labels)
        elif eeg_encoding_dim == 512:
            classifier = convolutional_encoder_model_512_dual(x_eeg_train.shape[1], x_eeg_train.shape[2], num_of_class_labels, num_of_class_type_labels)

    classifier.load_weights(classifier_model_path)
    layer_names = ['EEG_feature_BN2','EEG_Class_Labels', 'EEG_Class_type_Labels']
    encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
    encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)



elif args.ClassifierImplementation == "Torch":
    if args.classifierType == "Transformer":
        encoder_model = EEGViT_pretrained()
        encoder_model.load_state_dict(torch.load(classifier_model_path, map_location=torch.device('cpu')))
        encoder_model.eval() 
    ## Set up with custom training loop

history = {'Discriminator':[],'Generator':[]}
print(f"** Classifier used: {classifier_model_path}")


## Encode all signals first before training
if args.ClassifierImplementation == "TF":
    encoded_eeg_all, encoded_labels_all, encoded_labels_type_all = encoder_model.predict(x_eeg_train)
    
    predicted_labels = np.argmax(encoded_labels_all,axis=1)
    predicted_labels_type = np.argmax(encoded_labels_type_all,axis=1)

elif args.ClassifierImplementation == "Torch":
    with torch.no_grad():
        eeg_samples = x_eeg_train[:,np.newaxis,:,:]
        tensor_eeg  = torch.from_numpy(eeg_samples).to(device)
        encoded_labels_all, encoded_eeg_all  = encoder_model(tensor_eeg)
        predicted_labels = torch.argmax(encoded_labels_all, dim=1)

        encoded_eeg_all = encoded_eeg_all.cpu().numpy()
        predicted_labels = predicted_labels.cpu().numpy()
        encoded_eeg_all = tf.convert_to_tensor(encoded_eeg_all)
        predicted_labels = tf.convert_to_tensor(predicted_labels)

for epoch in range(epochs+1):

    # ---------------------
    #  Train Discriminator: Discriminator is trained using real and generated images with the goal to identify the difference
    # ---------------------
    # Sample EEG latent space from EEG Classifier as generator input
    sample_indexs = np.random.choice(x_eeg_train.shape[0], size=batch_size, replace=False)
    
    #Extract all processed data from the randomly chosen sampled indexes
    encoded_eeg = encoded_eeg_all[sample_indexs]
    encoded_primary_labels = predicted_labels[sample_indexs]
    encoded_secondary_labels = predicted_labels_type[sample_indexs]
    
    # Obtain the corresponding labels for the primary and secondary labels
    sampled_primary_labels = np.argmax(y_primary_train[sample_indexs],axis=1)
    sampled_secondary_labels = np.argmax(y_secondary_train[sample_indexs],axis=1)


    # Select corresponding Real Images from dataset
    imgs = x_imgs_train[sample_indexs]

    # Generate a half batch of new images
    gen_imgs = generator.predict([encoded_eeg, encoded_primary_labels, encoded_secondary_labels])

    # Train the discriminator, to recognise real/fake images
    if args.GAN_type == "CAPS":
        d_loss_real = discriminator.train_on_batch([imgs, y_primary_train[sample_indexs]], [valid, sampled_primary_labels, sampled_secondary_labels], return_dict=True)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, encoded_labels_all[sample_indexs]], [fake, encoded_primary_labels, encoded_secondary_labels], return_dict=True)
        d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)
        
        g_loss_disc = combined.train_on_batch([encoded_eeg, encoded_primary_labels, encoded_secondary_labels, encoded_labels_all[sample_indexs]], [valid, encoded_primary_labels, encoded_secondary_labels], return_dict=True)
        g_loss_mse = generator.train_on_batch([encoded_eeg, encoded_primary_labels, encoded_secondary_labels], imgs, return_dict = True)
        
        g_loss = {'loss' : g_loss_disc['loss'] + g_loss_mse['loss'], 'Discriminator_loss' : g_loss_disc['Discriminator_loss'], 'MSE_loss': g_loss_mse['loss']}


    else:
        d_loss_real = discriminator.train_on_batch(imgs, [valid, sampled_primary_labels, sampled_secondary_labels], return_dict=True)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, encoded_primary_labels, encoded_secondary_labels], return_dict=True)
        d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)

        g_loss_disc = combined.train_on_batch([encoded_eeg, encoded_primary_labels, encoded_secondary_labels], [valid, encoded_primary_labels, encoded_secondary_labels], return_dict=True)
        g_loss_mse = generator.train_on_batch([encoded_eeg, encoded_primary_labels, encoded_secondary_labels], imgs, return_dict = True)

        g_loss = {'loss' : g_loss_disc['loss'] + g_loss_mse['loss'], 'Discriminator_loss' : g_loss_disc['Discriminator_loss'], 'MSE_loss': g_loss_mse['loss']}

    # ---------------------
    #  Train Generator:
    # ---------------------
    # Train the generator using the combined GAN model so that Generator learns to create better images to fool the discriminator
    

    history['Discriminator'].append(d_loss)
    history['Generator'].append(g_loss)
    # Plot the progress
    print (f"Epoch {epoch:5d}: [D loss: {d_loss['loss']:.6f}, Validity acc.: {d_loss['Dis_Validity_accuracy']:.2%}, Label acc: {d_loss['Dis_Class_Label_accuracy']:.2%}, Label type acc: {d_loss['Dis_Class_type_Label_accuracy']:.2%}]")
    print(f"             [G loss: {g_loss['loss']:.6f}] [D loss: {g_loss['Discriminator_loss']:.6f}] [MSE loss: {g_loss['MSE_loss']:.6f}]")

    # If at save interval => save generated image samples
    if epoch % save_interval == 0 or epoch == epochs:
        save_model(generator, GAN_SubType, classifier_id, f"{GAN_SubType}_EEG_Generator_{epoch}", main_dir, model_save_path)
        save_model(discriminator, GAN_SubType, classifier_id, f"{GAN_SubType}_EEG_Discriminator_{epoch}", main_dir, model_save_path)
        sample_images_eeg(epoch, gen_imgs, [sampled_primary_labels,encoded_primary_labels], main_dir, model_save_path_imgs)



save_path = model_save_path

print(f"** Saving model to {save_path}")
with open(os.path.join(save_path,f"{GAN_SubType}_EEGGan_history.pkl"),"wb") as f:
    pickle.dump(history,f)

combined.save_weights(os.path.join(save_path,f"{GAN_SubType}_EEGGan_combined_weights.h5"))
generator.save_weights(os.path.join(save_path,f"{GAN_SubType}_EEGGan_generator_weights.h5"))
discriminator.save_weights(os.path.join(save_path,f"{GAN_SubType}_EEGGan_discriminator_weights.h5"))


pass