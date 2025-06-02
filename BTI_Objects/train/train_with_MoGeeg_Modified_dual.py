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
from models.eegclassifier import convolutional_encoder_model_512_dual, LSTM_Classifier_dual_512
from models.dual_models.eeggan import (build_discriminator, build_EEGgan, build_MoGCgenerator, build_MoGMgenerator, build_generator)

from models.dual_models.dcgan import (build_dc_discriminator, build_DCGgan, build_dc_generator)
from models.dual_models.capsgan import (build_caps_discriminator, build_capsGAN, build_dccaps_generator)
from models.EEGViT_pretrained import (EEGViT_pretrained)

from models.model_utils import (sample_images_eeg, save_model, combine_loss_metrics)


from utils.local_MNIST import get_balanced_mnist_subset, load_local_mnist
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
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training LSTM : 000thresh_AllStackLstm_dual_All_2.pkl / CNN 000thresh_AllSlidingCNN_All.pkl / 000thresh_AllStackTransformer_All.pkl", default = "000thresh_AllStackLstm_64_dual_All_2.pkl" , required=False)

parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)

parser.add_argument('--classifier_path', type=str, help="directory to the classifier", default= "trained_models/classifiers", required=False)
parser.add_argument('--classifier_model', type=str, help="Name of the model", default= "eeg_classifier_adm5", required=False)
parser.add_argument('--GAN_type', type=str, help="DC or AC or CAPS", default = "CAPS",required=False)
parser.add_argument('--model_type', type=str, help="M,B,C", default= "B", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/GANs",required=False)

parser.add_argument('--ClassifierImplementation', type = str, help = "TF or Torch", default = "TF")
parser.add_argument('--classifierType', type = str, help = "CNN or LSTM or Transformer", default = "LSTM")
parser.add_argument('--classifierName', type = str, help = "auto_encoder or spectrogram_auto_encoder or LSTM_all_stacked_signals or Transformer_all_stacked_signals", default = "LSTM_all_stacked_signals_dual_512_64_ori")

parser.add_argument('--datasetType', type = str, help = "CNN_encoder or LSTM_encoder or Transformer_encoder", default = "LSTM_encoder")


parser.add_argument('--batch_size', type=int, help="Batch size", default = 32,required=False)
parser.add_argument('--epochs', type=int, help="Number of epochs to run", default = 5000,required=False)
parser.add_argument('--save_interval', type=int, help="how many epochs before saving", default = 250,required=False)


args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
save_interval = args.save_interval
generator_type = args.model_type #C for concatenation M for Multiplication B for Basic



class_labels = [0,1,2,3,4,5,6,7,8,9]
eeg_encoding_dim = 512



if args.GAN_type == "AC":
    if generator_type == "B":
        model_type = "Basic"
    else:
        model_type = f"MoG{generator_type}"

else: model_type = args.GAN_type


indexes = [i for i, char in enumerate(args.dataset_pickle) if char == '_']
run_id = args.dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
classifier_id = f"{run_id}_{args.epochs}_{args.classifierName}_{model_type}"

#Output save path name
model_save_path = f"{args.output_dir}/{args.classifierType}_GAN/{args.GAN_type}/{run_id}_{model_type}_512"
model_save_path_imgs = f"{model_save_path}/imgs"

#Generate classifier path name
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

## load the MNIST trainig data
eeg_data_dir = f"{args.root_dir}/{args.datasetType}/{args.input_dir}"
eeg_data_file = f"{eeg_data_dir}/{args.dataset_pickle}"
print(f"** Reading data file {eeg_data_file}")
object_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
(x_train, y_train, y_secondary_train) , (x_test, y_test, y_secondary_test) = (object_data['x_train_img'], object_data['y_train'], object_data['y_secondary_train']) , (object_data['x_test_img'], object_data['y_test'], object_data['y_secondary_test'])

x_train = (np.array(x_train) - 127.5) / 127.5
x_test = (np.array(x_test) - 127.5) / 127.5
print((x_train.shape[1],x_train.shape[2],x_train.shape[3]))

num_of_class_labels = y_train.shape[1]
num_of_class_type_labels = y_secondary_train.shape[1]
## load the eeg training data

# dataset = "2022Data"

print(f"Reading data file {eeg_data_file}")
eeg_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
#x_train, y_train, x_test, y_test = eeg_data['x_train'], eeg_data['y_train'], eeg_data['x_test'  ], eeg_data['y_test'  ]
## ################
## Create GAN model
## ################
gan_optimizer = Adam(0.0002, 0.5, decay=1e-6)
discrim_losses = ['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']  #sparse_
gen_losses = ['categorical_crossentropy']
# build discriminator sub model
print("Shape of training is")
print((x_train.shape[1],x_train.shape[2],x_train.shape[3]))

print(f"** Training model for type: {args.GAN_type}")
if args.GAN_type == "AC":
    print(f"*** Training sub model for type: {args.model_type}")

    discriminator = build_discriminator((x_train.shape[1],x_train.shape[2],x_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
    discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    # build generator sub model

    if generator_type == "C":
        generator = build_MoGCgenerator(eeg_encoding_dim,x_train.shape[3],num_of_class_labels, num_of_class_type_labels)
    elif generator_type == "M":
        generator = build_MoGMgenerator(eeg_encoding_dim,x_train.shape[3],num_of_class_labels, num_of_class_type_labels)
    elif generator_type == "B":
        generator = build_generator(eeg_encoding_dim,    x_train.shape[3],num_of_class_labels, num_of_class_type_labels)

    generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])
elif args.GAN_type == "DC":
    discriminator = build_dc_discriminator((x_train.shape[1],x_train.shape[2],x_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
    discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    generator = build_dc_generator(eeg_encoding_dim, x_train.shape[3],num_of_class_labels, num_of_class_type_labels)
    generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])

elif args.GAN_type == "CAPS":

    discriminator, _, _ = build_caps_discriminator((x_train.shape[1],x_train.shape[2],x_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
    discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    generator = build_dccaps_generator(eeg_encoding_dim, x_train.shape[3], num_of_class_labels, num_of_class_type_labels)
    generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])

# prime generator.
noise = Input(shape=(eeg_encoding_dim,))
label = Input(shape=(1,), dtype=tf.int32)
label_type = Input(shape=(1,), dtype=tf.int32)

img = generator([noise, label, label_type])
print("Shape of image is ", img.shape)
# set discriminator used in combined model to none trainable.
discriminator.trainable = False

if args.GAN_type == "CAPS":
    masking_label = Input(shape=(len(class_labels),))
    valid_class, target_label, target_label_type = discriminator([img, masking_label])
    combined = build_capsGAN(eeg_encoding_dim, len(class_labels), generator, discriminator)
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
        classifier = LSTM_Classifier_dual_512(eeg_data['x_train_eeg'].shape[1],  eeg_data['x_train_eeg'].shape[2], 512, num_of_class_labels, num_of_class_type_labels)

    elif args.classifierType == "CNN":
        print(eeg_data['x_train_eeg'].shape[1], eeg_data['x_train_eeg'].shape[2])
        classifier = convolutional_encoder_model_512_dual(eeg_data['x_train_eeg'].shape[1], eeg_data['x_train_eeg'].shape[2], num_of_class_labels, num_of_class_type_labels)

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
    encoded_eeg_all, encoded_labels_all, encoded_labels_type_all = encoder_model.predict(eeg_data['x_train_eeg'])
    
    predicted_labels = np.argmax(encoded_labels_all,axis=1)
    predicted_labels_type = np.argmax(encoded_labels_type_all,axis=1)

elif args.ClassifierImplementation == "Torch":
    with torch.no_grad():
        # print(eeg_samples.shape)
        eeg_samples = eeg_data['x_train_eeg'][:,np.newaxis,:,:]
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
    # _train_ run used eeg data from train to predict on, so the model had seen this data.
    # _test_ run should use data from test as the classifier hasn't seen this data before.
    sample_indexs = np.random.choice(eeg_data['x_train_eeg'].shape[0], size=batch_size, replace=False)
    # eeg_samples = eeg_data['x_train_eeg'  ][sample_indexs]
    encoded_eeg = encoded_eeg_all[sample_indexs]
    encoded_labels = predicted_labels[sample_indexs]
    encoded_labels_type = predicted_labels_type[sample_indexs]
    # The labels of the digits that the generator tries to create an
    # image representation of
    sampled_labels = np.argmax(eeg_data['y_train'][sample_indexs],axis=1)
    sampled_labels_type = np.argmax(eeg_data['y_secondary_train'][sample_indexs],axis=1)


    # Select a random batch of REAL images with corresponding lables
    # from MNIST image data
    imgs = x_train[sample_indexs]
    # Image labels. 0-9
    # img_labels = np.argmax(y_train[sample_indexs], axis=1)
    # print(img_labels)
    


    #sampled_lables = to_categorical(sampled_labels,num_classes=len(class_labels),dtype=np.int32)
    # if args.ClassifierImplementation == "TF":
    #     encoded_eeg, encoded_labels = encoder_model.predict(eeg_samples)
    #     predicted_labels = np.argmax(encoded_labels,axis=1)
    # elif args.ClassifierImplementation == "Torch":
    #     with torch.no_grad():
    #         # print(eeg_samples.shape)
    #         eeg_samples = eeg_samples[:,np.newaxis,:,:]
    #         tensor_eeg  = torch.from_numpy(eeg_samples).to(device)
    #         encoded_labels, encoded_eeg  = encoder_model(tensor_eeg)
    #         predicted_labels = torch.argmax(encoded_labels, dim=1)

    #         encoded_eeg = encoded_eeg.cpu().numpy()
    #         predicted_labels = predicted_labels.cpu().numpy()
    #         encoded_eeg = tf.convert_to_tensor(encoded_eeg)
    #         predicted_labels = tf.convert_to_tensor(predicted_labels)

    # Generate a half batch of new images
    gen_imgs = generator.predict([encoded_eeg, encoded_labels, encoded_labels_type])

    # Train the discriminator, to recognise real/fake images
    # loss_real : using real images selected from training data
    # loss_fake : using images generated by the generator
    # {'loss': 3.244841694831848, 'Validity_loss': 0.8591426908969879, 'Class_Label_loss': 2.3856990337371826, 'Validity_accuracy': 0.421875, 'Class_Label_accuracy': 0.09375}
    if args.GAN_type == "CAPS":
        d_loss_real = discriminator.train_on_batch([imgs, y_train[sample_indexs]], [valid, sampled_labels, sampled_labels_type], return_dict=True)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, encoded_labels_all[sample_indexs]], [fake, encoded_labels, encoded_labels_type], return_dict=True)
        d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)
        g_loss = combined.train_on_batch([encoded_eeg, encoded_labels, encoded_labels_type, encoded_labels_all[sample_indexs]], [valid, encoded_labels, encoded_labels_type], return_dict=True)

    else:
        d_loss_real = discriminator.train_on_batch(imgs, [valid, sampled_labels, sampled_labels_type], return_dict=True)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, encoded_labels, encoded_labels_type], return_dict=True)

    
        d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)
        g_loss = combined.train_on_batch([encoded_eeg, encoded_labels, encoded_labels_type], [valid, encoded_labels, encoded_labels_type], return_dict=True)

    # ---------------------
    #  Train Generator:
    # ---------------------
    # Train the generator using the combined GAN model so that Generator learns to create better images to fool the discriminator
    #g_loss = combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels], return_dict=True)
    

    history['Discriminator'].append(d_loss)
    history['Generator'].append(g_loss)
    # Plot the progress
    print (f"Epoch {epoch:5d}: [D loss: {d_loss['loss']:.6f}, Validity acc.: {d_loss['Dis_Validity_accuracy']:.2%}, Label acc: {d_loss['Dis_Class_Label_accuracy']:.2%}, Label type acc: {d_loss['Dis_Class_type_Label_accuracy']:.2%}]")
    print(f"             [G loss: {g_loss['loss']:.6f}] [D loss: {g_loss['Discriminator_loss']:.6f}]")

    # If at save interval => save generated image samples
    if epoch % save_interval == 0 or epoch == epochs:
        save_model(generator, model_type, classifier_id, f"{model_type}_EEG_Generator_{epoch}", main_dir, model_save_path)
        save_model(discriminator, model_type, classifier_id, f"{model_type}_EEG_Discriminator_{epoch}", main_dir, model_save_path)
        sample_images_eeg(epoch, gen_imgs, [sampled_labels,encoded_labels], main_dir, model_save_path_imgs)



save_path = model_save_path

print(f"** Saving model to {save_path}")
with open(os.path.join(save_path,f"{model_type}_EEGGan_history.pkl"),"wb") as f:
    pickle.dump(history,f)

combined.save_weights(os.path.join(save_path,f"{model_type}_EEGGan_combined_weights.h5"))
generator.save_weights(os.path.join(save_path,f"{model_type}_EEGGan_generator_weights.h5"))
discriminator.save_weights(os.path.join(save_path,f"{model_type}_EEGGan_discriminator_weights.h5"))


pass