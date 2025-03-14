import os
import pickle
import random
import sys

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from models.eegclassifier import convolutional_encoder_model, LSTM_Classifier
from models.eeggan import (build_discriminator, build_EEGgan, build_MoGCgenerator, build_MoGMgenerator, build_generator,
                           combine_loss_metrics, sample_images_eeg, save_model)
from utils.local_MNIST import get_balanced_mnist_subset, load_local_mnist
from utils.general_funcs_Jared import use_or_make_dir
import argparse

print(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
print(os.getcwd())

#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset", default = "processed_dataset/filter_mne_car",required=False)
parser.add_argument('--dataset_pickle', type=str, help="Dataset to use for training", default = "000thresh_AllStackLstm_sub-01.pkl" , required=False)

parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "sub-01",required=False)
parser.add_argument('--object_dataset_pickle', type=str, help="Dataset to use for training", default = "object_img_dataset_unsorted.pkl" , required=False)

parser.add_argument('--classifier_path', type=str, help="directory to the classifier", default= "trained_models/classifiers", required=False)
parser.add_argument('--classifier_model', type=str, help="Name of the model", default= "eeg_classifier_adm5", required=False)
parser.add_argument('--model_type', type=str, help="M,B,C", default= "C", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/GANs",required=False)
parser.add_argument('--GAN_Name', type=str, help="Directory to output", default = "trial1",required=False)


parser.add_argument('--classifierName', type = str, help = "auto_encoder or spectrogram_auto_encoder", default = "LSTM_all_stacked_signals")

parser.add_argument('--batch_size', type=int, help="Batch size", default = 32,required=False)
parser.add_argument('--epochs', type=int, help="Number of epochs to run", default = 5000,required=False)
parser.add_argument('--save_interval', type=int, help="how many epochs before saving", default = 250,required=False)


args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
save_interval = args.save_interval
generator_type = args.model_type #C for concatenation M for Multiplication B for Basic

class_labels = [0,1,2,3,4,5,6,7,8,9]
eeg_encoding_dim = 128




if generator_type == "B":
    model_type = "Basic"
else:
    model_type = f"MoG{generator_type}"


indexes = [i for i, char in enumerate(args.dataset_pickle) if char == '_']
run_id = args.dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
classifier_id = f"{run_id}_{args.epochs}_{args.classifierName}_{model_type}"

#Output save path name
model_save_path = f"{args.output_dir}/{run_id}_{model_type}"
model_save_path_imgs = f"{model_save_path}/imgs"

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

## load the MNIST trainig data
eeg_data_dir = f"{args.root_dir}/{args.input_dir}"
eeg_data_file = f"{eeg_data_dir}/{args.dataset_pickle}"
print(f"** Reading data file {eeg_data_file}")
object_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
(x_train, y_train) , (x_test, y_test) = (object_data['x_train_img'], object_data['y_train']) , (object_data['x_test_img'], object_data['y_test'])

x_train = (np.array(x_train) - 127.5) / 127.5
x_test = (np.array(x_test) - 127.5) / 127.5
print((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
## load the eeg training data

# dataset = "2022Data"

print(f"Reading data file {eeg_data_file}")
eeg_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
#x_train, y_train, x_test, y_test = eeg_data['x_train'], eeg_data['y_train'], eeg_data['x_test'  ], eeg_data['y_test'  ]
## ################
## Create GAN model
## ################
gan_optimizer = Adam(0.0002, 0.5, decay=1e-6)
discrim_losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']  #sparse_
gen_losses = ['categorical_crossentropy']
# build discriminator sub model
print("Shape of training is")
print((x_train.shape[1],x_train.shape[2],x_train.shape[3]))
discriminator = build_discriminator((x_train.shape[1],x_train.shape[2],x_train.shape[3]),len(class_labels))
discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
# build generator sub model

if generator_type == "C":
    generator = build_MoGCgenerator(eeg_encoding_dim,x_train.shape[3],len(class_labels))
elif generator_type == "M":
    generator = build_MoGMgenerator(eeg_encoding_dim,x_train.shape[3],len(class_labels))
elif generator_type == "B":
    generator = build_generator(eeg_encoding_dim,x_train.shape[3],len(class_labels))


generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])
# prime generator.
noise = Input(shape=(eeg_encoding_dim,))
label = Input(shape=(1,))
img = generator([noise, label])
# set discriminator used in combined model to none trainable.
discriminator.trainable = False
valid_class, target_label = discriminator(img)
# Create combined EEGGan model.
combined = build_EEGgan(eeg_encoding_dim, len(class_labels), generator, discriminator)
combined.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])

## #############
# EEG Classifier
## #############
classifier = LSTM_Classifier(eeg_data['x_train_eeg'].shape[1], eeg_data['x_train_eeg'].shape[2], len(class_labels))
#classifier_optimizer = Adam(learning_rate=0.0001, decay=1e-6)
#classifier.compile(loss='categorical_crossentropy', optimizer=classifier_optimizer, metrics=['accuracy'])
classifier_model_path = f"{args.classifier_path}/{args.input_dir}/{run_id}/{args.classifierName}/eeg_classifier_adm5_final.h5"
classifier.load_weights(classifier_model_path)
layer_names = ['EEG_feature_BN2','EEG_Class_Labels']
encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

## Set up with custom training loop
history = {'Discriminator':[],'Generator':[]}

print(f"** Classifier used: {classifier_model_path}")
print(f"** Training model for type: {model_type}")
for epoch in range(epochs+1):

    # ---------------------
    #  Train Discriminator: Discriminator is trained using real and generated images with the goal to identify the difference
    # ---------------------
    # Sample EEG latent space from EEG Classifier as generator input
    # _train_ run used eeg data from train to predict on, so the model had seen this data.
    # _test_ run should use data from test as the classifier hasn't seen this data before.
    sample_indexs = np.random.choice(eeg_data['x_train_eeg'  ].shape[0], size=batch_size, replace=False)
    eeg_samples = eeg_data['x_train_eeg'  ][sample_indexs]
    # The labels of the digits that the generator tries to create an
    # image representation of
    sampled_labels = np.argmax(eeg_data['y_train'][sample_indexs],axis=1)

    # Select a random batch of REAL images with corresponding lables
    # from MNIST image data
    imgs = x_train[sample_indexs]
    # Image labels. 0-9
    img_labels = np.argmax(y_train[sample_indexs], axis=1)
    # print(img_labels)
    


    #sampled_lables = to_categorical(sampled_labels,num_classes=len(class_labels),dtype=np.int32)
    encoded_eeg = encoder_model.predict(eeg_samples)
    predicted_labels = np.argmax(encoded_eeg[1],axis=1)
    # Generate a half batch of new images
    gen_imgs = generator.predict([encoded_eeg[0], predicted_labels])

    # Train the discriminator, to recognise real/fake images
    # loss_real : using real images selected from training data
    # loss_fake : using images generated by the generator
    # {'loss': 3.244841694831848, 'Validity_loss': 0.8591426908969879, 'Class_Label_loss': 2.3856990337371826, 'Validity_accuracy': 0.421875, 'Class_Label_accuracy': 0.09375}
    d_loss_real = discriminator.train_on_batch(imgs, [valid, img_labels], return_dict=True)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, predicted_labels], return_dict=True)
    #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator:
    # ---------------------
    # Train the generator using the combined GAN model so that Generator learns to create better images to fool the discriminator
    #g_loss = combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels], return_dict=True)
    g_loss = combined.train_on_batch([encoded_eeg[0], predicted_labels], [valid, predicted_labels], return_dict=True)

    history['Discriminator'].append(d_loss)
    history['Generator'].append(g_loss)
    # Plot the progress
    print (f"Epoch {epoch:5d}: [D loss: {d_loss['loss']:.6f}, Validity acc.: {d_loss['Dis_Validity_accuracy']:.2%}, Label acc: {d_loss['Dis_Class_Label_accuracy']:.2%}]")
    print(f"             [G loss: {g_loss['loss']:.6f}] [D loss: {g_loss['Discriminator_loss']:.6f}]")

    # If at save interval => save generated image samples
    if epoch % save_interval == 0 or epoch == epochs:
        save_model(generator, model_type, classifier_id, f"{model_type}_EEG_Generator_{epoch}", main_dir, model_save_path)
        save_model(discriminator, model_type, classifier_id, f"{model_type}_EEG_Discriminator_{epoch}", main_dir, model_save_path)
        sample_images_eeg(epoch, gen_imgs, [sampled_labels,predicted_labels], main_dir, model_save_path_imgs)



save_path = model_save_path

print(f"** Saving model to {save_path}")
with open(os.path.join(save_path,f"{model_type}_EEGGan_history.pkl"),"wb") as f:
    pickle.dump(history,f)

combined.save_weights(os.path.join(save_path,f"{model_type}_EEGGan_combined_weights.h5"))
generator.save_weights(os.path.join(save_path,f"{model_type}_EEGGan_generator_weights.h5"))
discriminator.save_weights(os.path.join(save_path,f"{model_type}_EEGGan_discriminator_weights.h5"))


pass