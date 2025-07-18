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
from models.eegclassifier import convolutional_encoder_model_128_dual, LSTM_Classifier_dual_512, convolutional_encoder_model_512_dual
from models.dual_models.eeggan import (build_discriminator, build_EEGgan, build_MoGCgenerator, build_MoGMgenerator, build_generator)

from models.dual_models.dcgan import (build_dc_discriminator, build_DCGgan, build_dc_generator)
from models.dual_models.capsgan import (build_caps_discriminator, build_capsGAN, build_dccaps_generator)
from models.EEGViT_pretrained import (EEGViT_pretrained)

from models.model_utils import (sample_images_eeg, save_model, combine_loss_metrics)


from utils.local_MNIST import get_balanced_mnist_subset, load_local_mnist
from utils.general_funcs_Jared import use_or_make_dir
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from tensorflow.keras import backend as K
import gc

## new metrics
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score 
from skimage.metrics import peak_signal_noise_ratio as psnr

from image_similarity_measures.quality_metrics import fsim
from skimage.metrics import structural_similarity as ssim

import tensorflow as tf
import cv2
import torch
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms

import torch
import argparse
from matplotlib import pyplot as plt

def save_imgs(images, name, class_label, conditioning_labels, conditioning_type,  predicted_labels, pred_type, real_label, real_type, output_dir, label_dictionary):
    # Set up the grid dimensions (10x10)
    rows = 10
    cols = 10

    # Create a figure to display the grid of images
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    # Loop through the axes and images to display them

    for k, ax in enumerate(axes.flat):
        if k < len(images):
            ax.imshow(images[k])
            #C conditioning
            #P predicted
            #R True
            ax.set_title(f"C: {int(conditioning_labels[k])}, C_type: {int(conditioning_type[k])},\n P: {int(predicted_labels[k])},  P_type: {int(pred_type[k])},\n R: {int(real_label[k])}, R_type: {int(real_type[k])}", fontsize=8)
            
            ax.axis('off')  # Hide the axes
        else:
            ax.axis('off')  # In case there are empty subplots

        
    # Save the grid of images to a file
    output_path = f'{output_dir}/{name}_class{class_label}_{label_dictionary[class_label]}.png'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def compute_uiq(true, pred):
    true = true.astype(np.float32)
    pred = pred.astype(np.float32)

    values = []

    for i in range(true.shape[2]):

        true_cur = true[:,:,i]
        pred_cur = pred[:,:,i]
        # Mean values
        mu_x = np.mean(true_cur)
        mu_y = np.mean(pred_cur)

        # Variance and covariance
        sigma_x2 = np.var(true_cur)
        sigma_y2 = np.var(pred_cur)
        sigma_xy = np.cov(true_cur.flatten(), pred_cur.flatten())[0, 1]

        # Compute UIQ
        numerator = 4 * sigma_xy * mu_x * mu_y
        denominator = (sigma_x2 + sigma_y2) * (mu_x**2 + mu_y**2)

        UIQ_value = numerator / denominator if denominator != 0 else 1.0  # Avoid division by zero

        values.append(UIQ_value)

    return np.mean(values)


def compute_fsim(true, pred):
    true = true.astype(np.float32)
    pred = pred.astype(np.float32)

    values = []

    for i in range(true.shape[2]):
        true_cur = true[:,:,i]
        pred_cur = pred[:,:,i]
        fsim_value = fsim(true_cur,pred_cur)

        values.append(fsim_value)

    return np.mean(values)


def compute_ssim(true, pred):


    values = []

    for i in range(true.shape[2]):

        true_cur = true[:,:,i]
        pred_cur = pred[:,:,i]
        data_range = pred_cur.max() - pred_cur.min()

        ssim_value = ssim(true_cur,pred_cur, data_range = data_range)

        values.append(ssim_value)

    return np.mean(values)

print(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
print(os.getcwd())

#Jared Edition make sure we are back in the main directory to access all relevant files
main_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__)))) 
os.chdir(main_dir) #Jared Edition

parser = argparse.ArgumentParser(description="Process some variables.")
parser.add_argument('--root_dir', type=str, help="Directory to the dataset", default = "processed_dataset/filter_mne_car",required=False)
parser.add_argument('--input_dir', type=str, help="Directory to the dataset", default = "All",required=False)

parser.add_argument('--classifier_path', type=str, help="directory to the classifier", default= "trained_models/classifiers/crossVal", required=False)
parser.add_argument('--classifier_model', type=str, help="Name of the model", default= "eeg_classifier_adm5", required=False)
parser.add_argument('--output_dir', type=str, help="Directory to output", default = "trained_models/GANs/crossVal",required=False)

parser.add_argument('--ClassifierImplementation', type = str, help = "TF or Torch", default = "TF")
parser.add_argument('--batch_size', type=int, help="Batch size", default = 32,required=False)
parser.add_argument('--epochs', type=int, help="Number of epochs to run", default = 2000,required=False)
parser.add_argument('--save_interval', type=int, help="how many epochs before saving", default = 250,required=False)
parser.add_argument('--num_of_folds', type=int, help="Number of folds", default = 5 , required=False)

#Model specific settings (Tune These)
parser.add_argument('--latent_size', type=int, help="Size of the latent, 128 or 512", default = 512, required=False)
parser.add_argument('--classifierName', type = str, help = "CNN_all_stacked_signals_dual_128 or CNN_all_stacked_signals_dual_512_28_ori or LSTM_all_stacked_signals_dual_512_64_ori or Transformer_all_stacked_signals", default = "LSTM_all_stacked_signals_dual_512_64_ori")
parser.add_argument('--fold_indexes', type=str, help="Obtain indexes of fold", default = "trained_models/classifiers/crossVal/All/000thresh/LSTM_all_stacked_signals_dual_512_64_ori/saved_indexes.npy" , required=False)
parser.add_argument('--classifierType', type = str, help = "CNN or LSTM or Transformer", default = "LSTM")
parser.add_argument('--datasetType', type = str, help = "CNN_encoder or LSTM_encoder or Transformer_encoder", default = "LSTM_encoder")
parser.add_argument('--GAN_type', type=str, help="DC or AC or CAPS", default = "CAPS",required=False)
parser.add_argument('--model_type', type=str, help="M,B,C", default= "C", required=False)

args = parser.parse_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

batch_size = args.batch_size
epochs = args.epochs
save_interval = args.save_interval
generator_type = args.model_type #C for concatenation M for Multiplication B for Basic

index_dictionary = np.load(args.fold_indexes,allow_pickle = True).item()
# index_dictionary = npz_dictionary["arr_0"].item()
dataset_pickle = index_dictionary["dataset_pickle"]

print(dataset_pickle)
print(type(dataset_pickle))

class_labels = [0,1,2,3,4,5,6,7,8,9]
eeg_encoding_dim = args.latent_size

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

## load the Things-EEG trainig data
eeg_data_dir = f"{args.root_dir}/{args.datasetType}/{args.input_dir}"
eeg_data_file = f"{eeg_data_dir}/{dataset_pickle}"

print(f"** Reading data file {eeg_data_file}")
eeg_data = pickle.load(open(f"{eeg_data_file}", 'rb'), encoding='bytes')
train_imgs, test_imgs = eeg_data['x_train_img'] , eeg_data['x_test_img']

x_train_img = (np.array(train_imgs) - 127.5) / 127.5
x_test_img = (np.array(test_imgs) - 127.5) / 127.5
X_img = np.vstack((x_train_img, x_test_img))


x_train_eeg_data, y_primary_train_data, y_secondary_train_data, x_test_eeg_data, y_primary_test_data, y_secondary_test_data = eeg_data['x_train_eeg'], eeg_data['y_train'], eeg_data['y_secondary_train'], eeg_data['x_test_eeg'], eeg_data['y_test'], eeg_data['y_secondary_test']
label_dictionary = eeg_data['dictionary']
class_primary_labels = eeg_data['y_train'].shape[1]

X_eeg = np.vstack((x_train_eeg_data, x_test_eeg_data))

Y_primary = np.vstack((y_primary_train_data, y_primary_test_data))
Y_secondary = np.vstack((y_secondary_train_data, y_secondary_test_data))

# X_eeg = X_eeg[:100]
# Y_primary = Y_primary[:100]
# Y_secondary = Y_secondary[:100]
# X_img = X_img[:100]

Y_eeg = [f"{a}-{b}" for a, b in zip(Y_primary, Y_secondary)]  # or tuple: list(zip(y1, y2))


num_of_class_labels = Y_primary.shape[1]
num_of_class_type_labels = Y_secondary.shape[1]

if args.GAN_type == "AC":
    if generator_type == "B":
        model_type = "Basic"
    else:
        model_type = f"MoG{generator_type}"

else: model_type = args.GAN_type



indexes = [i for i, char in enumerate(dataset_pickle) if char == '_']
run_id = dataset_pickle[:indexes[0]] #"90thresh_"# "example_data_" #Extract the prefix to be used as the run id
classifier_id = f"{run_id}_{args.epochs}_{args.classifierName}_{model_type}"

model_save_path = f"{args.output_dir}/{args.classifierType}_GAN/{args.GAN_type}/{run_id}_{model_type}_{eeg_encoding_dim}"


previous_results = []
for i in range(args.num_of_folds):


    #Generate classifier path name
    if args.ClassifierImplementation == "TF":
        classifier_model_path = f"{args.classifier_path}/{args.input_dir}/{run_id}/{args.classifierName}/fold_{i}/eeg_classifier_adm5_final.h5"
    elif args.ClassifierImplementation == "Torch":
        classifier_model_path = f"{args.classifier_path}/{args.input_dir}/{run_id}/{args.classifierName}/fold_{i}/eeg_classifier_adm5_final.pth"
        if torch.cuda.is_available():
            gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

    else:
        raise FileNotFoundError(f"{args.ClassifierImplementation} is not a valid implementation")

    print("Taking Classifier from : ", classifier_model_path)

    #Clear resources after each fold
    train_index = index_dictionary[i]["train_idx"]
    test_index = index_dictionary[i]["val_idx"]

    K.clear_session()
    gc.collect()
    print(f"Current Fold is {i}")


    #Output save path name
    model_save_dir = os.path.join(model_save_path, f"fold{i}")
    model_save_path_imgs = f"{model_save_dir}/imgs"

    check_path(model_save_dir)

    x_eeg_train = X_eeg[train_index]
    y_eeg_train = Y_primary[train_index]
    y_eeg_secondary_train = Y_secondary[train_index]
    
    x_eeg_test = X_eeg[test_index]
    y_eeg_test = Y_primary[test_index]
    y_eeg_secondary_test = Y_secondary[test_index]

    x_img_train = X_img[train_index]
    x_img_test = X_img[test_index]


    ## ################
    ## Create GAN model
    ## ################
    gan_optimizer = Adam(0.0002, 0.5, decay=1e-6)
    discrim_losses = ['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']  #sparse_
    gen_losses = ['mse']

    # build discriminator sub model
    print("Shape of training is")
    print((x_img_train.shape[1],x_img_train.shape[2],x_img_train.shape[3]))

    print(f"** Training model for type: {args.GAN_type}")
    if args.GAN_type == "AC":
        print(f"*** Training sub model for type: {args.model_type}")

        discriminator = build_discriminator((x_img_train.shape[1],x_img_train.shape[2],x_img_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
        discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
        # build generator sub model

        if generator_type == "C":
            generator = build_MoGCgenerator(eeg_encoding_dim, x_img_train.shape[3],num_of_class_labels, num_of_class_type_labels)
        elif generator_type == "M":
            generator = build_MoGMgenerator(eeg_encoding_dim, x_img_train.shape[3],num_of_class_labels, num_of_class_type_labels)
        elif generator_type == "B":
            generator = build_generator(eeg_encoding_dim,    x_img_train.shape[3],num_of_class_labels, num_of_class_type_labels)

        generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])
    elif args.GAN_type == "DC":
        discriminator = build_dc_discriminator((x_img_train.shape[1],x_img_train.shape[2],x_img_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
        discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
        generator = build_dc_generator(eeg_encoding_dim, x_img_train.shape[3],num_of_class_labels, num_of_class_type_labels)
        generator.compile(loss=gen_losses, optimizer=gan_optimizer, metrics=['accuracy'])

    elif args.GAN_type == "CAPS":

        discriminator, val_discriminator, _ = build_caps_discriminator((x_img_train.shape[1],x_img_train.shape[2],x_img_train.shape[3]),num_of_class_labels, num_of_class_type_labels)
        discriminator.compile(loss=discrim_losses, optimizer=gan_optimizer, metrics=['accuracy'])
        generator = build_dccaps_generator(eeg_encoding_dim, x_img_train.shape[3], num_of_class_labels, num_of_class_type_labels)
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
            print(x_eeg_train.shape[1], x_eeg_train.shape[2])
            classifier = LSTM_Classifier_dual_512(x_eeg_train.shape[1],  x_eeg_train.shape[2], 512, num_of_class_labels, num_of_class_type_labels)

        elif args.classifierType == "CNN":
            print(x_eeg_train.shape[1], x_eeg_train.shape[2])
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
            # print(eeg_samples.shape)
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
        # _train_ run used eeg data from train to predict on, so the model had seen this data.
        # _test_ run should use data from test as the classifier hasn't seen this data before.
        sample_indexs = np.random.choice(x_eeg_train.shape[0], size=batch_size, replace=False)
        # eeg_samples = eeg_data['x_train_eeg'  ][sample_indexs]
        encoded_eeg = encoded_eeg_all[sample_indexs]
        encoded_labels = predicted_labels[sample_indexs]
        encoded_labels_type = predicted_labels_type[sample_indexs]
        # The labels of the digits that the generator tries to create an
        # image representation of
        sampled_labels = np.argmax(y_eeg_train[sample_indexs],axis=1)
        sampled_labels_type = np.argmax(y_eeg_secondary_train[sample_indexs],axis=1)


        # Select a random batch of REAL images with corresponding lables
        # from MNIST image data
        imgs = x_img_train[sample_indexs]

        # Generate a half batch of new images
        gen_imgs = generator.predict([encoded_eeg, encoded_labels, encoded_labels_type])

        if args.GAN_type == "CAPS":
            d_loss_real = discriminator.train_on_batch([imgs, y_eeg_train[sample_indexs]], [valid, sampled_labels, sampled_labels_type], return_dict=True)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, encoded_labels_all[sample_indexs]], [fake, encoded_labels, encoded_labels_type], return_dict=True)
            d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)
            
            g_loss_disc = combined.train_on_batch([encoded_eeg, encoded_labels, encoded_labels_type, encoded_labels_all[sample_indexs]], [valid, encoded_labels, encoded_labels_type], return_dict=True)
            g_loss_mse = generator.train_on_batch([encoded_eeg, encoded_labels, encoded_labels_type], imgs, return_dict = True)
            
            g_loss = {'loss' : g_loss_disc['loss'] + g_loss_mse['loss'], 'Discriminator_loss' : g_loss_disc['Discriminator_loss'], 'MSE_loss': g_loss_mse['loss']}


        else:
            d_loss_real = discriminator.train_on_batch(imgs, [valid, sampled_labels, sampled_labels_type], return_dict=True)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, encoded_labels, encoded_labels_type], return_dict=True)
            d_loss = combine_loss_metrics(d_loss_real, d_loss_fake)

            g_loss_disc = combined.train_on_batch([encoded_eeg, encoded_labels, encoded_labels_type], [valid, encoded_labels, encoded_labels_type], return_dict=True)
            g_loss_mse = generator.train_on_batch([encoded_eeg, encoded_labels, encoded_labels_type], imgs, return_dict = True)

            g_loss = {'loss' : g_loss_disc['loss'] + g_loss_mse['loss'], 'Discriminator_loss' : g_loss_disc['Discriminator_loss'], 'MSE_loss': g_loss_mse['loss']}

        # ---------------------
        #  Train Generator:
        # ---------------------        
        history['Discriminator'].append(d_loss)
        history['Generator'].append(g_loss)
        # Plot the progress
        print (f"Epoch {epoch:5d}: [D loss: {d_loss['loss']:.6f}, Validity acc.: {d_loss['Dis_Validity_accuracy']:.2%}, Label acc: {d_loss['Dis_Class_Label_accuracy']:.2%}, Label type acc: {d_loss['Dis_Class_type_Label_accuracy']:.2%}]")
        print(f"             [G loss: {g_loss['loss']:.6f}] [D loss: {g_loss['Discriminator_loss']:.6f}] [MSE loss: {g_loss['MSE_loss']:.6f}]")

        # If at save interval => save generated image samples
        # if epoch % save_interval == 0 or epoch == epochs:
            # save_model(generator, model_type, classifier_id, f"{model_type}_EEG_Generator_{epoch}", main_dir, model_save_dir)
            # save_model(discriminator, model_type, classifier_id, f"{model_type}_EEG_Discriminator_{epoch}", main_dir, model_save_dir)
            # sample_images_eeg(epoch, gen_imgs, [sampled_labels,encoded_labels], main_dir, model_save_path_imgs)


    print(f"** Saving model to {model_save_dir}")

    # combined.save_weights(os.path.join(model_save_dir,f"{model_type}_EEGGan_combined_weights.h5"))
    # generator.save_weights(os.path.join(model_save_dir,f"{model_type}_EEGGan_generator_weights.h5"))
    # discriminator.save_weights(os.path.join(model_save_dir,f"{model_type}_EEGGan_discriminator_weights.h5"))

    ##Rebuild the GAN model. This is to account for capsGAN having a specific train and valid model

    if args.GAN_type == "AC":
        print(f"*** Evaluating for sub model for type: {args.model_type}")

        discriminator_val = build_discriminator((x_img_test.shape[1],x_img_test.shape[2],x_img_test.shape[3]),num_of_class_labels, num_of_class_type_labels)
        # build generator sub model

        if generator_type == "C":
            generator_val = build_MoGCgenerator(eeg_encoding_dim,x_img_test.shape[3],num_of_class_labels, num_of_class_type_labels)
        elif generator_type == "M":
            generator_val = build_MoGMgenerator(eeg_encoding_dim,x_img_test.shape[3],num_of_class_labels, num_of_class_type_labels)
        elif generator_type == "B":
            generator_val = build_generator(eeg_encoding_dim,    x_img_test.shape[3],num_of_class_labels, num_of_class_type_labels)

    elif args.GAN_type == "DC":
        discriminator_val = build_dc_discriminator((x_img_test.shape[1],x_img_test.shape[2],x_img_test.shape[3]),num_of_class_labels, num_of_class_type_labels)
        generator_val = build_dc_generator(eeg_encoding_dim, x_img_test.shape[3],num_of_class_labels, num_of_class_type_labels)

    elif args.GAN_type == "CAPS":

        _, discriminator_val, _ = build_caps_discriminator((x_img_test.shape[1],x_img_test.shape[2],x_img_test.shape[3]),num_of_class_labels, num_of_class_type_labels)
        generator_val = build_dccaps_generator(eeg_encoding_dim, x_img_test.shape[3], num_of_class_labels, num_of_class_type_labels)


    # Create combined EEGGan model.

    discriminator_val.set_weights(discriminator.get_weights())
    generator_val.set_weights(generator.get_weights())
    combined_val = build_EEGgan(eeg_encoding_dim,generator_val,discriminator_val)
    combined_val.set_weights(combined.get_weights())


    # generator.load_weights(os.path.join(model_save_dir,f"{model_type}_EEGGan_generator_weights.h5"))
    # combined = build_EEGgan(eeg_encoding_dim,generator,discriminator)
    # combined.load_weights(os.path.join(model_save_dir,f"{model_type}_EEGGan_combined_weights.h5"))

    ##End of it
    layer_names = ['EEG_feature_BN2','EEG_Class_Labels','EEG_Class_type_Labels']
    encoder_outputs = [classifier.get_layer(layer_name).output for layer_name in layer_names]
    encoder_model = Model(inputs=classifier.input, outputs=encoder_outputs)

    dataset = tf.data.Dataset.from_tensor_slices(x_eeg_test)
    dataset = dataset.batch(batch_size)

    encoded_latents = []
    encoded_labels = []
    encoded_type_labels = []

    for batch in tqdm(dataset):
        encodedEEG, encodedLabel, encodedTypeLabels = encoder_model(batch, training=False)
        encoded_latents.append(encodedEEG)
        encoded_labels.append(encodedLabel)
        encoded_type_labels.append(encodedTypeLabels)

    # Combine results
    encoded_latents = np.concatenate(encoded_latents, axis=0)
    encoded_labels = np.concatenate(encoded_labels, axis=0)
    encoded_type_labels = np.concatenate(encoded_type_labels, axis =0)
    to_labels = np.argmax(y_eeg_test,axis=1)  ## since eeg labels are in one-hot encoded format


    history = {}
    for labs in class_labels:  ## outer loop per class
        print("Current class label is : ", labs)
        ## get all EEG data for class i
        matching_indices = np.where(to_labels == labs)
        true_images = x_img_test[matching_indices[0]]
        labels = y_eeg_test[matching_indices[0]]

        encoded_eegs = encoded_latents[matching_indices[0]]
        conditioning_labels_raw = encoded_labels[matching_indices[0]]
        
        conditioning_labels_type_raw = encoded_type_labels[matching_indices[0]]
        true_conditioning_labels_type = y_eeg_secondary_test[matching_indices[0]]

            
        # minibatch_size = np.clip(8192 // encoded_eegs.shape[1], 4, 256)
        minibatch_size = 4
        # print("The encoded eegs shape is :", encoded_eegs.shape)
        # print("The calculated minibatch size is :", minibatch_size)
        conditioning_labels_argmax = np.argmax(conditioning_labels_raw, axis=1)
        conditioning_labels_type_argmax = np.argmax(conditioning_labels_type_raw, axis = 1)

        generated_samples = generator_val.predict([encoded_eegs, conditioning_labels_argmax, conditioning_labels_type_argmax],batch_size=32)
        ## predict on GAN
        validitys, labels_pred, labels_type_pred = combined_val.predict([encoded_eegs, conditioning_labels_argmax, conditioning_labels_type_argmax],batch_size=32)
        
        ## Prediction on original images

        validitys_true, labels_true_pred, labels_type_true_pred  = discriminator_val.predict([true_images], batch_size=32)

        generated_samples = generated_samples*127.5 + 127.5
        generated_samples = np.clip(generated_samples,0,255)
        generated_samples = generated_samples.astype(np.uint8)

        true_samples = true_images*127.5 + 127.5
        true_samples = np.clip(true_samples,0,255)
        true_samples = true_samples.astype(np.uint8)

        ## collate results
        history[labs] = {'generated':generated_samples,'true':true_samples,'valid':validitys,'predicted':labels_pred,'conditioning':conditioning_labels_raw, 'true_labels':labels, \
            'predicted_type': labels_type_pred,'conditioning_type': conditioning_labels_type_raw, 'true_type': true_conditioning_labels_type, \
            'labels_true_pred' : labels_true_pred, 'labels_type_true_pred': labels_type_true_pred}


    inception = InceptionScore()

    comparison_imgs = []

    all_generated_images = []
    all_true_images = []

    mean_ssim_scores = []
    mean_rmse_scores = []
    mean_psnr_scores =[]
    mean_fsim_scores = []
    mean_uiq_scores = []
    mean_inception_scores = []
    mean_inception_stds = []

    mean_real_inception_scores = []
    mean_real_inception_stds = []

    mean_accuracy_scores = []
    mean_accuracy_type_scores = []

    mean_precision_scores = []
    mean_recall_scores = []
    mean_F1_scores =[]

    mean_type_precision_scores = []
    mean_type_recall_scores = []
    mean_type_F1_scores =[]

    num_of_sample_images = 10

    sample_img_per_class = []
    list_of_labels = []
    text_to_save = []

    evaluation ={}


    # Transform image for inception
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),        # InceptionV3 expects 299x299
        transforms.PILToTensor(),                # Converts to [0, 1] float tensor and rearranges to (C, H, W)
    ])

    for labs in class_labels:
        class_data = history[labs]

        classes_added = []
        types_added = []

        ssim_scores = []
        rmse_scores = []
        psnr_scores =[]
        fsim_scores = []
        uiq_scores = []
        inception_scores = []

        precision_scores = []
        recall_scores = []
        F1_scores = []

        precision_type_scores = []
        recall_type_scores = []
        F1_type_scores = []

        true_positives = 0
        true_positives_type = 0

        #predict for generated labels
        conditioning_labels_array = np.argmax(class_data['conditioning'], axis = 1)
        predicted_labels_array = np.argmax(class_data['predicted'], axis = 1)
        true_labels_array = np.argmax(class_data['true_labels'], axis = 1)

        #For label type
        conditioning_labels_type_array = np.argmax(class_data['conditioning_type'], axis = 1)
        pred_labels_type_array = np.argmax(class_data['predicted_type'], axis = 1)
        true_labels_type_array = np.argmax(class_data['true_type'], axis = 1)

        #For Discriminator Prediction on Real images
        pred_real_labels_array = np.argmax(class_data['labels_true_pred'], axis = 1)
        pred_real_labels_type_array = np.argmax(class_data['labels_type_true_pred'], axis = 1)


        Index_of_relevant_classes = np.where(conditioning_labels_array == i)[0]
        sample_img_per_class.append(class_data['generated'][Index_of_relevant_classes[:num_of_sample_images]])

        labels_array = np.ones(num_of_sample_images)*labs
        list_of_labels.append(labels_array)

        # fin_sampling = os.path.join(model_save_dir, "sampling")
        # print("Sampling real images")
        # save_imgs(class_data['true'], "Real", i, true_labels_array, true_labels_type_array, \
        #     pred_real_labels_array, pred_real_labels_type_array, \
        #     true_labels_array, true_labels_type_array , fin_sampling, label_dictionary)

        # print("Sampling generated images")
        # save_imgs(class_data['generated'], "Generated", i ,conditioning_labels_array, conditioning_labels_type_array, \
        #     predicted_labels_array, pred_labels_type_array, \
        #     true_labels_array, true_labels_type_array, fin_sampling, label_dictionary)

        temp_hold_imgs = []
        sampling_imgs_taken = False
        last = 0
        for j in range(class_data['generated'].shape[0]):
            if labs == np.argmax(class_data['conditioning'][j]):
                true_positives += 1

            if np.argmax(class_data['true_type'][j]) == np.argmax(class_data['conditioning_type'][j]):
                true_positives_type += 1

            y_true = class_data['true'][j][:,:,:]
            y_pred = class_data['generated'][j][:,:,:]
            all_true_images.append(y_true)

            all_generated_images.append(y_pred)


            if labs not in classes_added:
                # print("Hello")
                # if j not in types_added:
                    # print("In here")
                    
                if  np.argmax(class_data['true_type'][j]) == last:
                    last += 1

                    # print("Now adding images")
                    temp_hold_imgs.append([y_true, y_pred])
                    types_added.append(j)
                if len(types_added) == eeg_data['y_secondary_test'].shape[1] and not sampling_imgs_taken:

                    classes_added.append(i)
                    comparison_imgs.append(temp_hold_imgs)
                    # print(len(temp_hold_imgs))
                    sampling_imgs_taken = True                          

            #SSIM score
            ssim_value = compute_ssim(y_true, y_pred)
            
            #RMSE
            rmse_value = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
            
            #PSNR value
            psnr_value = psnr(y_true, y_pred, data_range = 255)

            #FSIM
            fsim_value = fsim(y_true, y_pred)

            #UIQ 
            uiq_value = compute_uiq(y_true, y_pred)




            #Append all scores
            ssim_scores.append(ssim_value)
            rmse_scores.append(rmse_value)
            psnr_scores.append(psnr_value)
            fsim_scores.append(fsim_value)
            uiq_scores.append(uiq_value)



        # Convert all images
        tensor_images = torch.stack([transform(img) for img in class_data['generated']])
        inception_value = inception(tensor_images)

        tensor_images_real = torch.stack([transform(img) for img in class_data['true']])
        inception_value_real = inception(tensor_images_real)
        
        #F1
        F1_value = f1_score(true_labels_array, conditioning_labels_array, average='macro')
        F1_type_value = f1_score(true_labels_type_array, conditioning_labels_type_array, average='macro')


        #Recall
        recall_value = recall_score(true_labels_array, conditioning_labels_array, average='macro')
        recall_type_value = recall_score(true_labels_type_array, conditioning_labels_type_array, average='macro')

        #Precision
        precision_value = precision_score(true_labels_array,conditioning_labels_array, average='macro')
        precision_type_value = precision_score(true_labels_type_array,conditioning_labels_type_array, average='macro')
            
        class_acc = true_positives / class_data['generated'].shape[0]
        class_type_acc = true_positives_type / class_data['generated'].shape[0]

        evaluation[labs] = {'average_ssim': np.mean(ssim_scores),'average_rmse':np.mean(rmse_scores),'average_psnr':np.mean(psnr_scores),'average_fsim':np.mean(fsim_scores),'average_uiq':np.mean(uiq_scores), \
            'average_inception':inception_value[0], 'average_inception_std':inception_value[1],\
            'average_real_inception': inception_value_real[0], 'average_real_inception_std':inception_value_real[1], \
            'class_accuracy':class_acc, 'class_type_accuracy': class_type_acc, \
            'average_F1':F1_value,'average_recall':recall_value,'average_precision':precision_value,\
            'average_type_F1':F1_type_value,'average_type_recall':recall_type_value,'average_type_precision':precision_type_value}


        text_to_print = f"Class {labs} ({label_dictionary[labs]}): mean ssim: {evaluation[labs]['average_ssim']:.2f}, mean rmse: {evaluation[labs]['average_rmse']:.2f}, mean psnr: {evaluation[labs]['average_psnr']:.2f},  \
            mean fsim: {evaluation[labs]['average_fsim']:.2f}, mean uiq: {evaluation[labs]['average_uiq']:.2f}, Fake Inception: {evaluation[labs]['average_inception']:.3f} ~ inception std {evaluation[labs]['average_inception_std']:.3f}, Real Inception: {evaluation[labs]['average_real_inception']:.3f} ~ inception std {evaluation[labs]['average_real_inception_std']:.3f} ,classification acc: {class_acc:.1%}, classification type acc: {class_type_acc:.1%}, \n \
            mean F1 {evaluation[labs]['average_F1']:.2f}, mean recall {evaluation[labs]['average_recall']:.2f}, mean precision {evaluation[labs]['average_precision']:.2f} , \n \
            mean type F1 {evaluation[labs]['average_type_F1']:.2f}, mean type recall {evaluation[labs]['average_type_recall']:.2f}, mean type precision {evaluation[labs]['average_type_precision']:.2f} \n   "
        text_to_save.append(text_to_print)
        print(text_to_print)


        mean_ssim_scores.append(evaluation[labs]['average_ssim'])
        mean_rmse_scores.append(evaluation[labs]['average_rmse'])
        mean_psnr_scores.append(evaluation[labs]['average_psnr'])
        mean_fsim_scores.append(evaluation[labs]['average_fsim'])
        mean_uiq_scores.append(evaluation[labs]['average_uiq'])
        mean_inception_scores.append(evaluation[labs]['average_inception'])
        mean_inception_stds.append(evaluation[labs]['average_inception_std'])
        mean_real_inception_scores.append(evaluation[labs]['average_real_inception'])
        mean_real_inception_stds.append(evaluation[labs]['average_real_inception_std'])

        mean_accuracy_scores.append(class_acc)
        mean_accuracy_type_scores.append(class_type_acc)

        mean_F1_scores.append(evaluation[labs]['average_F1'])
        mean_recall_scores.append(evaluation[labs]['average_recall'])
        mean_precision_scores.append(evaluation[labs]['average_precision'])

        mean_type_F1_scores.append(evaluation[labs]['average_type_F1'])
        mean_type_recall_scores.append(evaluation[labs]['average_type_recall'])
        mean_type_precision_scores.append(evaluation[labs]['average_type_precision'])


        ##Inception on whole dataset
    all_tensor_images = torch.stack([transform(img) for img in all_generated_images])
    inception_value = inception(all_tensor_images)

    all_real_tensor_images = torch.stack([transform(img) for img in all_true_images])
    real_inception_value = inception(all_real_tensor_images)

    mean_evaluation = {'average_ssim':np.mean(mean_ssim_scores),'average_rmse':np.mean(mean_rmse_scores),'average_psnr':np.mean(mean_psnr_scores),'average_fsim':np.mean(mean_fsim_scores),'average_uiq':np.mean(mean_uiq_scores), 'average_inception':np.mean(mean_inception_scores), 'average_stds': np.mean(mean_inception_stds),\
        'average_real_inception':np.mean(mean_real_inception_scores), 'average_real_stds': np.mean(mean_real_inception_stds), \
        'overall_fake_inception': inception_value[0], 'overall_fake_std': inception_value[1], "overall_real_inception": real_inception_value[0], "overall_real_stds": real_inception_value[1], \
        'average_accuracy':np.mean(mean_accuracy_scores), 'average_type_accuracy': np.mean(mean_accuracy_type_scores), \
        'average_f1' : np.mean(mean_F1_scores), 'average_recall' : np.mean(mean_recall_scores), 'average_precision' : np.mean(mean_precision_scores), \
        'average_type_f1' : np.mean(mean_type_F1_scores), 'average_type_recall' : np.mean(mean_type_recall_scores), 'average_type_precision' : np.mean(mean_type_precision_scores), 
            }




    mean_text_to_print = f"Average Class Results: mean ssim: {mean_evaluation['average_ssim']:.2f}, mean rmse: {mean_evaluation['average_rmse']:.2f}, mean psnr: {mean_evaluation['average_psnr']:.2f}, \
        mean fsim: {mean_evaluation['average_fsim']:.2f}, mean uiq: {mean_evaluation['average_uiq']:.2f}, mean classification acc: {mean_evaluation['average_accuracy']:.1%} ,mean type classification acc: {mean_evaluation['average_type_accuracy']:.1%} \n \
        Fake mean inception: {mean_evaluation['average_inception']:.2f} ~ mean stds: {mean_evaluation['average_stds']:.2f} | Real mean inception: {mean_evaluation['average_real_inception']:.2f} ~ mean stds: {mean_evaluation['average_real_stds']:.2f} | Overall Fake inception: {inception_value[0]:.2f} ~ Overall stds: {inception_value[1]:.2f} | Overall Real inception: {real_inception_value[0]:.2f} ~ Overall stds: {real_inception_value[1]:.2f}\n \
        mean F1: {mean_evaluation['average_f1']:.2f}, mean recall: {mean_evaluation['average_recall']:.2f}, mean precision: {mean_evaluation['average_precision']:.2f} \n \
        mean type F1: {mean_evaluation['average_type_f1']:.2f}, mean type recall: {mean_evaluation['average_type_recall']:.2f}, mean type precision: {mean_evaluation['average_type_precision']:.2f} \n        "
    print(mean_text_to_print)
    text_to_save.append(mean_text_to_print)


    ## Save fold results
    with open(f"{model_save_dir}/results.txt", "w") as file:
        file.write("\n".join(text_to_save) + "\n")
    
    with open(f"{model_save_dir}/fold_indices.txt", "w") as file:
        file.write(f"Dataset used {eeg_data_file}\n\n")
        file.write(f"Fold {i}\n\n")
        file.write(f"Train indices: {train_index.tolist()}\n\n")
        file.write(f"Val indices:   {test_index.tolist()}\n\n")

    previous_results.append([evaluation, mean_evaluation])

## Summarise folds
mean_per_class = np.zeros((10, len(evaluation[0])))
mean_fold = np.zeros((1,len(mean_evaluation)))
for fold in range(args.num_of_folds):
    cur_perClass_dict, cur_mean_dict = previous_results[fold]
    
    rows = []
    for lab in range(class_primary_labels):
        vals = np.array(list(cur_perClass_dict[lab].values()))
        rows.append(vals)
    
    result = np.vstack(rows)

    mean_result = np.array(list(cur_mean_dict.values()))
    
    # print(result)
    # print(result.shape)
    mean_per_class = result + mean_per_class
    mean_fold = mean_result + mean_fold
    # print(mean_fold)
    # print(mean_fold.shape)

mean_per_class /= args.num_of_folds
mean_fold /= args.num_of_folds

text_to_save = []

# print(mean_per_class)
# print(mean_per_class.shape)

# print(mean_fold)
# print(mean_fold.shape)
for lab in range(class_primary_labels):
    
    mean_class = mean_per_class[lab, :]    
    text_to_print = f"Class {lab} ({label_dictionary[lab]}): mean ssim: {mean_class[0]:.2f}, mean rmse: {mean_class[1]:.2f}, mean psnr: {mean_class[2]:.2f},  \
            mean fsim: {mean_class[3]:.2f}, mean uiq: {mean_class[4]:.2f}, Fake Inception: {mean_class[5]:.3f} ~ inception std {mean_class[6]:.3f}, Real Inception: {mean_class[7]:.3f} ~ inception std {mean_class[8]:.3f} ,classification acc: {mean_class[9]:.1%}, classification type acc: {mean_class[10]:.1%}, \n \
            mean F1 {mean_class[11]:.2f}, mean recall {mean_class[12]:.2f}, mean precision {mean_class[13]:.2f} , \n \
            mean type F1 {mean_class[14]:.2f}, mean type recall {mean_class[15]:.2f}, mean type precision {mean_class[16]:.2f} \n   "

    
    text_to_save.append(text_to_print)
    print(text_to_print)

mean_fold_extract = mean_fold[0,:]   
mean_text_to_print = f"Average Class Results: mean ssim: {mean_fold_extract[0]:.2f}, mean rmse: {mean_fold_extract[1]:.2f}, mean psnr: {mean_fold_extract[2]:.2f}, \
        mean fsim: {mean_fold_extract[3]:.2f}, mean uiq: {mean_fold_extract[4]:.2f} \n \
        Fake mean inception: {mean_fold_extract[5]:.2f} ~ mean stds: {mean_fold_extract[6]:.2f} | Real mean inception: {mean_fold_extract[7]:.2f} ~ mean stds: {mean_fold_extract[8]:.2f} | Overall Fake inception: {mean_fold_extract[9]:.2f} ~ Overall stds: {mean_fold_extract[10]:.2f} | Overall Real inception: {mean_fold_extract[11]:.2f} ~ Overall stds: {mean_fold_extract[12]:.2f}\n \
        mean classification acc: {mean_fold_extract[13]:.1%} ,mean type classification acc: {mean_fold_extract[14]:.1%} \n \
        mean F1: {mean_fold_extract[15]:.2f}, mean recall: {mean_fold_extract[16]:.2f}, mean precision: {mean_fold_extract[17]:.2f} \n \
        mean type F1: {mean_fold_extract[18]:.2f}, mean type recall: {mean_fold_extract[19]:.2f}, mean type precision: {mean_fold_extract[20]:.2f} \n        "
    

text_to_save.append(mean_text_to_print)
print(mean_text_to_print)  

## Save fold results
with open(f"{model_save_path}/results.txt", "w") as file:
    file.write("\n".join(text_to_save) + "\n")