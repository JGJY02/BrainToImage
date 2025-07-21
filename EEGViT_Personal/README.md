## EEG-Vision Transformer (EEGViT)

Accepted KDD 2023: https://arxiv.org/pdf/2308.00454.pdf
## Overview
This is a modified version of the original EEGViT which can be found [here](https://github.com/ruiqiRichard/EEGViT)

## Dataset download
Dataset is obtained using the codes from BTI_Objects. The instrucitons on obtaining the dataset can be found there as well. Once obtained move the dataset into dataset

## Running Training
  1. run run_dualClass.py
  2. Done!

Once training is complete, ensure that the transformer model is moved into trained_models/classifiers/All/000thresh/

## Running Evaluation 
  1. Go to eval
  2. run **eval_gan_eeg_per_class_dual.py** for Most significant channel or run **obtain_loss_and_acc_plots.py** to obtain loss and accuracy plots of model
  3. Results can be found under results!

Evaluation of the transformer can be found under th BTI_Objects_ProgressiveGAN_dual as it evaluate the whole pipeline.


