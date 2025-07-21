This folder consists of all the codes relating to:
  - Dataset preprocessing Code
  - CNN, LSTM and Transformer Dataset Formatting
  - CNN and LSTM Encoder Training
  - ACGAN, DCGAN and CapsGAN Decoder Training
  - Model Evaluation for CNN and LSTM encoders and ACGAN, DCGAN and CapsGAN models
  - Cross-validation for CNN and LSTM encoders and ACGAN, DCGAN and CapsGAN models

To perform dataset preprocessing Take the following steps
  1. Create a folder raw_dataset
  2. Place the EEG Things folder and the Things Dataset into it. The folders should have the file paths raw_dataset/EEG_dataset for the Things EEG dataset and raw_dataset raw_dataset/object_dataset for the Images from the Things         dataset
     - Things EEG Dataset [link](https://osf.io/hd6zk/) : dataset [link](https://openneuro.org/datasets/ds003825/versions/1.2.0) 
     - Things Image Dataset [link](https://osf.io/jum2f/) (Make sure to go under the files section and extract the folder object_dataset from the zip file images_THINGS.zip)
  3. Run raw_to_pickle.py
  4. Run filter_data.py
  5. Run dataset_packaging_(model) where (model) is replaced with whatever encoder type model you want (CNN/ LSTM / ViT)
  6. Done!

 If running the ViT model take the Transformer dataset produced in processed_dataset/filter_mne_car and bring it to EEGViT_Personal folder
  
To perform CNN and LSTM Encoder Training
  1. Go to train folder
  2. Run train_encoderClassifier_dual.py with the relevant settings set in argparse
  3. Done!

To perform ACGAN, DCGAN and CapsGAN Training
  1. Go to train folder
  2. Run train_Decoder_dual.py with the relevant settings set in argparse
  3. Done!
