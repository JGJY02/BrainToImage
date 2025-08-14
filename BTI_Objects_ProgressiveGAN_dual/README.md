## Modified Progressive GAN Network based on the original works of [ProGAN](https://github.com/tkarras/progressive_growing_of_gans)

## Preparing datasets for training
The ProGAN model uses a different dataset style compared to the other GAN models used within this study. This is because ProGAN builds the model progressively. Hence the following steps should be taken to create the desired dataset. Specifically if we were to recreate the ViT-ProGAN Pipeline

1. Place the (model) dataset into file path datasets/processed_dataset/filter_mne_car/All
   (An example of the correct file path datasets/processed_dataset/filter_mne_car/All/000thresh_AllStack_Transformer_dual_2_64.pkl)
  
2. Place the pretrained (model) into trained_models/classifiers/All/000thresh  
   (This will be used to preprocess all signals first in case you want to test different architectures, An example trained_models/classifiers/All/000thresh/Transformer_dual_2_512/eeg_classifier_adm5_final.pth)

3. Run dataset_tool.py create_DualObjects [output_path] [input_file_path] [pretrained_network_path]

   (An example would be : python dataset_tool.py create_DualObjects datasets\processed_dataset\objects_transformer_dual_2_512_64 datasets\processed_dataset\filter_mne_car\All 000thresh_AllStack_Transformer_dual_2_64.pkl trained_models\classifiers\All\000thresh\Transformer_dual_2_512\eeg_classifier_adm5_final.pt)

4. Now you should have your ProGAN dataset saved under the processed_dataset which should include file ssuch as
   - tf.records (5 splits)
   - encodedLabels
   - encodedLabelsType
   - labels
   - signal

## Training networks

Once the necessary datasets are set up, you can proceed to train your own networks. The general procedure is as follows:

1. Edit `config.py` to specify the dataset and training configuration by uncommenting/editing specific lines.
2. Run the training script with `python train.py`.
3. The results are written into a newly created subdirectory under `config.result_dir`
4. Wait several days (or weeks) for the training to converge, and analyze the results.


## Analyzing results

Training results can be analyzed in several ways:

* **Metrics Evaluation**: This can be found under **eval/eval_gan_eeg_per_class_progressive.py** where it runs the full evaluation of the pipeline providing the metrics of both the encoder and decoder networks on the test data. Settings are configured under **config.py**
* **Cross Validation**: In order to run cross validation please first run the cross validation of the classifiers to obtain the specific indexes and pretrained models then setup as you would normally train a model as described in the training networks section
