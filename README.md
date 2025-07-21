# BrainToImage

This is the GitHub Repository with all files relating to the master's dissertation, **"Image Reconstruction from brain activity using generative AI"**. There are four main folders present in this repository. (*) Folders indicate important files to use

1. BTI_Objects (*)
    - This folder contains code relating to
        - Dataset preprocessing Code
        - CNN, LSTM and Transformer Dataset Formatting
        - CNN and LSTM Encoder Training
        - ACGAN, DCGAN and CapsGAN Decoder Training
        - Model Evaluation for CNN and LSTM encoders and ACGAN, DCGAN and CapsGAN models
        - Cross-validation for CNN and LSTM encoders and ACGAN, DCGAN and CapsGAN models
          
2. BTI_Objects_ProgressiveGAN_dual (*)
  - Progressive GAN Decoder Training along with all relevant preprocessing codes
    
3. BTI_Objects_ProgressiveGAN_single
  - Older version of the Progressive GAN (ProGAN) model in which a single conditioning class was used
  - Model Evaluation for the ProGAN model
  
4. EEGViT_Personal (*)
  - Training code for the Vision Transformer model
  - Model Evaluation for the Vision Transformer model
  - Cross-validation for the Vision Transformer model
    


To recreate the models from the paper, the following folder orders should be used. 

<u> CNN - ACGAN (B, M and C) - 128 or 512</u>
BTI_Objects

<u> LSTM - DCGAN - 512 and LSTM - DCGAN - 512 </u>
BTI_Objects

<u> ViT - ProGAN - 512 </u>
BTI_Objects -> EEGViT_Personal -> BTI_Objects_ProgressiveGAN_dual
