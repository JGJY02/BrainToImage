import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class EEGViT_pretrained_dual(nn.Module):
    def __init__(self, num_of_classes, num_of_types):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 21),
            stride=(1, 1),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (63,14)})
        config.update({'patch_size': (8,1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.BatchNorm1d(1000),
                                     torch.nn.Dropout(p=0.1)
                                     )
        self.ViT = model

        self.extract_latent = torch.nn.Sequential(
                                    torch.nn.Linear(1000,128,bias=True),
                                    torch.nn.BatchNorm1d(128),
                                    torch.nn.Dropout(p=0.1))

        self.classification_head = torch.nn.Linear(128,num_of_classes,bias=True)

        self.type_classification_head = torch.nn.Linear(128,num_of_types,bias=True)

        
    def forward(self,x):
        
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        latent = self.extract_latent(x)
        class_out = self.classification_head(latent)
        class_type_out =  self.type_classification_head(latent)
        
        return class_out, latent, class_type_out