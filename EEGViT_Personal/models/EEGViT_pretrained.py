import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class EEGViT_pretrained(nn.Module):
    def __init__(self, num_of_classes):
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
                                    torch.nn.Linear(1000,512,bias=True),
                                    torch.nn.BatchNorm1d(512),
                                    torch.nn.Dropout(p=0.1))

        self.classification_head = torch.nn.Linear(512,num_of_classes,bias=True)

        # self.type_classification_head = torch.nn.Linear(512,num_of_types,bias=True)

        
    def forward(self,x):
        
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        latent = self.extract_latent(x)
        class_out = self.classification_head(latent)
        # class_type_out =  self.type_classification_head(latent)
        
        return class_out, latent


class EEGViT_pretrained_512(nn.Module):
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
                                    torch.nn.Linear(1000,512,bias=True),
                                    torch.nn.BatchNorm1d(512),
                                    torch.nn.Dropout(p=0.1))

        self.classification_head = torch.nn.Linear(512,num_of_classes,bias=True)

        self.type_classification_head = torch.nn.Linear(512,num_of_types,bias=True)

        
    def forward(self,x):
        
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        latent = self.extract_latent(x)
        class_out = self.classification_head(latent)
        class_type_out =  self.type_classification_head(latent)
        
        return class_out, latent, class_type_out

class EEGViT_pretrained_512_images(nn.Module):
    def __init__(self, num_of_classes, num_of_types):
        super().__init__()

        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 3})
        config.update({'image_size': (64,64)})
        config.update({'patch_size': (8,8)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(3, 768, kernel_size=8, stride=8, groups=3)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.BatchNorm1d(1000),
                                     torch.nn.Dropout(p=0.1)
                                     )
        self.ViT = model

        self.extract_latent = torch.nn.Sequential(
                                    torch.nn.Linear(1000,512,bias=True),
                                    torch.nn.BatchNorm1d(512),
                                    torch.nn.Dropout(p=0.1))

        self.classification_head = torch.nn.Linear(512,num_of_classes,bias=True)

        self.type_classification_head = torch.nn.Linear(512,num_of_types,bias=True)

        
    def forward(self,x):
        

        x=self.ViT.forward(x).logits
        latent = self.extract_latent(x)
        class_out = self.classification_head(latent)
        class_type_out =  self.type_classification_head(latent)
        
        return class_out, latent, class_type_out

class EEGViT_pretrained_512_experimental(nn.Module):
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

        model = transformers.ViTModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        # model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768, 512*14*14,bias=True),
        #                              torch.nn.BatchNorm1d(512*14*14),
        #                              torch.nn.Dropout(p=0.1)
        #                              )
        self.ViT = model

        # VGG covnersion
        self.project = nn.Conv2d(768, 512, kernel_size=1)  # reduce channel dim
        
        self.conv_vgg14 = nn.Sequential(
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            nn.BatchNorm2d(512),
            nn.ReLU()
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            # nn.ReLU(),
        ) #1 512 x 14 x 14
        self.conv_vgg7 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0, 1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            nn.BatchNorm2d(512),
            nn.ReLU()
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),  # refine
            # nn.ReLU(),
        ) #1 512 x 14 x 14

        # end of VGG conversions

        self.extract_latent = torch.nn.Sequential(
                    
                                    torch.nn.Linear(512*7*7,512,bias=True),
                                    torch.nn.BatchNorm1d(512),
                                    torch.nn.Dropout(p=0.1))

        self.classification_head = torch.nn.Linear(512,num_of_classes,bias=True)

        self.type_classification_head = torch.nn.Linear(512,num_of_types,bias=True)

        
    def forward(self,x):
        
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).last_hidden_state
        # size is  99, 768
        B, T, C = x.size()
        x = x.transpose(1,2)
        feature_map = x.view(B, C, 9, 11) 
        feature_map_14 = self.project(feature_map) 
        feature_map_14 = self.conv_vgg14(feature_map_14)
        feature_map_7 = self.conv_vgg7(feature_map_14)


        feature_map_flattened = torch.flatten(feature_map_7, start_dim=1) 
        #Creates the feature maps of size 512, 14, 14

        latent = self.extract_latent(feature_map_flattened)
        class_out = self.classification_head(latent)
        class_type_out =  self.type_classification_head(latent)
        
        return class_out, latent, class_type_out, feature_map_14, feature_map_7