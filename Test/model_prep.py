import logging
import pdb

import torchvision.models
import torchvision.transforms as transformers
from DataSets_Deal import CIFAR_DATA

import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel,CLIPTokenizer

import torch.nn as nn
import torch
from guided_diffusion.unet import perpectual_Encoder_model
import torch.distributed as dist
import guided_diffusion.dist_util as dist_util

class pre_trained_models():
    def __init__(self,image_size = 32,dataset_name = None,model_name = "perpectual_Encoder_model"):
        # self.resnet_depth = resnet_depth
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.out_space_channels = 512
        self.model_name = model_name


    def Pre_train_Models(self):
        resnet = None
        # if self.resnet_depth == 50:
        #     resnet = torchvision.models.resnet50(pretrained=True)
        # elif self.resnet_depth ==101:
        #     resnet = torchvision.models.resnet101(pretrained=True)
        # resnet._modules['avgpool'] = nn.Identity()
        # resnet._modules['fc'] =nn.Linear(2 * self.image_size**2,1536*32*32)

        resnet,FE = self._Resnet_Num()

        with torch.no_grad():
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        models_dic = {}
        models_dic["resnet"] = resnet
        models_dic["clip_vision_model"] = model
        models_dic["image_processor"] = processor
        models_dic["FE"] = FE
        return models_dic

    def _Resnet_Num(self):
        resnet =None
        FE = None
        if self.dataset_name == "imagenet_50":
            with torch.no_grad():
                resnet = torchvision.models.resnet50(pretrained=True)
                resnet._modules['avgpool'] = nn.Identity()
                resnet._modules['fc'] =nn.Identity()

        elif self.dataset_name == "imagenet_101":
            resnet = torchvision.models.resnet101(pretrained=True)
            resnet._modules['avgpool'] = nn.Identity()
            resnet._modules['fc'] =nn.Linear(2 * self.image_size**2 ,1536*self.image_size*self.image_size)


        if "cifar10" == self.dataset_name:
            # Load model directly
            # from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            # with torch.no_grad():
            #     FE = AutoFeatureExtractor.from_pretrained("tzhao3/vit-CIFAR10")
            #     model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-CIFAR10")
            #     model._modules["classifier"] = nn.Identity()
            model = perpectual_Encoder_model(image_size=self.image_size,out_space_channels=self.out_space_channels)
            resnet = model

        # if "cifar100" == self.dataset_name :
        #     # Load model directly
        #     from transformers import AutoFeatureExtractor, AutoModelForImageClassification
        #     with torch.no_grad():
        #         FE = AutoFeatureExtractor.from_pretrained("Ahmed9275/Vit-Cifar100")
        #         model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
        #         model._modules["classifier"] = nn.Identity()
        #
        #     resnet = None

        if "lsun_church" == self.dataset_name:
            model = perpectual_Encoder_model(image_size=self.image_size,out_space_channels=self.out_space_channels)
            resnet = model

        if "lsun_bedroom" == self.dataset_name:
            model = perpectual_Encoder_model(image_size=self.image_size,out_space_channels=self.out_space_channels)
            resnet = model

        if "tinyimagenet" == self.dataset_name:
            model = perpectual_Encoder_model(image_size=self.image_size,out_space_channels=self.out_space_channels)
            resnet = model

        if "celeba" == self.dataset_name:
            model = perpectual_Encoder_model(image_size=self.image_size,out_space_channels=self.out_space_channels)
            resnet = model

        return resnet,FE

    def Pre_inference_models(self,steps = "020000",dir = None):

        resnet = self._infer_models(steps,dir)

        with torch.no_grad():
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        models_dic = {}
        models_dic["resnet"] = resnet
        models_dic["clip_vision_model"] = model
        models_dic["image_processor"] = processor
        models_dic["FE"] = None
        return models_dic


    def _infer_models(self,steps,dir):
        import blobfile as bf

        model = perpectual_Encoder_model(image_size=self.image_size, out_space_channels=self.out_space_channels)

        with bf.BlobFile(f"{dir}/_modelname__Encoder_{steps}.pt","rb") \
        as f:
            with torch.no_grad():
                state_dict = torch.load(f,map_location=dist_util.dev())
                model.load_state_dict(state_dict)

        return model
















