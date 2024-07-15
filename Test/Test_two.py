import logging
import pdb

import torch
import torchvision.models
import torchvision.transforms as transformers
from DataSets_Deal import CIFAR_DATA

import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

import torch.nn as nn
from guided_diffusion.unet import perpectual_Encoder_model
from model_prep import pre_trained_models

# modeldic = pre_trained_models(32,"cifar100").Pre_train_Models()
#
# print(modeldic["resnet"])
#
# data = torch.rand(4,3,32,32)
# # data = transformers.Resize(224)(data) #这个模型只接受244像素的图像！！！
#
# data2 = torch.rand(4,3,64,64)
#
# #要这么写：
# input = modeldic['FE'](data,return_tensors="pt")
#
# result = modeldic["resnet"](**input)
#
# print(result)

#----------------------------------------------------------------------------
#测试采样部分：


#----------------------------------------------------------------------------

# data = torch.randn([2,256,512])
# data2 = torch.randn([2,256,32,32]).view(2,256,-1)
#
# result1 = nn.Linear(512*256,2048)(data)
# print(result1.shape)
# result2 = nn.Linear(32*32,2048)(data2)
# print(result2.shape)

#----------------------------------------------------------------------------
# encoder = perpectual_Encoder_model()
# data = torch.randn([4,3,32,32])
#
# print(encoder._modules)
#
# result = encoder(data)
# print(result.shape)
# from pytorch_model_summary import summary
# result =  summary(encoder,data)
# print(result.shape)
# print(result)


#----------------------------------------------------------------------------

models_dic = pre_trained_models().Pre_inference_models()


data = torch.randn([4,3,32,32])

result = models_dic['resnet'](data)

print(result.shape,result.size()) #torch.Size([4, 1536, 32, 32]) torch.Size([4, 1536, 32, 32])

#----------------------------------------------------------------------------









