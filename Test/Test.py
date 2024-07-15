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

transformers1 = transformers.Compose([transformers.ToTensor()])

# dataset =  CIFAR_DATA.CIFAR10("./datasets", train=True, download=True, transform=transformers1)

# all_files = []
# classes = []
# #print(type(dataset[0]),dataset[0])
# print(len(dataset),type(dataset))
# print(dataset[0][1])

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# image = image.resize((32, 32), Image.ANTIALIAS) # 这个image也是32*32
#
#
# image_tensor = torchvision.transforms.ToTensor()(image)[None,...]
# assert len(image_tensor.shape) == 4
#
# image_tensor = image_tensor.repeat(4,1,1,1)
#
# print(image_tensor.shape)

#--------------------------------------------------------------------------------
# resnet50 = torchvision.models.resnet50(pretrained=True)


# resnet50._modules['fc'] = nn.Linear(in_features=2048,out_features=1024,bias=True)

# resnet50._modules['avgpool'] = nn.Identity()
# resnet50._modules['fc'] = nn.Identity()

# print(resnet50._modules)

# for keys in resnet50.state_dict():
#     print(keys)
#
# result = resnet50(image_tensor)
# print(result.shape)

# result = result.reshape(1,1536,20,20)
# Spaces = [chunk.tolist()  for chunk in result.chunk(3,dim=1)]
# spaces = result.chunk(3,dim=1)
# print(spaces[0])

# sp1 ,sp2,sp3 = result.chunk(3,dim=1)
# print(f"sp1.shape:{sp1.shape}")


# print(f"result:{result.shape}")


#--------------------------------------------------------------------

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
#
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image_tensor, return_tensors="pt", padding=True)
# print(inputs.__getstate__)
#
# outputs = model(**inputs)
# # outputs = outputs.to_tuple()
# print(f"image_embeds{outputs.image_embeds},{outputs.image_embeds.shape}")
#
#
#
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# print(logits_per_image.shape)
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
# print(probs)

# -------------------------------------------------------------------
from guided_diffusion.unet import DecoderUnetModel
Decoder = DecoderUnetModel( generate_image_size = 64,
        in_channels = 512,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_new_attention_order=False,)
# print(Decoder._modules)
data = torch.randn(4,512,32,32)
from pytorch_model_summary import summary
result =  summary(Decoder,data)
print(result)

import torchvision.transforms as transforms

# 创建一个转换函数
to_pil = transforms.ToPILImage()
# # 遍历向量的第一维，得到每个图像
# for i in range(result.size(0)):
#     # 把张量转换成PIL.Image
#     image = to_pil(result[i])
#     # 显示或保存图像
#     # image.show()
#     image.save(f'images/image_{i}.png')

# -----------------------------------------------------------------------

# import pytorch_ssim
# from torch.autograd import Variable
# from torch import optim
# import torch
# # from scikitimage.metrics import structural_similarity
# # from skimage.metrics import structural_similarity
# import torchvision.transforms as transformers
#
#
# img1 = torch.randn(4,3,512,512,requires_grad=False)
# resize_fun = transforms.Resize(32)
#
# print(img1.shape)
# img1 = resize_fun(img1)
# print(img1.shape)
#
# img2 = torch.rand(4,3,32,32)
# print(img2.shape)
#
# ssim_val = pytorch_ssim.ssim(img1, img2).item()
#
# ssim_loss  = pytorch_ssim.SSIM()
# print(ssim_val)
# print(ssim_loss(img1,img2))














