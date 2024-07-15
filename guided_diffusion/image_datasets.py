import os
import math
import pdb
import random
import json

import torchvision.datasets
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, Dataset

import DataSets_Deal.CIFAR_DATA


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False, #如果为假，那么就shuffle,几个进程读取的是不同的数据
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    dataset = get_dataset(data_dir,class_cond,image_size,random_crop,random_flip)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []

    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "pt"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        is_lmdb = False
    ):
        super().__init__()
        self.resolution = resolution
        # self.allfiles = image_paths
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.is_lmdb = is_lmdb

    def __len__(self):
        if self.is_lmdb:
            return self.local_images.shape[0]
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        if not self.is_lmdb:
            if not path.endswith(".pt") :
                with bf.BlobFile(path, "rb") as f:
                    pil_image = Image.open(f)
                    pil_image.load()
                pil_image = pil_image.convert("RGB")

                if self.random_crop:
                    arr = random_crop_arr(pil_image, self.resolution)
                else:
                    arr = center_crop_arr(pil_image, self.resolution)

                if self.random_flip and random.random() < 0.5:
                    arr = arr[:, ::-1]

                arr = arr.astype(np.float32) / 255 #归一化到 0-1 ;arr.astype(np.float32) / 127.5 - 1 归一化到[-1,1]
                arr = np.transpose(arr, [2, 0, 1])

                arr = torch.from_numpy(arr)
        
        else:
            if self.is_lmdb:
                arr = path / 255
            else:
                with bf.BlobFile(path, "rb") as f:
                    arr = torch.load(f, map_location="cpu")
                    arr = arr * 0.18215 # https://github.com/facebookresearch/DiT/blob/39cc506776770ac18153d90ac15342ef1ab4008c/main.py#L52
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr, out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def get_dataset(data_dir,class_cond,image_size,random_crop=False,random_flip=True):
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files= []
    classes = None

    if 'imagenet-64x64' in data_dir:
        with open(os.path.join(data_dir, 'dataset.json'), 'r') as f:
            meta_info = json.load(f)
        file_list = meta_info['labels']
        all_files = [os.path.join(data_dir, item[0]) for item in file_list]
        classes = None
        if class_cond:
            classes = [item[1] for item in file_list]
    # else:
    #     all_files = _list_image_files_recursively(data_dir)
    #     classes = None
    #     if class_cond:
    #         # Assume classes are the first part of the filename,
    #         # before an underscore.
    #         class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #         classes = [sorted_classes[x] for x in class_names]
            
    if "celeba" in data_dir:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        #image_size = 128 #这个还是要斟酌
        random_crop = True

    if "tiny-imagenet" in data_dir:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        image_size = 64

    if "lsun_bedroom" in data_dir:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        image_size = 256

    #这个特别一些，因为是lmdb形式，所以用np来读
    if "lsun_church" in data_dir:
        arr = np.load(data_dir) # "/data/students/liuzhou/projects/DataSets/lsun_church/church_outdoor_train_lmdb_color_64.npy",
        arr = np.transpose(arr,[0,3,1,2])
        all_files = torch.from_numpy(arr).to(torch.float32)
        classes = None

    if "coco" in data_dir:
        all_files = _list_image_files_recursively(data_dir) #"/data/students/liuzhou/projects/DataSets/coco"
        classes = None
        image_size = 256 #这个还是要斟酌

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=dist.get_rank() if dist.is_initialized() else 0, # MPI.COMM_WORLD.Get_rank(),
        num_shards=dist.get_world_size() if dist.is_initialized() else 1, # MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        is_lmdb = True if type(all_files)==torch.Tensor else False
    )

    if 'cifar-10' in data_dir:
        import torchvision.transforms as transformers
        transformers1 = transformers.Compose([transformers.ToTensor()
                                              ]) #transformers.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
        dataset = DataSets_Deal.CIFAR_DATA.CIFAR10("./datasets",train=True,download = True,transform=transformers1)


    return dataset


