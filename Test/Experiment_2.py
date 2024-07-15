# import edm.fid
# dataset_path  = "/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/gen_training_images/cifar_fid_generateimgs"
# dest_path = "/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/"
# batch = 16
#
# edm.fid.ref(dataset_path,dest_path,batch)

#-----------------------------------------------------------------------------------------------------------------------
import blobfile as bf
import os
import numpy as np
from PIL import Image

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

# data_dir = "/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/gen_inferences_images/tinyimagenet/0"
# data_dir = "/data/students/liuzhou/projects/DataSets/tinyimagenet/tinyimagenet-64px-10w-pure"
# data_dir = "/data/students/liuzhou/projects/DataSets/lsun_bedroom/data0/lsun/bedroom"
data_dir = "/data/students/liuzhou/projects/DataSets/mini-imagenet-256/imagenet-mini"

results = _list_image_files_recursively(data_dir)

# for addr,i in zip(results,range(len(results))):
#     if "image_no_uvit" in addr:
#         with bf.BlobFile(addr,"rb") as f:
#             content = f.read()
#
#         with bf.BlobFile(f"/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/imagenet_fid_generateimgs/{os.path.basename(addr)}","wb") as f:
#             f.write(content)
#
#         print(f"write Done {i}.img")


#-----------------------------------------------------------------------------------------------------------------------
# for addr,i in zip(results,range(len(results))):
#     with bf.BlobFile(addr,"rb") as f:
#         # content = f.read()
#         pil_image = Image.open(f)
#         pil_image.load()
#     pil_image = pil_image.convert("RGB")
#     arr = center_crop_arr(pil_image,256)
#     image = Image.fromarray(arr) #将np数组转成PIL格式
#
#     with bf.BlobFile(f"/data/students/liuzhou/projects/DataSets/mini-imagenet-256/imagenet_mini_256_pure_imgs/{os.path.basename(addr)}","wb") as f:
#         image.save(f)
#
#     print(f"write Done {i}.img")

#-----------------------------------------------------------------------------------------------------------------------

list1 = list(range(1,38000,5))
print(len(list1),list1)
# rand_num =

for addr,i in zip(results,range(len(results))):
    # if i > 10000 and i < 60000:
    if i in list1:
        with bf.BlobFile(addr,"rb") as f:
            # content = f.read()
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = center_crop_arr(pil_image,256)
        image = Image.fromarray(arr) #将np数组转成PIL格式

        with bf.BlobFile(f"/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/imagenet_256_fid_generateimgs/{os.path.basename(addr)}","wb") as f:
            image.save(f)

        print(f"write Done {i}.img")