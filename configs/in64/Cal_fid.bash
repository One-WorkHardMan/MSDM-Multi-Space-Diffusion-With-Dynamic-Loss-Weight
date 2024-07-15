pip install -r requirements.txt
pip install -e .

export CUDA_VISIBLE_DEVICES=1,2,3

BATCH=$1
NUM_SAMPLES=$2
GPUS=$3
#testimages="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/cifar_fid_generateimgs"
#testimages="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/celeba_fid_generateimgs"
#testimages="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/imagenet_fid_generateimgs"
#testimages="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/lsunbedrooms_fid_generateimgs"
testimages="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/imagenet_256_fid_generateimgs"


#originimages="/data/students/liuzhou/projects/DataSets/tinyimagenet/tinyimagenet-64px-10w-pure"
#originimages="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/Test"
#originimages="/data/students/liuzhou/projects/DataSets/lsun_bedroom/lsunbedrooms_pure_imgs_303124"
originimages="/data/students/liuzhou/projects/DataSets/mini-imagenet-256/imagenet_mini_256_pure_imgs"



#ref="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz"

#destref="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
#destref="/data/students/liuzhou/projects/fid_ref/lsunbedrooms-256px.npz"
#destref="/data/students/liuzhou/projects/fid_ref/celeba—64px.npz"
destref="/data/students/liuzhou/projects/fid_ref/mini_imagenet_256px.npz"


cd /data/students/liuzhou/projects/Min-snr-diffusion-2/edm
#python fid.py ref --data=$originimages --dest=$destref

torchrun --standalone --nproc_per_node=$GPUS
python fid.py calc --images=$testimages \
--ref $destref \
--num $NUM_SAMPLES
cd ..