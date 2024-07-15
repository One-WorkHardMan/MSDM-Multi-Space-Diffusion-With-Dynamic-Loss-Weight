OriginDataPath="/data/students/liuzhou/projects/DataSets/tinyimagenet/tiny-imagenet-200/train"
OutPath="/data/students/liuzhou/projects/fid_ref/tinyimagenet.npz"
TestDataPath="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/imagenet_fid_generateimgs"

python -m pytorch_fid --save-stats $OriginDataPath $OutPath
#python -m pytorch_fid $TestDataPath $OriginDataPath \
#--dim 2048