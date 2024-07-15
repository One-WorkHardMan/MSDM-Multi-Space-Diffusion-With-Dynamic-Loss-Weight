# set -ex

pip install -r requirements.txt
pip install -e .

export CUDA_VISIBLE_DEVICES=2

#DATA_DIR="/data/students/liuzhou/projects/DataSets/cifar-10"
#DATA_DIR="/data/students/liuzhou/projects/DataSets/lsun_church/church_outdoor_train_lmdb_color_64.npy"
#DATA_DIR="/data/students/liuzhou/projects/DataSets/lsun_bedroom/data0"
#DATA_DIR="/data/students/liuzhou/projects/DataSets/tinyimagenet/tiny-imagenet-200/train"
DATA_DIR="/data/students/liuzhou/projects/DataSets/celeba/img_align_celeba"


GPUS=$1 #4
BATCH_PER_GPU=$2
TIME=$3
LR=$4
WANDB=$5
STAGE=$6
DATASETNAME=$7

#需要设置 时间，学习率，来区分实验；
#PRETRAINMODLE_DIR="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/guided_diffusion/Test_81-4-Idea2_cifar10_vit_b_layer--lr-1e-4-099_099_pred_x0_minsnr5_fp16_GPUS_3-BATCH_PER_GPU_4"
#PRETRAINMODLE_DIR='/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/guided_diffusion/Test_87-1-lsun_church-vit_b_layer--lr-1e-5-099_099_pred_x0_minsnr5_fp16_GPUS_3-BATCH_PER_GPU_2'
PRETRAINMODLE_DIR='/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/guided_diffusion/Test_stage-one-827-1-celeba-vit_b_layer--lr-1e-4-099_099_pred_x0_minsnr5_fp16_GPUS_3-BATCH_PER_GPU_2'
#PRETRAINMODLE_DIR="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/guided_diffusion/Test_stage-one-812-1-tinyimagenet-vit_b_layer--lr-1e-4-099_099_pred_x0_minsnr5_fp16_GPUS_3-BATCH_PER_GPU_2"

PRETRAINSTEP="030000"
EXP_NAME=Test_stage-$STAGE-$TIME-$DATASETNAME-vit_b_layer-$DEPTH-lr-$LR-099_099_pred_x0_minsnr5_fp16_GPUS_$GPUS-BATCH_PER_GPU_$BATCH_PER_GPU

MODEL_BLOB="/mnt/external"
if [ ! -d $MODEL_BLOB ]; then
    MODEL_BLOB="."
fi

OPENAI_LOGDIR="${MODEL_BLOB}/exp/guided_diffusion/$EXP_NAME"
# if permission denied
mkdir -p $OPENAI_LOGDIR #&& sudo chmod 777 $OPENAI_LOGDIR
OPENAI_LOGDIR=$OPENAI_LOGDIR \
    torchrun --nproc_per_node=${GPUS} --master_port=23457 scripts_vit/Train_stage_one.py \
    --data_dir $DATA_DIR --class_cond True --diffusion_steps 1000 \
    --noise_schedule cosine --rescale_learned_sigmas False \
    --learn_sigma False \
    --lr ${LR} \
    --batch_size ${BATCH_PER_GPU} \
    --log_interval 10 \
    --beta1 0.99 --beta2 0.99 \
    --exp_name $EXP_NAME --use_fp16 True --weight_decay 0.03 \
    --use_wandb $WANDB \
    --model_name vit_base_patch2_32 \
    --depth 12 \
    --predict_xstart True \
    --warmup_steps 0 \
    --lr_anneal_steps 0 \
    --mse_loss_weight_type min_snr_5 \
    --clip_norm -1 \
    --in_chans 512 \
    --drop_label_prob 0.15 \
    --use_pretrained_models True \
    --split_spaces 3 \
    --pre_model_on_gpu True \
    --generate_image_size 64 \
    --image_size 64 \
    --save_interval 10000 \
    --dataset_name $DATASETNAME \
    --use_stageonepremodel_continuetrain True \
    --pretrained_step $PRETRAINSTEP \
    --pretrained_model_dir $PRETRAINMODLE_DIR
    #--in_chans 512
