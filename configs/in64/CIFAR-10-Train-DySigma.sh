# set -ex

pip install -r requirements.txt
pip install -e .

DATA_DIR="/data/students/liuzhou/projects/DataSets/cifar-10"

GPUS=$1 #4
BATCH_PER_GPU=$2 #4

#需要设置 时间，学习率，来区分实验；
EXP_NAME=cifar-vit-b_layer12_lr_1e-5_099_099_pred_x0__dy_Sigma__fp16_7_12_GPUS_$GPUS-BATCH_PER_GPU_$BATCH_PER_GPU

MODEL_BLOB="/mnt/external"
if [ ! -d $MODEL_BLOB ]; then
    MODEL_BLOB="."
fi

OPENAI_LOGDIR="${MODEL_BLOB}/exp/guided_diffusion/$EXP_NAME"
# if permission denied
mkdir -p $OPENAI_LOGDIR #&& sudo chmod 777 $OPENAI_LOGDIR
OPENAI_LOGDIR=$OPENAI_LOGDIR \
    torchrun --nproc_per_node=${GPUS} --master_port=23456 scripts_vit/image_train_vit.py \
    --data_dir $DATA_DIR --image_size 32 --class_cond True --diffusion_steps 1000 \
    --noise_schedule cosine --rescale_learned_sigmas False \
    --learn_sigma True \
    --lr 1e-5 --batch_size ${BATCH_PER_GPU} \
    --log_interval 10 \
    --beta1 0.99 --beta2 0.99 \
    --exp_name $EXP_NAME --use_fp16 True --weight_decay 0.03 \
    --use_wandb False --model_name vit_base_patch2_32 --depth 12 \
    --predict_xstart True --warmup_steps 0 --lr_anneal_steps 0 \
    --mse_loss_weight_type dysigma \
    --clip_norm -1 \
    --in_chans 3 \
    --drop_label_prob 0.15