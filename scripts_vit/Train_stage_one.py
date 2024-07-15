"""
Train a diffusion model on images.
"""

import os
import argparse
import pdb

import wandb
import torch.distributed as dist

import torch
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_stage_one import TrainLoop

from Test.model_prep import pre_trained_models
from guided_diffusion.unet import perpectual_Encoder_model
from guided_diffusion.unet import DecoderUnetModel
import blobfile as bf
from pytorch_model_summary import summary
import torch as th

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist('pytorch')
    logger.configure()

    seed = 2022 + dist.get_rank()

    torch.manual_seed(seed)

    np.random.seed(seed)

    # wandb
    USE_WANDB = args.use_wandb
    if int(os.environ["RANK"]) == 0 and USE_WANDB:
        wandb_log_dir = os.path.join(os.environ.get('OPENAI_LOGDIR', 'exp'), 'wandb_logger')
        os.makedirs(wandb_log_dir, exist_ok=True)
        wandb.login(key='ff39f513528b6118b78086a2b7617aa4105ac171')
        wandb.init(project="guided-diffusion-version01", sync_tensorboard=True,
                   name=args.exp_name, id=args.exp_name, dir=wandb_log_dir)

    logger.log(">>>>>>>>>>>>>>>>>>>>>>>creating model and diffusion...<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # model = beit_base_patch4_64()

    # log the args & model
    logger.log(f"==> args \n{args}")

    print(f"dev: {dist_util.dev()}\t rank: {dist.get_rank()}\t worldsize: {dist.get_world_size()}")

    # print("args",args.__dict__)
    import json
    print("args",json.dumps(args.__dict__))

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(">>>>>>>>>>>>>creating data loader...<<<<<<<<<<<<<<<")


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        # deterministic=True #测试用，马上删掉
    )

    models_dic = {}
    if args.use_pretrained_models:
        if args.use_stageonepremodel_continuetrain == False:
            models_dic = pre_trained_models(args.image_size,args.dataset_name).Pre_train_Models()
            # models_dic = pre_trained_models(args.image_size, args.dataset_name).Pre_inference_models(steps="050000",dir="/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/guided_diffusion/Test_stage-one-812-1-tinyimagenet-vit_b_layer--lr-1e-4-099_099_pred_x0_minsnr5_fp16_GPUS_3-BATCH_PER_GPU_2")
            UnetDecoder = DecoderUnetModel(generate_image_size=args.generate_image_size,
                                           num_heads=args.num_heads,
                                           input_size=args.image_size
                                           )
        else:
            models_dic = pre_trained_models(args.image_size, args.dataset_name).Pre_inference_models(steps=args.pretrained_step,
                                                                                                     dir=args.pretrained_model_dir)

            UnetDecoder = DecoderUnetModel(generate_image_size=args.generate_image_size,
                                           num_heads=args.num_heads,
                                           input_size=args.image_size
                                           )
            with bf.BlobFile(f"{args.pretrained_model_dir}/_modelname__Decoder_{args.pretrained_step}.pt", "rb") \
                    as f:
                with torch.no_grad():
                    state_dict = torch.load(f, map_location=dist_util.dev())
                    UnetDecoder.load_state_dict(state_dict)

    # model_parapms(args.image_size,1,models_dic['resnet'])
    # model_parapms(args.image_size,2,UnetDecoder)
    # pdb.set_trace()


    logger.log("training...")
    entropies = []
    TrainLoop(
        model=None,
        diffusion=None,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=None,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        betas=(args.beta1, args.beta2),
        warmup_steps=args.warmup_steps,
        lr_final=args.lr_final,
        # 导入预训练模型
        pre_trained_models=models_dic,
        use_pretrained_models = args.use_pretrained_models,
        split_spaces =args.split_spaces,

        #传入解码器
        decode_model = UnetDecoder if args.use_pretrained_models else None,

        #预训练上GPU
        pre_model_on_gpu=args.pre_model_on_gpu,
        want_generative_imange_size = args.generate_image_size,
        image_size = args.image_size,
        dataset_name = args.dataset_name,
        entropies = entropies
    ).run_loop()

    if os.environ["RANK"] == 0 and USE_WANDB:
        wandb.finish()

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        beta1=0.9,
        beta2=0.999,
        exp_name="debug",
        use_wandb=True,
        model_name='beit_base_patch4_64',

        # vit related settings
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_conv_last=False,
        depth=12,
        use_shared_rel_pos_bias=False,
        warmup_steps=0,
        lr_final=1e-5,
        clip_norm=-1.,
        local_rank=0,
        drop_label_prob=0.,
        use_rel_pos_bias=False,
        in_chans=3,

        #是否使用预训练模型
        use_pretrained_models = False,
        split_spaces = 3,

        #想要生成图像的尺寸
        generate_image_size = 64,
        pre_model_on_gpu = True,

        #数据集的名称
        dataset_name = "",

        #是否使用第一阶段的预训练模型，接着训练？
        use_stageonepremodel_continuetrain = False,
        pretrained_step = "050000",
        pretrained_model_dir =""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def model_parapms(image_size,num,model):
    data1 = th.randn((1,3,image_size,image_size))
    data2 = th.randn((1,1536,image_size,image_size))
    if num == 1 :
        summary(model, data1,print_summary=True)
    else:
        summary(model,data2,3,print_summary=True)


if __name__ == "__main__":
    main()
