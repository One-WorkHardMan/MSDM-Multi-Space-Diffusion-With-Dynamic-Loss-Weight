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
from guided_diffusion.train_util_stage_two import TrainLoop
from Test.model_prep import pre_trained_models
from guided_diffusion.unet import perpectual_Encoder_model
from guided_diffusion.unet import DecoderUnetModel
import blobfile as bf
from pytorch_model_summary import summary
import torch as th

def build_model(**kwargs):
    if kwargs['model_name'] == 'vit_base_patch2_32':
        from guided_diffusion.vision_transformer import vit_base_patch2_32
        _model = vit_base_patch2_32(**kwargs)
    elif kwargs['model_name'] == 'vit_large_patch2_32':
        from guided_diffusion.vision_transformer import vit_large_patch2_32
        _model = vit_large_patch2_32(**kwargs)
    elif kwargs['model_name'] == 'vit_base_patch4_64':
        from guided_diffusion.vision_transformer import vit_base_patch4_64
        _model = vit_base_patch4_64(**kwargs)
    elif kwargs['model_name'] == 'vit_large_patch4_64':
        from guided_diffusion.vision_transformer import vit_large_patch4_64
        _model = vit_large_patch4_64(**kwargs)
    elif kwargs['model_name'] == 'vit_xl_patch2_32':
        from guided_diffusion.vision_transformer import vit_xl_patch2_32
        _model = vit_xl_patch2_32(**kwargs)
    else:
        raise NotImplementedError(f'Such model is not supported')
    return _model


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
    #创建VIT模型

    model = build_model(**vars(args))

    # model_parapms(args.image_size,3,model)
    # pdb.set_trace()

    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # log the args & model
    logger.log(f"==> args \n{args}")
    logger.log(f"==> Model \n{model}")

    print(f"dev: {dist_util.dev()}\t rank: {dist.get_rank()}\t worldsize: {dist.get_world_size()}")

    model.to(dist_util.dev())
    # print("args",args.__dict__)
    import json
    print("args",json.dumps(args.__dict__))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(">>>>>>>>>>>>>creating data loader...<<<<<<<<<<<<<<<")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic = args.deterministic,
    )

    if args.use_trained_uvit_parameters != "0":
        for n in range(args.split_spaces):
            if dist.get_rank() == n:
                with bf.BlobFile(f"{args.load_uvitmodel_dir}/model_space_{n}_model_name_{n}_Uvit_{args.use_trained_uvit_parameters}.pt", "rb") \
                        as f:
                    state_dict = torch.load(f,map_location=dist_util.dev())
                    model.load_state_dict(state_dict)
    models_dic = {}
    if args.use_pretrained_models:
        models_dic = pre_trained_models(args.image_size,args.dataset_name).Pre_inference_models(steps = args.pre_train_steps,dir = args.load_model_dir)
        UnetDecoder = DecoderUnetModel(generate_image_size=args.generate_image_size,
                                       num_heads=args.num_heads,
                                       input_size=args.image_size
                                       )
        with bf.BlobFile(f"{args.load_model_dir}/_modelname__Decoder_{args.pre_train_steps}.pt","rb") \
        as f:
            with torch.no_grad():
                state_dict = torch.load(f,map_location=dist_util.dev())
                UnetDecoder.load_state_dict(state_dict)

    # import pdb;pdb.set_trace()
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
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
        schedule_sampler=schedule_sampler,
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
        uvit_resume_step=int(args.use_trained_uvit_parameters),
        generate_image_size = args.generate_image_size,
        # image_size = args.image_size
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
        #训练steps ， 用于inference阶段
        pre_train_steps = "020000",
        # 这个决定 我是不是要shuffle，在infer阶段是不需要的。所以为True
        deterministic = True, #阶段一 为False，阶段二为 True

        load_model_dir = None,
        load_uvitmodel_dir = None,
        #是否使用已经前面训练好的模型的参数？从哪一步开始？
        use_trained_uvit_parameters = "0",
        dataset_name = ""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def model_parapms(image_size,num,model):
    data1 = th.randn((1,3,image_size,image_size))
    data2 = th.randn((1,1536,image_size,image_size))
    data3 = th.randn((1,512,image_size,image_size))
    timesteps = th.randint(0,10,(1,))
    cond = th.randn((1,512))

    if num == 1 :
        summary(model, data1,print_summary=True)
    if num == 2 :
        summary(model,data2,3,print_summary=True)
    else:
        summary(model, data3,timesteps,cond,print_summary=True)

if __name__ == "__main__":
    main()
