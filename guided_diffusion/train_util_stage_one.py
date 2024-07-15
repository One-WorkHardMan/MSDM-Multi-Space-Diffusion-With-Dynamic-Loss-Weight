import copy
import functools
import os
import glob
import pdb
import sys

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch
import torchvision
import pytorch_ssim

try:
    from apex.optimizers import FusedAdam
except:
    FusedAdam = None

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import (
    LossAwareSampler, UniformSampler
)
from torch.cuda.amp import autocast, GradScaler
from .lr_scheduler import cosine_scheduler, update_lr_weightdecay, get_lr_wd
import numpy as np
from scipy.stats import entropy

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model = None,
            diffusion = None,
            data,
            batch_size,
            microbatch,
            lr=1e-4,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            betas=(0.9, 0.999),
            warmup_steps=0,
            lr_final=1e-5,
            use_fused_adam=False,

            clip_norm=-1.,  # < 0 to not use 梯度裁剪

            hack_not_ema=False,
            _debug_log_detailed_timesteps=False,
            bin_id=-1,
            pg_inv=False,

            pre_trained_models={},
            use_pretrained_models=False,
            split_spaces=3,
            decode_model=None,
            pre_model_on_gpu=True,
            want_generative_imange_size=64,
            image_size=32,
            dataset_name = "",
            entropies
    ):
        self.pre_trained_models = pre_trained_models
        self.pre_model_on_gpu = pre_model_on_gpu
        self.decode_model = decode_model

        self.model = model

        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = None
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        # self.Test_dict=None

        self.use_pretrained_models = use_pretrained_models  # 是否使用预训练模型？
        self.split_spaces = split_spaces  # 划分子空间个数

        self.hack_not_ema = hack_not_ema

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.origin_data_before_split = None

        self.want_generative_imange_size = want_generative_imange_size
        self.image_size = image_size
        self.big_to_small = self.want_generative_imange_size < self.image_size
        self.dataset_name = dataset_name
        self.sync_cuda = th.cuda.is_available()
        self.scaler = GradScaler()
        self.entropies = entropies

        # ---------------------------------------------------------------------------------
        if self.decode_model is not None and self.use_pretrained_models == True:
            # 建立 Decoder的 训练器，这里 没有 用fp32.
            self.mp_trainer_decoder = MixedPrecisionTrainer(
                model=self.decode_model,
                use_fp16=False,
                fp16_scale_growth=fp16_scale_growth,
                clip_norm=clip_norm,
            )
            self.mp_trainer_encoder = MixedPrecisionTrainer(
                model=self.pre_trained_models["resnet"],
                use_fp16=False,
                fp16_scale_growth=fp16_scale_growth,
                clip_norm=clip_norm,
            )
        # ------------------------------------------------------------------------------------------
        logger.info(">>>>>>>>> Using AdamW")
        self.opt_decoder = AdamW(
            self.mp_trainer_decoder.master_params, lr=self.lr,
            weight_decay=self.weight_decay, betas=betas
        )
        self.opt_encoder = AdamW(
            self.mp_trainer_encoder.master_params, lr=self.lr,
            weight_decay=self.weight_decay, betas=betas
        )

        # if self.resume_step:
        #     self._load_optimizer_state()
        #     # Model was resumed, either due to a restart or a checkpoint
        #     # being specified at the command line.
        #     self.ema_params = [
        #         self._load_ema_parameters(rate) for rate in self.ema_rate
        #     ]
        # else:
        #     self.ema_params = [
        #         copy.deepcopy(self.mp_trainer.master_params)
        #         for _ in range(len(self.ema_rate))
        #     ]

        th.cuda.empty_cache()  # 清缓存

        self.back_bone_models = {}

        # ----------------------------------------------------------------------------------------
        if th.cuda.is_available():
            self.use_ddp = True
            # 给预训练模型上GPU
            if self.pre_model_on_gpu:
                if self.pre_trained_models['FE'] is not None:
                    logger.info("Features_Extractor is Ready.......")

                # Encoder和Decoder都应该在三个进程中保持一致
                self.pre_trained_models["resnet"].to(dist_util.dev())
                self.pre_trained_models["resnet"] = DDP(
                    self.pre_trained_models["resnet"],
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=True,
                    bucket_cap_mb=128,
                    find_unused_parameters=True,
                )
                # 给解码器上GPU,并且多进程训练一个解码器。
                self.decode_model = self.decode_model.to(dist_util.dev())
                self.decode_model = DDP(
                    self.decode_model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=True,
                    bucket_cap_mb=128,
                    find_unused_parameters=True,
                )
        else:
            logger.info("没有GPU你跑nm呢？")
            exit(-1)
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        if self.step + self.resume_step >= self.lr_anneal_steps and self.lr_anneal_steps > 0:
            exit(-1)

        # ------------------------------------------------------------------------------------------
        # learning scheduler
        # if lr_anneal_steps > 0 and warmup_steps > 0:
        #     self.lr_sched = cosine_scheduler(
        #         base_value=lr,
        #         final_value=lr_final,
        #         start_warmup_value=0.0,
        #         warmup_steps=warmup_steps,
        #         total_steps=lr_anneal_steps
        #     )
        #     update_lr_weightdecay(
        #         self.step + self.resume_step,
        #         lr_schedule_values=self.lr_sched,
        #         wd_schedule_values=None,
        #         optimizer=self.opt
        #     )
        # else:
        #     self.lr_sched = None

        # whether to log detailed loss
        self.detailed_log_loss = None
        if _debug_log_detailed_timesteps:
            self.detailed_log_loss = detailed_logger(
                os.getenv("OPENAI_LOGDIR", "exp"), rank=dist.get_rank())
        self.bin_id = bin_id
        self.pg_inv = pg_inv

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    resume_checkpoint, dist_type='pytorch', map_location=dist_util.dev()
                )
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                    # pos embedding
                    # import pdb; pdb.set_trace()
                    if self.model.state_dict()['pos_embed'].shape[1] == state_dict['pos_embed'].shape[1] + 1:
                        pos_embed = th.zeros(size=self.model.state_dict()['pos_embed'].shape)
                        pos_embed[:, 0, :] = state_dict['pos_embed'][:, 0, :]
                        pos_embed[:, 2:, :] = state_dict['pos_embed'][:, 1:, :]
                        state_dict['pos_embed'] = pos_embed

                self.model.load_state_dict(state_dict, strict=False)
                del state_dict

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, dist_type='pytorch', map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
                del state_dict

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, dist_type='pytorch', map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
            del state_dict

    # ------------------------------------------------------------------------------------------------------
    def run_loop(self):

        if self.use_pretrained_models == True:
            while (
                    (not self.lr_anneal_steps
                     or self.step + self.resume_step < self.lr_anneal_steps)
                    and
                    # self.step < 50000
                    self.step < 100000

            ):
                if self.step % self.save_interval == 0 and self.step > 0:

                    self.save_stage_one()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                batch, cond = next(self.data)
                self.origin_data_before_split = copy.deepcopy(batch)  # 保存未分割之前的数据
                if self.use_pretrained_models:
                    # with torch.cuda.amp.autocast():
                    resnet = self.pre_trained_models["resnet"]
                    batch = batch.to(dist_util.dev())  # 训练数据放到GPU上面
                    if self.pre_trained_models["FE"] is not None:
                        inputs = self.pre_trained_models['FE'](batch, return_tensors="pt")
                        inputs = inputs.to(dist_util.dev())
                    outputs = resnet(batch) if self.pre_trained_models['FE'] is None else resnet(
                        **inputs).logits
                    batch = outputs

                # pdb.set_trace()
                self.run_step(batch, cond)

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                    # log some other values
                    # logger.logkv("custom_step", self.step + self.resume_step)
                    min_lr, max_lr, _ = get_lr_wd(optimizer=self.opt_decoder)
                    logger.logkv("min_lr", min_lr)
                    logger.logkv("max_lr", max_lr)
                    logger.logkv("entropy",np.mean(np.array(self.entropies)))
                self.step += 1
            # Save the last checkpoint if it wasn't already saved.
            if (self.step - 1) % self.save_interval != 0 and self.step >= 100:
                self.save_stage_one()

            if self.detailed_log_loss is not None:
                self.detailed_log_loss.close()
        # ------------------------------------------------------------------------------------------------------
        # if self.use_pretrained_models == False:
        #
        #     for n in range(self.split_spaces):
        #         if dist.get_rank() != n:
        #             pass
        #         else:
        #             while (
        #                     (not self.lr_anneal_steps
        #
        #                      or self.step + self.resume_step < self.lr_anneal_steps)
        #
        #                     and
        #
        #                     self.step < 50000
        #
        #                     # or self.step > 50000
        #
        #             ):
        #                 if self.step % self.save_interval == 0 and self.step > 0:
        #                     self.save_multiproc(n)
        #                     # Run for a finite amount of time in integration tests.
        #                     if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
        #                         return
        #
        #                 batch, cond = next(self.data)
        #                 self.origin_data_before_split = copy.deepcopy(batch)  # 保存未分割之前的数据
        #                 spaces = {}
        #                 # 如果使用了预训练模型 那么，就是在隐空间进行训练，数据的channel要改变,默认in_channel是3；
        #                 if self.use_pretrained_models:
        #                     resnet = self.pre_trained_models["resnet"]
        #
        #                     batch = batch.to(dist_util.dev())  # 训练数据放到GPU上面
        #
        #                     if self.pre_trained_models["FE"] is not None:
        #                         inputs = self.pre_trained_models['FE'](batch, return_tensors="pt")
        #                         inputs = inputs.to(dist_util.dev())
        #
        #                     outputs = resnet(batch) if self.pre_trained_models['FE'] is None else resnet(
        #                         **inputs).logits
        #                     # from pytorch_model_summary import summary
        #                     # result = summary(resnet, batch)
        #                     # print(result)
        #
        #                     # if self.pre_trained_models["FE"] is not None:
        #                     #     out_layer = torch.nn.Linear(768,1536*batch.shape[2]*batch.shape[2]).to(dist_util.dev())
        #                     #     act = torch.nn.ReLU()
        #                     #     outputs = out_layer(outputs)
        #                     #     # pdb.set_trace()
        #                     #     outputs = act(outputs)
        #                     #     outputs = outputs.reshape(self.batch_size, 1536, 32, 32)
        #
        #                     for i in range(self.split_spaces):
        #                         spaces[f"sp{i}"] = outputs.chunk(self.split_spaces, dim=1)[i]
        #
        #                 batch = spaces[f"sp{n}"]
        #                 # 训练！！！！！！！！！！
        #                 self.run_step(batch, cond)
        #
        #                 if self.step % self.log_interval == 0:
        #                     logger.dumpkvs()
        #                     # log some other values
        #                     # logger.logkv("custom_step", self.step + self.resume_step)
        #                     min_lr, max_lr, _ = get_lr_wd(optimizer=self.opt)
        #                     logger.logkv("min_lr", min_lr)
        #                     logger.logkv("max_lr", max_lr)
        #
        #                 self.step += 1
        #
        #             # Save the last checkpoint if it wasn't already saved.
        #             if (self.step - 1) % self.save_interval != 0 and self.step >= 100:
        #                 self.save(n)
        #
        #             if self.detailed_log_loss is not None:
        #                 self.detailed_log_loss.close()
        #
        #             # self.back_bone_models[f"sp{n}_model"] = copy.deepcopy(self.ddp_model)
        #             # self.mp_trainers[f"sp{n}_model_mp_trainer"] = copy.deepcopy(self.ddp_model)
        #             #
        #             # self.step = 0
        #             # self.back_bone_models[f"sp{n+1}_model"] = copy.deepcopy(self.back_bone_models[f"sp{n}_model"])
        #             # self.mp_trainers[f"sp{n+1}_model_mp_trainer"] = copy.deepcopy(self.mp_trainers[f"sp{n}_model_mp_trainer"])

    # ------------------------------------------------------------------------------------------------------
    def run_step(self, batch, cond):


        self.mp_trainer_decoder.zero_grad()
        self.mp_trainer_encoder.zero_grad()
        # self.mp_trainer.zero_grad()
        self.forward_backward(batch, cond)
        # pdb.set_trace()
        self.mp_trainer_decoder.optimize(self.opt_decoder)
        self.mp_trainer_encoder.optimize(self.opt_encoder)
        # self.scaler.step(self.opt_decoder)
        # self.scaler.step(self.opt_encoder)
        # self.scaler.update()

        # took_step = self.mp_trainer.optimize(self.opt)
        # if took_step and not self.hack_not_ema:
        #     self._update_ema()
        # self._anneal_lr()
        self.log_step()

    # ------------------------------------------------------------------------------------------------------
    def forward_backward(self, batch, cond):

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            # pdb.set_trace()
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            # clip_outputs = None
            # # 如果使用预训练模型，那么y就要改变，不再是类别，而是CLIP的输出；
            # if self.use_pretrained_models:
            #     clip_vision_model = self.pre_trained_models["clip_vision_model"]
            #     image_processor = self.pre_trained_models['image_processor']
            #     # 这里可能还有点的问题，这里是的self.origin_data_before_split 是没有做microminibatch 的分割，也就是说，返回的tensor的size是BatchSize，而不是mini_batch。
            #     inputs = image_processor(text=["a dog"], images=self.origin_data_before_split, return_tensors="pt",
            #                              padding=True)
            #     clip_outputs = clip_vision_model(**inputs)
            #     # pdb.set_trace()
            #     # micro_cond["y"] = clip_outputs.image_embeds[i:i + self.microbatch].to(dist_util.dev())
            #     micro_cond["y"] = clip_outputs.image_embeds
            #     micro_cond['y'] = micro_cond['y'].to(dist_util.dev())
            # # pdb.set_trace()
            last_batch = (i + self.microbatch) >= batch.shape[0]
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            #
            # compute_losses = functools.partial(
            #     self.diffusion.training_losses,
            #     self.ddp_model,
            #     micro,
            #     t,
            #     model_kwargs=micro_cond,
            #     decode_model=self.decode_model,
            #     origin_data_without_trans=self.origin_data_before_split,
            #     iteration=self.step,
            # )
            # 计算信息熵
            # entropy_value = self.Cal_info_entropy(micro.cpu(),2,True)
            # self.entropies.append(entropy_value)

            compute_losses = functools.partial(
                self.stage_one_trainging_loss,
                micro,
                decode_model=self.decode_model,
                origin_data_without_trans=self.origin_data_before_split,
                iteration=self.step,
                split_space = self.split_spaces
            )
            if last_batch or not self.use_ddp:
                # pdb.set_trace()
                losses = compute_losses()
            else:
                with self.decode_model.no_sync():
                    losses = compute_losses()

            _lambda = 0.3
            loss =  _lambda * losses["ssim_loss"] + (1 - _lambda) * losses["L1_loss"]
            losses['Total_loss'] = loss
            log_loss_dict_stage_one(
                {k : v for k, v in losses.items()}
            )
            self.mp_trainer_decoder.backward(loss)
            # self.scaler.scale(loss).backward() # 用GradScaler对象缩放损失，并反向传播


    # ------------------------------------------------------------------------------------------------------

    # def _update_ema(self):
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         update_ema(params, self.mp_trainer.master_params, rate=rate)
    #
    # def _anneal_lr(self):
    #     if not self.lr_anneal_steps or (self.lr_sched is None):
    #         return
    #
    #     if self.lr_sched is not None:
    #         update_lr_weightdecay(
    #             self.step + self.resume_step,
    #             lr_schedule_values=self.lr_sched,
    #             wd_schedule_values=None,
    #             optimizer=self.opt
    #         )
    #         return
    #
    #     frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
    #     lr = self.lr * (1 - frac_done)
    #     for param_group in self.opt.param_groups:
    #         param_group["lr"] = lr

    # ------------------------------------------------------------------------------------------------------

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save_multiproc(self, n: int):
        def save_checkpoint(rate, params, mp_trainer, name):
            state_dict = mp_trainer.master_params_to_state_dict(params)

            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model_space_{n}_modelname_{n}_{name}_{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_space{n}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params, self.mp_trainer, "Uvit")
        save_checkpoint(0, self.mp_trainer_decoder.master_params, self.mp_trainer_decoder, "Decoder")
        save_checkpoint(0, self.mp_trainer_encoder.master_params, self.mp_trainer_encoder, "Encoder")

        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"space_{n}_Uvit_opt_{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"space_{n}_Decoder_opt_{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt_decoder.state_dict(), f)
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"space_{n}_Encoder_opt_{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt_encoder.state_dict(), f)
        dist.barrier()

    def save(self, n: int):
        def save_checkpoint(rate, params, trainers=None):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_space{n}_{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_space{n}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"space{n}_opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def save_stage_one(self):
        def save_checkpoint(rate, params, mp_trainer, name):
            state_dict = mp_trainer.master_params_to_state_dict(params)

            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"_modelname__{name}_{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        # save_checkpoint(0, self.mp_trainer.master_params, self.mp_trainer, "Uvit")
        save_checkpoint(0, self.mp_trainer_decoder.master_params, self.mp_trainer_decoder, "Decoder")
        save_checkpoint(0, self.mp_trainer_encoder.master_params, self.mp_trainer_encoder, "Encoder")

        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)
        if dist.get_rank() ==0 :
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"space__Decoder_opt_{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt_decoder.state_dict(), f)

            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"space__Encoder_opt_{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt_encoder.state_dict(), f)
        dist.barrier()

    def stage_one_trainging_loss(self,micro = None,
                                 decode_model= None,
                                 origin_data_without_trans= None,
                                 iteration= None,
                                 split_space = 3,
                                 ):
        # pdb.set_trace()
        # with torch.cuda.amp.autocast():
        decode_data = decode_model(micro,split_space)
        terms = {}
        origin_data_without_trans_toimg = copy.deepcopy(origin_data_without_trans)
        decode_data_toimg = decode_data.clone()

        if iteration % 10== 0:
            self.generate_train_images(decode_data_toimg, origin_data_without_trans_toimg)

        # pdb.set_trace()
        origin_data_without_trans = origin_data_without_trans.requires_grad_(False).to(dist_util.dev())
        terms["ssim_loss"] = self.calc_ssim_score(origin_data_without_trans, decode_data)
        terms["L1_loss"] = self.HuberLoss(decode_data, origin_data_without_trans)
        print(terms["L1_loss"])
        print(f"-----------------迭代次数：{iteration}-----------------")
        return terms

    def generate_train_images(self,decode_data,origin_data_without_trans):
        import torchvision.transforms as transforms
        import calendar
        import time
        to_pil = transforms.ToPILImage(mode="RGB")
        #暂时resize 32
        resize_func = torchvision.transforms.Resize(self.want_generative_imange_size)
        # resize_func = torchvision.transforms.Resize(32)
        # entropy_value = self.Cal_info_entropy(origin_data_without_trans,2,True)
        # self.entropies.append(entropy_value)

        pdb.set_trace()
        with torch.no_grad():
            decode_data_uint = decode_data.clone()
            for i in range(decode_data.size(0)):
                # 把张量转换成PIL.Image
                decode_data_uint[i] = (decode_data[i] * 255).to(torch.uint8)

                image = to_pil(decode_data_uint[i])

                origin_data_without_trans[i] = resize_func(origin_data_without_trans[i])

                origin_images = to_pil(origin_data_without_trans[i])

                float_imag = to_pil(decode_data[i])
                # imag_insertlinear = resize_func(decode_data[i])
                # float_imag_insertlinear = to_pil(imag_insertlinear)

                # 显示或保存图像
                # image.show()
                path = f'/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/gen_training_images/{self.dataset_name}/{dist.get_rank()}'
                # path = f"/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/Testimages_for_FID/celeba_fid_generateimgs"
                os.makedirs(path,exist_ok=True)
                image.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_pred.png')
                origin_images.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_origin.png')
                float_imag.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_float_image_pred.png')
                # float_imag_insertlinear.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_insert_image.png')


    def calc_ssim_score(self,img1,img2,big_to_small=True):
        #不一定是32
        if big_to_small:
            resize_func = torchvision.transforms.Resize(self.image_size)
            img2 = resize_func(img2)
            ssim_loss = pytorch_ssim.SSIM()
            return -ssim_loss(img1,img2)
        else:
            resize_func = torchvision.transforms.Resize(self.want_generative_imange_size)
            img1 = resize_func(img1)
            ssim_loss = pytorch_ssim.SSIM()
            return -ssim_loss(img1,img2)


    def L1_loss(self,pred,target,big_to_small=True):
        #不一定是32
        if big_to_small:
            resize_func = torchvision.transforms.Resize(self.image_size)
            pred = resize_func(pred)
            return th.nn.L1Loss()(pred,target)
        else:
            resize_func = torchvision.transforms.Resize(self.want_generative_imange_size)
            target = resize_func(target)
            return th.nn.L1Loss()(pred,target)

    def HuberLoss(self,pred,target,big_to_small=True):
        #不一定是32
        if big_to_small:
            resize_func = torchvision.transforms.Resize(self.image_size)
            pred = resize_func(pred)
            return th.nn.L1Loss()(pred,target)
        else:
            resize_func = torchvision.transforms.Resize(self.want_generative_imange_size)
            target = resize_func(target)
            return th.nn.L1Loss()(pred,target)

    def Cal_info_entropy(self, preds, base=2,average=True):
        from torch.nn import functional as F
        entropies = []
        for sub_tensor in preds:
            # 将子tensor展平成一维数组
            # sub_tensor = sub_tensor.flatten().detach().numpy()
            # total = np.sum(sub_tensor)
            # prob = sub_tensor / total
            prob = F.softmax(sub_tensor.flatten()).data.cpu().numpy()

            # 调用entropy函数，计算概率分布的信息熵，并添加到列表中
            entropies.append(entropy(prob,base=base))

        if average:
            entropies =  np.mean(np.array(entropies),axis=0).tolist()
            return entropies
        else:
            return list(entropies)

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    log_dir = get_blob_logdir()
    files = glob.glob(f"{log_dir}/model*.pt")
    max_iter = 0
    for _f in files:
        parsed_iter = int(_f.split('/model')[-1].split('.')[0])
        max_iter = max(parsed_iter, max_iter)

    if max_iter > 0:
        logger.info(f">>>>> Auto resume from {log_dir}/model{max_iter:06d}.pt")
        return f"{log_dir}/model{max_iter:06d}.pt"

    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_loss_dict_stage_one(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

# loss logger for more detailed loss at different timesteps
# ---------------------------------------------------------
from collections import defaultdict


class detailed_logger(object):
    def __init__(self, dir, rank=0) -> None:
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)

        self.out_file = os.path.join(dir, f'log_rank{rank}.txt')
        self.f = open(self.out_file, "w")

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def do_log(self):
        processed_str = ''
        for key, val in self.name2val.items():
            processed_str += f"{key}(cnt{self.name2cnt[key]}):{val:.6f},"
        processed_str += '\n'
        self.f.writelines(processed_str)

    def close(self):
        self.f.close()


def log_detailed_loss_dict(diffusion, ts, losses, detailed_logger: detailed_logger, bins=4):
    for key, values in losses.items():
        detailed_logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(bins * sub_t / diffusion.num_timesteps)
            detailed_logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

    # detailed_logger.do_log()