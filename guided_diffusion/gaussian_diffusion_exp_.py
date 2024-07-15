"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import copy
import os
import pdb as pdb

import torch
import torchvision.transforms

import guided_diffusion.logger as logger, wandb
import enum
import math
from copy import deepcopy

import numpy as np
import torch as th

import torch.distributed as dist

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
import pytorch_ssim
from guided_diffusion import dist_util


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# --------------------------------------------------
def get_snr(steps=100):
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    betas = betas_for_alpha_bar(
        steps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )
    betas = np.array(betas, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    _alpha = sqrt_alphas_cumprod
    _sigma = sqrt_one_minus_alphas_cumprod

    snr = (_alpha / _sigma) ** 2
    return snr


# --------------------------------------------------


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    VELOCITY = enum.auto()  # the model predicts v


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class Net(torch.nn.Module):
    def __init__(self,
                 model,
                 img_resolution,  # Image resolution.
                 img_channels,  # Number of color channels.
                 pred_x0=False,
                 label_dim=0,  # Number of class labels, 0 = unconditional.
                 use_fp16=False,  # Execute the underlying model at FP16 precision?
                 C_1=0.001,  # Timestep adjustment at low noise levels.
                 C_2=0.008,  # Timestep adjustment at high noise levels.
                 M=1000,  # Original number of timesteps in the DDPM formulation.
                 noise_schedule='cosine',
                 # model_type      = 'DhariwalUNet',   # Class name of the underlying model.
                 # **model_kwargs,                     # Keyword arguments for the underlying model.
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        # self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)
        self.model = model
        self.noise_schedule = noise_schedule

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

        self.pred_x0 = pred_x0

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        if model_kwargs.get('guidance_scale', 0) > 0:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
        else:
            combined = x

        # pdb.set_trace()

        F_x = self.model((c_in * combined).to(dtype), c_noise.flatten().repeat(x.shape[0]).int(), y=class_labels,
                         **model_kwargs)

        assert F_x.dtype == dtype
        if not self.pred_x0:
            if model_kwargs.get('guidance_scale', 0) > 0:
                cond, uncond = torch.split(F_x, len(F_x) // 2, dim=0)
                cond = uncond + model_kwargs['guidance_scale'] * (cond - uncond)
                F_x = torch.cat([cond, cond], dim=0)

            D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        else:
            D_x = F_x
            if model_kwargs.get('guidance_scale', 0) > 0:
                cond, uncond = torch.split(D_x, len(D_x) // 2, dim=0)
                cond = uncond + model_kwargs['guidance_scale'] * (cond - uncond)
                D_x = torch.cat([cond, cond], dim=0)

        return D_x

    def alpha_bar(self, j):
        if self.noise_schedule == 'cosine':
            j = torch.as_tensor(j)
            return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
        elif self.noise_schedule == 'linear':
            j = torch.as_tensor(j)
            betas = np.linspace(0.0001, 0.02, self.M + 1, dtype=np.float64)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            return alphas_cumprod[self.M - j]

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
                            self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
            mse_loss_weight_type='constant',
            dataset_name="no_name"
    ):
        self.test_eq = None
        self.dataset_name = dataset_name

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.mse_loss_weight_type = mse_loss_weight_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        # print("系数1：", _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape),"系数2：",_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape))
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:

            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)

            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)

        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [
            ModelMeanType.START_X,
            ModelMeanType.EPSILON,
            ModelMeanType.VELOCITY
        ]:

            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_v(x_t=x, t=t, v=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(self, x_t, t, v):
        assert x_t.shape == v.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape) * x_t
                - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape) * v
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - pred_xstart
               ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            **kwargs,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        if kwargs.get('experts', None) is not None:
            # t -> index -> model
            idx = kwargs['experts']['inds'][t[0]]
            model = kwargs['experts']['models'][idx]
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            **kwargs,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        tmp_vars = [] if kwargs.get('save_tmp_var', False) else None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                **kwargs,
        ):
            final = sample
            if kwargs.get('save_tmp_var', False):
                tmp_vars.append(sample['tmp_var'])
        if kwargs.get('save_tmp_var', False):
            return final["sample"], th.stack(tmp_vars).transpose(1, 0)
        return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            **kwargs,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    **kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model,
                        x_start,
                        t,
                        model_kwargs=None,
                        noise=None,
                        decode_model=None,
                        origin_data_without_trans=None,
                        iteration=None,
                        cross_att_model=None,
                        split_space=3,
                        origin_encode_data=None,
                        generate_image_size=64
                        ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if decode_model == None:
            logger.info("No Decode Model,Just Train Channels = 3")

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        mse_loss_weight = None
        dy_loss_weight = None
        alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
        sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
        snr = (alpha / sigma) ** 2

        velocity = (alpha[:, None, None, None] * x_t - x_start) / sigma[:, None, None, None]

        # get loss weight
        if self.model_mean_type is not ModelMeanType.START_X or self.mse_loss_weight_type == 'constant':
            mse_loss_weight = th.ones_like(t)
            if self.mse_loss_weight_type.startswith("min_snr_"):
                k = float(self.mse_loss_weight_type.split('min_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0] / snr
                # guided_diffusion.logger.info("mse_loss_weight-860",mse_loss_weight)

            elif self.mse_loss_weight_type.startswith("max_snr_"):
                k = float(self.mse_loss_weight_type.split('max_snr_')[-1])
                # max{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0] / snr

        else:
            if self.mse_loss_weight_type == 'trunc_snr':
                # max{snr, 1}
                mse_loss_weight = th.stack([snr, th.ones_like(t)], dim=1).max(dim=1)[0]
            elif self.mse_loss_weight_type == 'snr':
                mse_loss_weight = snr

            elif self.mse_loss_weight_type == 'inv_snr':
                mse_loss_weight = 1. / snr

            elif self.mse_loss_weight_type.startswith("min_snr_"):
                k = float(self.mse_loss_weight_type.split('min_snr_')[-1])
                # min{snr, k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).min(dim=1)[0]
                logger.logkv("mse_loss_weight", mse_loss_weight.mean())

                # wandb.log({f"mse_loss_weight_minsnr_{k}":mse_loss_weight.mean()})


            elif self.mse_loss_weight_type.startswith("max_snr_"):
                k = float(self.mse_loss_weight_type.split('max_snr_')[-1])
                # max{snr，k}
                mse_loss_weight = th.stack([snr, k * th.ones_like(t)], dim=1).max(dim=1)[0]

        if mse_loss_weight is None:
            logger.info("Its not SNR way!!!Its Dynamic Loss Weight!!!")

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps


        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # import pdb;pdb.set_trace()

            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:

                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)

                if self.mse_loss_weight_type == "dysigma":
                    model_variance = self.p_mean_variance(model, x_t, t, model_kwargs=model_kwargs)["variance"]
                    model_variance = mean_flat(model_variance)
                    dy_loss_weight = model_variance.detach()

                    # import pdb;pdb.set_trace();
                    dy_loss_weight = dy_loss_weight.clamp(0, 10)

                    logger.logkv("dy_loss_weight", dy_loss_weight.mean())

                    mse_loss_weight = dy_loss_weight

                    # wandb.log({"dy_loss_weight":dy_loss_weight.mean()})

                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]

                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.VELOCITY: velocity,
            }[self.model_mean_type]

            if decode_model is not None and iteration % 10 == 0:

                decode_model.eval()

                # 采样再输出
                z = torch.randn(
                    [model_output.shape[0], model_output.shape[1], model_output.shape[2], model_output.shape[3]],
                    device="cuda")

                with torch.no_grad():
                    net = Net(
                        model=model,
                        img_channels=model_output.shape[1],
                        img_resolution=model_output.shape[2],
                        pred_x0=True,
                        noise_schedule="cosine"
                    ).to(dist_util.dev())

                    # 原本这里写成了model_output = selef.ablation_sampler,那就大错特错了，导致model_output不是vit的输出，从而无法反向传播。
                    model_output_samplers_by_edm = self.ablation_sampler(
                        net, latents=noise,  # z ,x_t 都测试过
                        num_steps=50,
                        solver="euler",
                        class_labels=model_kwargs["y"],
                        guidance_scale=0.0,
                    ).to(th.float32)
                    print("0步", f"{torch.nn.MSELoss()(model_output, x_start)}")
                    print("50步", f"{torch.nn.MSELoss()(model_output_samplers_by_edm, x_start)}")
                    # model_output_samplers_by_edm_2 = self.ablation_sampler(
                    #     net, latents=noise,
                    #     num_steps=18,
                    #     solver="euler",
                    #     class_labels=model_kwargs["y"],
                    #     guidance_scale=0.0,
                    # ).to(th.float32)
                    # print("18步", f"{torch.nn.MSELoss()(model_output_samplers_by_edm_2, x_start)}")
                    #
                    # model_output_samplers_by_edm_3 = self.ablation_sampler(
                    #     net, latents=noise,
                    #     num_steps=300,
                    #     solver="euler",
                    #     class_labels=model_kwargs["y"],
                    #     guidance_scale=0.0,
                    # ).to(th.float32)
                    # print("300步", f"{torch.nn.MSELoss()(model_output_samplers_by_edm_3, x_start)}")
                    #
                    # model_output_samplers_by_edm_4 = self.ablation_sampler(
                    #     net, latents=noise,
                    #     num_steps=1000,
                    #     solver="euler",
                    #     class_labels=model_kwargs["y"],
                    #     guidance_scale=0.0,
                    # ).to(th.float32)
                    # print("1000步", f"{torch.nn.MSELoss()(model_output_samplers_by_edm_4, x_start)}")
                    #
                    # print(f"时间步为：{t}_xt 和 噪声z 的差距：{torch.nn.MSELoss()(x_t,noise)}")
                    dist.barrier()
                    # pdb.set_trace()
                    Total_samples_data = [torch.zeros_like(model_output_samplers_by_edm) for i in
                                          range(dist.get_world_size())]
                    Total_model_output_data = [torch.zeros_like(model_output) for i in range(dist.get_world_size())]
                    dist.all_gather(Total_samples_data, model_output_samplers_by_edm)

                    dist.all_gather(Total_model_output_data, model_output)

                    # merge_data = self.CrossAttention(Total_samples_data,cross_att_model)
                    if dist.get_rank() == 0:
                        print(len(Total_samples_data))

                    # 这里不要 edm去采样了，直接把模型输出拿出来解码。。。反而效果更好
                    decode_data = decode_model(Total_model_output_data, split_space)
                    origin_encode_data = decode_model(origin_encode_data, split_space)

                    self.generate_train_images(decode_data, origin_data_without_trans, origin_encode_data,
                                               generate_image_size=generate_image_size)

                    logger.logkv("ssim_value",
                                 self.calc_ssim_score(origin_data_without_trans, decode_data, generate_image_size))
                    logger.logkv("L1loss_value",
                                 self.L1_loss(decode_data, origin_data_without_trans, generate_image_size))

                    print(f"{model.model._modules['linear_projection'].weight}_这是进程：{dist.get_rank()}")

                    # origin_data_without_trans = origin_data_without_trans.requires_grad_(False).to(dist_util.dev())

                # terms["ssim_loss"] = self.calc_ssim_score(origin_data_without_trans,decode_data)
                # terms["L1_loss"] = self.L1_loss(decode_data,origin_data_without_trans)

            # pdb.set_trace()
            assert model_output.shape == target.shape == x_start.shape
            print(f"--------------迭代次数：{iteration}---------------")
            # hack
            terms["mse"] = mse_loss_weight * mean_flat((target - model_output) ** 2)

            # terms['mse'] =  mean_flat((target - model_output) ** 2)
            terms["mse_raw"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def calc_ssim_score(self, img1, img2, generate_image_size):
        # 不一定是32
        resize_func = torchvision.transforms.Resize(generate_image_size)
        img1 = resize_func(img1)
        ssim_loss = pytorch_ssim.SSIM()
        return -ssim_loss(img1, img2)

    def generate_train_images(self, decode_data, origin_data_without_trans, origin_encode_data, generate_image_size=64):
        import torchvision.transforms as transforms
        import calendar
        import time
        to_pil = transforms.ToPILImage(mode='RGB')

        # 暂时resize 32
        resize_func = torchvision.transforms.Resize(generate_image_size)
        with torch.no_grad():
            decode_data_uint = decode_data.clone()
            for i in range(decode_data.size(0)):
                # 把张量转换成PIL.Image
                decode_data_uint[i] = (decode_data[i] * 255).to(torch.uint8)
                image = to_pil(decode_data_uint[i])
                origin_images = to_pil(resize_func(origin_data_without_trans[i]))
                float_imag = to_pil(decode_data[i])
                # imag_insertlinear = resize_func(decode_data[i])
                # float_imag_insertlinear = to_pil(imag_insertlinear)
                image_no_uvit = to_pil(resize_func(origin_encode_data[i]))

                # 显示或保存图像
                # image.show()
                path = f'/data/students/liuzhou/projects/Min-snr-diffusion-2/exp/gen_inferences_images/{self.dataset_name}/{dist.get_rank()}'
                os.makedirs(path, exist_ok=True)
                image.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_int8_pred.png')
                origin_images.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_origin.png')
                float_imag.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_float_image_pred.png')
                # float_imag_insertlinear.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_interpolation_image.png')
                image_no_uvit.save(f'{path}/num:{i}_{calendar.timegm(time.gmtime())}_image_no_uvit.png')

    # edm 采样器
    def ablation_sampler(self,
                         net, latents, class_labels=None, randn_like=torch.randn_like,
                         num_steps=18, sigma_min=None, sigma_max=None, rho=7,
                         solver='heun', discretization='edm', schedule='linear', scaling='none',
                         epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
                         S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
                         **model_kwargs,
                         ):
        assert solver in ['euler', 'heun']
        assert discretization in ['vp', 've', 'iddpm', 'edm']
        assert schedule in ['vp', 've', 'linear']
        assert scaling in ['vp', 'none']

        # Helper functions for VP & VE noise level schedules.
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
                sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma ** 2

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
            sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Define noise level schedule.
        if schedule == 'vp':
            sigma = vp_sigma(vp_beta_d, vp_beta_min)
            sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif schedule == 've':
            sigma = ve_sigma
            sigma_deriv = ve_sigma_deriv
            sigma_inv = ve_sigma_inv
        else:
            assert schedule == 'linear'
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma

        # Define scaling schedule.
        if scaling == 'vp':
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        else:
            assert scaling == 'none'
            s = lambda t: 1
            s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(net.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        t_next = t_steps[0]
        x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(
                t_hat) * S_noise * randn_like(x_cur)

            # Euler step.
            h = t_next - t_hat
            denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels, **model_kwargs).to(torch.float64)
            d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(
                t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            # Apply 2nd order correction.
            if solver == 'euler' or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == 'heun'
                denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels, **model_kwargs).to(torch.float64)
                d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(
                    t_prime)) * x_prime - sigma_deriv(
                    t_prime) * s(t_prime) / sigma(t_prime) * denoised
                x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

        return x_next

    def L1_loss(self, pred, target, generate_image_size):
        # 不一定是32
        resize_func = torchvision.transforms.Resize(generate_image_size)
        target = resize_func(target)
        return th.nn.HuberLoss()(pred, target)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# utils
@th.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return output

    tensors_gather = [th.ones_like(tensor)
                      for _ in range(th.distributed.get_world_size())]
    th.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = th.cat(tensors_gather, dim=0)
    return output