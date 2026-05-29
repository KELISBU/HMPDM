#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torchvision.transforms.functional as TF
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
import accelerate
import numpy as np
import re
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
import csv
import diffusers
from diffusers import StableDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from models import TaLCUNet, MaPE, HMPDM
from torch.utils.data import Dataset
from timm.models.vision_transformer import PatchEmbed
from torch.optim.swa_utils import AveragedModel


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class CrossDeviceEMA:
    def __init__(self, module, decay=0.999, device_ema="cuda:0", dtype=torch.float32):
        self.decay = decay
        self.device_ema = torch.device(device_ema)
        self.dtype = dtype
        # Map by name; only collect parameters with requires_grad=True
        self.shadow = {}
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().to(self.device_ema, dtype=self.dtype).clone()

    @torch.no_grad()
    def update(self, module):
        # Update by name; if a new trainable parameter appears (e.g. dynamic unfreezing), init a shadow entry
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            src = p.detach().to(self.device_ema, dtype=self.dtype, non_blocking=True)
            if name not in self.shadow:
                self.shadow[name] = src.clone()
            else:
                s = self.shadow[name]
                s.mul_(self.decay).add_(src, alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, module):
        # Copy shadow weights back (only entries that have a shadow)
        for name, p in module.named_parameters():
            if name in self.shadow and p.requires_grad:
                p.copy_(self.shadow[name].to(p.device, dtype=p.dtype), non_blocking=True)

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()
def init_from_scratch(module: nn.Module):
        import torch.nn as nn
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


class VideoFolderDataset(Dataset):
    def __init__(self, base_folder: str, num_samples=100000,
                 width=128, height=128, sample_frames=20,
                 file_exts=('.png', '.jpg', '.jpeg'),
                 augment=True ):
        self.num_samples = num_samples
        self.base_folder = base_folder
        self.folders = [d for d in os.listdir(base_folder)
                        if os.path.isdir(os.path.join(base_folder, d))]
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.file_exts = tuple(e.lower() for e in file_exts)
        self.augment = augment

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _natnum(fn: str) -> int:
        m = re.search(r'(\d+)', fn)
        return int(m.group()) if m else -1

    def __getitem__(self, idx):
        folder_idx = idx % len(self.folders)
        chosen_folder = self.folders[folder_idx]
        folder_path = os.path.join(self.base_folder, chosen_folder)

        frames = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(self.file_exts)]
        if len(frames) < self.sample_frames:
            raise ValueError(f"'{chosen_folder}' has only {len(frames)} frames, less than the required {self.sample_frames}.")

        frames.sort(key=self._natnum)

        '''max_start = len(frames) - self.sample_frames
        start_idx = 0 if max_start <= 0 else np.random.randint(0, max_start + 1)
        selected = frames[start_idx:start_idx + self.sample_frames]'''

        selected = frames[-self.sample_frames:]#save last sample_frmaes in folder

        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width), dtype=torch.float32)

        do_flip = self.augment and np.random.rand() < 0.5

        for i, fn in enumerate(selected):
            fp = os.path.join(folder_path, fn)
            with Image.open(fp) as img:
                img = img.convert('RGB')
                img = img.resize((self.width, self.height), Image.BICUBIC)
                if do_flip:
                    img = TF.hflip(img)
                arr = np.array(img, dtype=np.float32)
                arr = arr / 127.5 - 1.0
                arr = np.transpose(arr, (2, 0, 1))
                pixel_values[i] = torch.from_numpy(arr)

        return {'pixel_values': pixel_values}

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_folder",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--csv",
        default='/home/ke/datadiff/ROL/key.csv',
        type=str,
    )
    parser.add_argument(
        "--svdpretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-video-diffusion-img2vid-xt',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sdpretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-base',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=30,
        help="Total frames per clip; must equal F_hist + F_future.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Frame width (paper uses 128).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Frame height (paper uses 128).",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--p_sc",
        type=float,
        default=0.9,
        help="Self-conditioning probability (W.A.L.T. recommends 0.9).",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/1",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate after warmup (paper uses 2e-5).",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=3165,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--F_hist",
        type=int,
        default=2,
        help="Number of historical frames used as context (P in the paper).",
    )
    parser.add_argument(
        "--F_future",
        type=int,
        default=28,
        help="Number of future frames to predict (F in the paper).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2975,
        help="Number of training samples per epoch (Cityscapes=2975, KITTI=759).",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    args = parse_args()
    args.F_tot = args.F_hist + args.F_future  # total frames
    args.num_frames = args.F_tot
    '''if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead.",
        )'''
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        device_placement=False, 
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )
    print(">>> mixed_precision mode:", accelerator.state.mixed_precision)
    print(">>> scaler object   :", accelerator.scaler)
    '''generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)'''

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Single-GPU setup: everything goes on cuda:0 (with CUDA_VISIBLE_DEVICES this is whatever GPU you pick)
    device_unet = device_encoder = torch.device("cuda:0")
    # Load img encoder, tokenizer and models.
   
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.svdpretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16" )
    unet =TaLCUNet.from_pretrained(
        args.svdpretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=False,
        variant="fp16",
        ).to(device_unet)

    ''' unet.apply(init_from_scratch)

    # Zero / small-std init at the end of residual branches and attention output to suppress early instability
    for n, p in unet.named_parameters():
        if any(k in n for k in ["conv2.weight", "to_out.0.weight"]):
            nn.init.zeros_(p)

    if hasattr(unet, "conv_out"):
        nn.init.normal_(unet.conv_out.weight, mean=0.0, std=1e-5)
        if unet.conv_out.bias is not None:
            nn.init.zeros_(unet.conv_out.bias)'''
    unet.time_embedding_cond.load_state_dict(unet.time_embedding.state_dict())
    encoder=MaPE(input_size=args.height//8,num_frames=args.F_hist,n_pairs=3).to(device_unet)
    #ncoder.initialize_weights()
    model = HMPDM(unet, encoder).to(device_unet)
    

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    
    #unet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
 
    vae.to(dtype=weight_dtype).to(device_encoder)#.to(dtype=weight_dtype)
    

    # Create EMA for the unet.
    
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        '''def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()'''
        def save_model_hook(models, weights, output_dir):
    # `models` is the model list managed by accelerate (we only register a single wrapper)
            saved_any = False
            for wrapped in list(models):
                try:
                    unw = accelerator.unwrap_model(wrapped)  # wrapper: HMPDM
                except Exception:
                    unw = wrapped

                # A) Save UNet (diffusers API)
                unet_dir = os.path.join(output_dir, "unet")
                os.makedirs(unet_dir, exist_ok=True)
                unw.unet.save_pretrained(unet_dir)
                saved_any = True

                # B) Save history encoder (state_dict)
                ctx_path = os.path.join(output_dir, "ctx_encoder.pt")
                torch.save(unw.ctx.state_dict(), ctx_path)

                # C) EMA (if enabled)
                if args.use_ema:
                # UNet EMA
                    unet_ema_path = os.path.join(output_dir, "unet_ema.pt")
                    torch.save(
                        {
                            "shadow": {k: v.detach().cpu() for k, v in ema_unet.shadow.items()},
                            "decay": float(ema_unet.decay),
                        },
                        unet_ema_path,
                    )

                    # ctx EMA
                    ctx_ema_path = os.path.join(output_dir, "ctx_encoder_ema.pt")
                    torch.save(
                        {
                            "shadow": {k: v.detach().cpu() for k, v in ema_ctx.shadow.items()},
                            "decay": float(ema_ctx.decay),
                        },
                        ctx_ema_path,
                    )


            # Tell Accelerate we already saved the model manually; skip default weight saving
            weights.clear()

            if not saved_any:
                print(f"[save_hook] warning: models list empty; nothing saved. outdir={output_dir}")

        
        def load_model_hook(models, input_dir):
           

            # Helper: intersect the saved shadow with the current set of trainable parameter names (avoids errors when the structure changes)
            def filter_shadow_to_trainable(shadow_dict, module):
                trainable_names = {n for n, p in module.named_parameters() if p.requires_grad}
                filtered = {n: t for n, t in shadow_dict.items() if n in trainable_names}
                dropped = set(shadow_dict.keys()) - set(filtered.keys())
                if dropped:
                    print(f"[load_hook][EMA] dropped {len(dropped)} entries not in current trainable set.")
                return filtered

            while len(models) > 0:
                wrapped = models.pop()
                try:
                    unw = accelerator.unwrap_model(wrapped)  # HMPDM
                except Exception:
                    unw = wrapped

                # A) Load UNet (non-EMA)
                load_unet = TaLCUNet.from_pretrained(input_dir, subfolder="unet")
                unw.unet.register_to_config(**load_unet.config)
                unw.unet.load_state_dict(load_unet.state_dict())
                del load_unet

                # B) Load ctx (non-EMA)
                ctx_path = os.path.join(input_dir, "ctx_encoder.pt")
                if os.path.exists(ctx_path):
                    state = torch.load(ctx_path, map_location="cpu")
                    try:
                        unw.ctx.load_state_dict(state, strict=False)
                    except Exception as e:
                        print(f"[load_hook] warn: load ctx failed: {e}")
                else:
                    print("[load_hook] ctx_encoder.pt not found; skip.")

                # C) Restore EMA (shadow dict -> target device/dtype)
                if args.use_ema:
                    # --- UNet EMA ---
                    unet_ema_path = os.path.join(input_dir, "unet_ema.pt")
                    if os.path.exists(unet_ema_path):
                        pkg = torch.load(unet_ema_path, map_location="cpu")
                        raw_shadow = pkg.get("shadow", {})
                        # Filter to current trainable parameter names
                        raw_shadow = filter_shadow_to_trainable(raw_shadow, unw.unet)
                        # Move to the target device/dtype
                        ema_unet.shadow = {
                            k: t.to(ema_unet.device_ema, dtype=ema_unet.dtype).clone()
                            for k, t in raw_shadow.items()
                        }
                        ema_unet.decay = float(pkg.get("decay", ema_unet.decay))
                    else:
                        print("[load_hook] unet_ema.pt not found; skip UNet EMA restore.")

                    # --- ctx EMA ---
                    ctx_ema_path = os.path.join(input_dir, "ctx_encoder_ema.pt")
                    if os.path.exists(ctx_ema_path):
                        pkg = torch.load(ctx_ema_path, map_location="cpu")
                        raw_shadow = pkg.get("shadow", {})
                        raw_shadow = filter_shadow_to_trainable(raw_shadow, unw.ctx)
                        ema_ctx.shadow = {
                            k: t.to(ema_ctx.device_ema, dtype=ema_ctx.dtype).clone()
                            for k, t in raw_shadow.items()
                        }
                        ema_ctx.decay = float(pkg.get("decay", ema_ctx.decay))
                    else:
                        print("[load_hook] ctx_encoder_ema.pt not found; skip ctx EMA restore.")

                
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    
    to_train = []
    for name, param in model.named_parameters():
    # Skip parameters you want to keep frozen (example names; adjust to your module paths)
        if name.endswith("pos_embed") or name.endswith("temp_embed"):
            param.requires_grad = False
            continue
        param.requires_grad = True
        to_train.append(param)
    if args.use_ema:
        ema_unet = CrossDeviceEMA(accelerator.unwrap_model(model).unet,
                          decay=0.999, device_ema=device_encoder)
        
        ema_ctx = CrossDeviceEMA(accelerator.unwrap_model(model).ctx,
                         decay=0.999, device_ema=device_encoder)

    optimizer = optimizer_cls(
        to_train,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open('params_freeze.txt', 'w')
        rec_txt2 = open('params_train.txt', 'w')
        for name, para in model.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    train_dataset = VideoFolderDataset(args.base_folder, width=args.width, height=args.height, sample_frames=args.num_frames, num_samples=args.num_samples)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

   
        
    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("HMPDM", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    
    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = unet.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    loss_records = []
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                # Prepare inputs
                p_sc = args.p_sc
                use_sc = (torch.rand(()) < p_sc)
                pixel_values = batch["pixel_values"].to(weight_dtype).to(device=device_encoder, non_blocking=True)
                bsz = pixel_values.shape[0]  # MODIFIED: batch size
                
                # Split history frames for conditioning (0: F_hist)
    
                # Encode images to VAE latents (all frames)
                
                total_latents   = tensor_to_vae_latent(pixel_values, vae).to(device_unet)  
                    # [B, F_tot, C, H, W]
                F_hist          = args.F_hist
                F_future        = args.F_future
                B               = total_latents.shape[0]
                
                conditional_pixel_values = pixel_values[:, F_hist-1:F_hist, :, :, :]
                # Split
                hist_latents    = total_latents[:, :F_hist]
                       # [B, F_hist, C, H, W]
                future_latents  = total_latents[:, F_hist:]                    # [B, F_future, C, H, W]
                
                # histencoder
                
                #scheduler and timestep
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(future_latents.device)    
                sigmas = sigmas[:, None, None, None, None]
                # 2) Sample noise (future frames only)
                noise           = torch.randn_like(future_latents)               # [B*F_future, C, H, W]
                noisy_future =   future_latents+ noise * sigmas
                t = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)  
                     
                input_noisy_future=noisy_future / ((sigmas**2 + 1) ** 0.5)
                
                tot_noisy_latents = torch.cat([hist_latents, input_noisy_future], dim=1)     
                #condition
                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(device_encoder)
                noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                conditional_pixel_values=conditional_pixel_values.to(weight_dtype)
                
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents.to(accelerator.device)  / vae.config.scaling_factor
                conditional_latents=conditional_latents.unsqueeze(
                    1).repeat(1, total_latents.shape[1], 1, 1, 1)
                
                conditional_latents=conditional_latents.to(device_unet)
                # Classifier-free guidance dropout for text conditioning
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=accelerator.device)  # MODIFIED
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob  # MODIFIED
                    prompt_mask = prompt_mask.unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
                    null_conditioning = torch.zeros_like(encoder_hidden_states)  # [B, dim]
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1)
                    )  # [B,1,dim]  # MODIFIED
                ts = torch.cat([
                    torch.full((bsz,F_hist),-1.0, dtype=torch.float32, device=tot_noisy_latents.device),
                    t.view(bsz, 1).repeat(1, F_future).to(torch.float32)
                ], dim=1)  
                timesteps = ts.flatten()
                added_time_ids = _get_add_time_ids(
                    17, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    tot_noisy_latents.dtype,
                    bsz,
                    
                )
                added_time_ids = added_time_ids.to(total_latents.device) 
                cond_mask = torch.cat([
                    torch.ones(B, F_hist,dtype=total_latents.dtype),    # history frames: 1
                    torch.zeros(B, F_future,dtype=total_latents.dtype)     # future frames: 0
                ], dim=1).to(total_latents.device) 
                
                def forward_with_condition(conditional_latents_input):
                    # inp_noisy_latents = concat along channel dim
                    # Your code: torch.cat([tot_noisy_latents, conditional_latents], dim=2)
                    inp_noisy_latents = torch.cat([tot_noisy_latents, conditional_latents_input], dim=2)
                    # shape: [B, F_tot, C_concat, H, W]

                    model_pred = model(
                        inp_noisy_latents,
                        timesteps,                   # [B*F_tot]
                        added_time_ids=added_time_ids,
                        cond_mask=cond_mask,
                        x_hist=hist_latents
                    ).sample
                    # model_pred: [B, F_tot, C, H, W]

                    pred_future = model_pred[:, F_hist:, :, :, :]  # [B, F_future, C, H, W]

                    # Recover clean future \hat{x}_0
                    c_out  = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip =  1 / (sigmas**2 + 1)
                    denoised_latents = pred_future * c_out + c_skip * noisy_future
                    # shape: [B, F_future, C, H, W]

                    return denoised_latents  # this is all we need

                # ===== 4. First forward: use GT condition (no gradient stop yet) =====
                if use_sc:
                    # ---- Case A: use self-conditioning ----
                    # First forward: with GT condition. We only want its output, no gradient.
                    with torch.no_grad():
                        draft_future = forward_with_condition(conditional_latents)
                        # draft_future: the model's first-pass future prediction, shape [B, F_future, C, H, W]

                    # Build the self-conditioning input:
                    # concatenate hist_latents (real past latents) with draft_future (predicted future latents)
                    # to get [B, F_tot, C, H, W] — same shape as conditional_latents_gt
                    conditional_latents_sc = torch.cat(
                        [hist_latents, draft_future.detach()], dim=1
                    ).to(device_unet)

                    # Second forward: use the self-conditioned input, with gradients, to compute the final loss
                    denoised_final = forward_with_condition(conditional_latents_sc)

                else:
                    # ---- Case B: no self-conditioning ----
                    # Run once with the GT condition; this pass needs gradients, so no torch.no_grad()
                    denoised_final = forward_with_condition(conditional_latents)
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_final.float() -
                     future_latents.float()) ** 2).reshape(future_latents.shape[0], -1),
                    dim=1,    )                       
                loss = loss.mean()
                avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.update(accelerator.unwrap_model(model).unet)
                    ema_ctx.update(accelerator.unwrap_model(model).ctx)
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    loss_records.append((global_step, loss.detach().item()))
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        '''if args.use_ema:
            ema_unet.copy_to(accelerator.unwrap_model(model).unet.parameters())
            
            torch.save(ema_ctx.module.state_dict(),
               os.path.join(save_path, "ctx_encoder_ema.pt"))

        unet = accelerator.unwrap_model(unet)S
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.svdpretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(args.output_dir)'''
        csv_path = os.path.join(args.output_dir, "loss.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"]);
            for step_idx, loss_val in loss_records:
                writer.writerow([step_idx, loss_val])
        print(f"Saved loss curve to {csv_path}")
        
    accelerator.end_training()


if __name__ == "__main__":
    main()
