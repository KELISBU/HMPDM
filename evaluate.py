"""
Full-test-set evaluation for HMPDM.

Generates one mp4 per clip in a test set using a trained HMPDM checkpoint.
Streams clips from disk one at a time (no caching).
"""

import argparse
import os
import re
import random

import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

from train import tensor_to_vae_latent
from models import TaLCUNet, MaPE, HMPDM


########################################
# Dataset (keeps the existing VideoFolderDataset structure)
########################################
class VideoFolderDataset(Dataset):
    def __init__(self, base_folder: str, num_samples=100000, width=512, height=256, sample_frames=20):
        self.num_samples = num_samples
        self.base_folder = base_folder
        # Keep only subdirectories (e.g. video_xxx) and sort for reproducibility
        self.folders = sorted([
            d for d in os.listdir(self.base_folder)
            if os.path.isdir(os.path.join(self.base_folder, d))
        ])
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        # Return the number of available folders, not num_samples
        return len(self.folders)

    def __getitem__(self, idx):
        folder_idx    = idx % len(self.folders)
        chosen_folder = self.folders[folder_idx]  # video name
        folder_path   = os.path.join(self.base_folder, chosen_folder)

        # Collect image frames and sort by the numeric portion of the filename
        frames = sorted(
            [
                fn for fn in os.listdir(folder_path)
                if os.path.splitext(fn)[1].lower() in {'.png', '.jpg', '.jpeg'}
            ],
            key=lambda fn: int(re.search(r'(\d+)', fn).group(1)) if re.search(r'(\d+)', fn) else float('inf')
        )

        if len(frames) < self.sample_frames:
            raise ValueError(
                f"Folder '{chosen_folder}' has only {len(frames)} frames, less than the required {self.sample_frames}."
            )

        # Take the first `sample_frames` frames (note: this differs from evaluationsc.py which takes the last ones)
        selected_frames = frames[0:self.sample_frames]

        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width),
            dtype=torch.float32
        )

        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                img = img.convert('RGB')
                img = img.resize((self.width, self.height), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32)
                arr = arr / 127.5 - 1.0   # [-1,1]
                arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)
                pixel_values[i] = torch.from_numpy(arr)

        return {
            "pixel_values": pixel_values,   # (T, 3, H, W)
            "video_name": chosen_folder
        }


########################################
# Helper functions
########################################
def _resolve_ctx_path(checkpoint: str) -> str:
    """Return a local path to ctx_encoder.pt.

    If `checkpoint` is a local directory, just join the filename. Otherwise treat
    it as a HuggingFace Hub repo id (e.g. 'KELISBU/HMPDM-Cityscapes') and download
    the file via huggingface_hub.
    """
    local_path = os.path.join(checkpoint, "ctx_encoder.pt")
    if os.path.isfile(local_path):
        return local_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=checkpoint, filename="ctx_encoder.pt")


def _get_add_time_ids(
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
    unet,
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
            f"but a vector of {passed_add_embed_dim} was created. "
            "Please check `unet.config.time_embedding_type` and "
            "`text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids


########################################
# CLI
########################################
def parse_args():
    p = argparse.ArgumentParser(description="Full-test-set evaluation for HMPDM.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint dir containing unet/ and ctx_encoder.pt.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory of the test set (one subfolder per clip).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write generated mp4s.")
    p.add_argument("--scheduler_dir", type=str,
                   default="stabilityai/stable-video-diffusion-img2vid-xt",
                   help="Path or HF id holding the EulerDiscreteScheduler config (subfolder='scheduler').")
    p.add_argument("--vae_path", type=str,
                   default="stabilityai/stable-video-diffusion-img2vid-xt",
                   help="Path or HF id of the VAE.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--F_hist", type=int, default=2)
    p.add_argument("--F_future", type=int, default=28)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--num_trajectories", type=int, default=1,
                   help="Number of denoising trajectories to sample (paper uses 10 for #T=10 metrics).")
    p.add_argument("--num_samples", type=int, default=None,
                   help="If set, randomly sample this many clips from the test set instead of "
                        "running the full set. Sampling is reproducible via --master_seed. "
                        "Paper's '256 Random Samples' protocol: --num_samples 256.")
    p.add_argument("--base_noise_seed", type=int, default=1000)
    p.add_argument("--master_seed", type=int, default=9402)
    p.add_argument("--fps_out", type=int, default=17)
    p.add_argument("--num_workers", type=int, default=1)
    return p.parse_args()


########################################
# Main inference loop
########################################
def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 1) Load models
    # -------------------------
    unet = TaLCUNet.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float32,
        subfolder="unet",
        low_cpu_mem_usage=False,
    ).to(device)

    ctx = MaPE(input_size=args.height // 8, num_frames=args.F_hist).to(device)
    state = torch.load(_resolve_ctx_path(args.checkpoint), map_location="cpu")
    ctx.load_state_dict(state, strict=True)
    ctx = ctx.to(device=device, dtype=torch.float32).eval()

    model = HMPDM(unet, ctx).to(device).eval()

    scheduler = EulerDiscreteScheduler.from_pretrained(args.scheduler_dir, subfolder="scheduler")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.vae_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float32,
    ).to(device)

    # -------------------------
    # 2) Data
    # -------------------------
    random.seed(args.master_seed)
    np.random.seed(args.master_seed)
    torch.manual_seed(args.master_seed)

    F_hist = args.F_hist
    F_future = args.F_future
    total_required_frames = F_hist + F_future

    base_dataset = VideoFolderDataset(
        args.data_dir,
        num_samples=len(os.listdir(args.data_dir)),
        width=args.width,
        height=args.height,
        sample_frames=total_required_frames,
    )

    # Optional: randomly pick N clips for the paper's "256 Random Samples" protocol.
    # Sampling is reproducible via --master_seed (already seeded above).
    if args.num_samples is not None and args.num_samples < len(base_dataset):
        all_indices = list(range(len(base_dataset)))
        chosen_indices = random.sample(all_indices, args.num_samples)
        eval_dataset = Subset(base_dataset, chosen_indices)
        print(f">>> Sampled {len(eval_dataset)}/{len(base_dataset)} clips (seed={args.master_seed})")
    else:
        eval_dataset = base_dataset
        print(f">>> Using all {len(eval_dataset)} clips")

    test_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    fps_out = args.fps_out
    num_inference_steps = args.num_inference_steps
    OUT_ROOT_BASE = args.output_dir
    os.makedirs(OUT_ROOT_BASE, exist_ok=True)

    # -------------------------
    # 3) Generate trajectories; clips within one trajectory share the same initial noise
    # -------------------------
    base_noise_seed = args.base_noise_seed

    with torch.no_grad():
        # Use the iterator to grab the first sample as a probe to determine the noise shape;
        # the rest of the samples are processed afterwards.
        data_iter = iter(test_dataloader)
        try:
            first_batch = next(data_iter)
        except StopIteration:
            print("Dataset is empty.")
            return

        # Set the timesteps (the probe needs this too)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        pixel_values_probe = first_batch["pixel_values"].to(torch.float32).to(device)  # [1,T,3,H,W]
        block_probe = pixel_values_probe[:, :total_required_frames]                    # [1,30,3,H,W]
        total_latents_probe = tensor_to_vae_latent(block_probe, vae)                  # [1,30,C,H',W']
        future_latents_probe = total_latents_probe[:, F_hist:]                        # [1,28,C,H',W']
        noise_shape = future_latents_probe.shape

        # Use --num_trajectories to control how many denoising trajectories to sample
        for traj_id in range(args.num_trajectories):
            print(f"\n=== Generating trajectory {traj_id} (all clips) ===")
            OUT_ROOT = os.path.join(OUT_ROOT_BASE, f"traj_{traj_id}")
            os.makedirs(OUT_ROOT, exist_ok=True)

            # Fix the initial noise for this trajectory
            traj_seed = base_noise_seed + traj_id
            g = torch.Generator(device=device).manual_seed(traj_seed)
            # Note: set_timesteps must be called before init_noise_sigma is available
            # (already set above during the probe; it is also set before each clip below)
            initial_noise_template = torch.randn(
                noise_shape, dtype=future_latents_probe.dtype, device=device, generator=g
            ) * scheduler.init_noise_sigma

            # Process the first_batch first
            def process_one_batch(batch):
                pixel_values = batch["pixel_values"].to(torch.float32).to(device)   # [1,T,3,H,W]
                bsz = pixel_values.shape[0]
                assert bsz == 1, "this script assumes batch_size = 1"

                # Reset timesteps for every clip
                scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

                block = pixel_values[:, :total_required_frames]                     # [1,30,3,H,W]
                total_latents = tensor_to_vae_latent(block, vae)                    # [1,30,C,H',W']
                hist_latents  = total_latents[:, :F_hist]                           # [1,2,C,H',W']

                # All clips share this trajectory's initial noise
                noisy_future = initial_noise_template.clone()

                # Conditioning frame (use the last history frame; with F_hist=2 that is one frame)
                conditional_pixel_values = block[:, F_hist-1:F_hist, :, :, :]            # [1,1,3,H,W]
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0]
                conditional_latents = conditional_latents / vae.config.scaling_factor  # [1,C,H',W']
                conditional_latents = conditional_latents.unsqueeze(1).repeat(
                    1, total_latents.shape[1], 1, 1, 1
                )  # [1,30,C,H',W']

                added_time_ids = _get_add_time_ids(
                    10, 127, 0, unet.dtype, bsz, unet
                ).to(total_latents.device)

                cond_mask = torch.cat(
                    [
                        torch.ones (bsz, F_hist,   dtype=torch.float32, device=device),
                        torch.zeros(bsz, F_future, dtype=torch.float32, device=device),
                    ],
                    dim=1
                )  # [1,30]

                # Diffusion reverse process
                for t in scheduler.timesteps:
                
                    timestep = torch.cat([
                        torch.full((bsz,F_hist),-1.04, dtype=torch.float32, device=total_latents.device),
                        t.view(bsz, 1).repeat(1, F_future).to(torch.float32)
                    ], dim=1).flatten()
                    
                    noi=scheduler.scale_model_input(noisy_future, t)
                    tot_noisy_latents = torch.cat([hist_latents, noi], dim=1) 
                    
                    inp_noisy_latents=torch.cat([tot_noisy_latents, conditional_latents], dim=2) 
                    #print(inp_noisy_latents.dtype)
                    model_pred = model(inp_noisy_latents, timestep,added_time_ids=added_time_ids,cond_mask=cond_mask,x_hist=hist_latents).sample
                    model_pred_fu=model_pred[:,F_hist:]
                    
                    step_out=scheduler.step(model_pred_fu, t, noisy_future)
                    noisy_future=step_out.prev_sample
                    x0_future = step_out.pred_original_sample
                    conditional_latents=torch.cat([hist_latents, x0_future], dim=1)

                # Decode the 28 future frames
                future_latents = noisy_future / vae.config.scaling_factor
                bfchw = rearrange(future_latents, "b f c h w -> (b f) c h w")
                decoded = vae.decode(bfchw, F_future).sample                       # [(1*28),3,H,W]
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                video = rearrange(decoded, "(b f) c h w -> b f c h w", b=bsz)      # [1,28,3,H,W]
                video_uint8 = (video * 255).round().to(torch.uint8)

                # Save mp4
                vid_names = batch["video_name"]      # the dataloader puts str inside a list by default
                if not isinstance(vid_names, (list, tuple)):
                    vid_names = [vid_names]

                B, F, C, H, W = video_uint8.shape
                for bb in range(B):
                    video_path = os.path.join(OUT_ROOT, f"{vid_names[bb]}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, fps_out, (W, H))
                    for f in range(F):
                        frame = video_uint8[bb, f].permute(1, 2, 0).cpu().numpy()   # RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)              # -> BGR
                        writer.write(frame)
                    writer.release()

            # Process the first sample
            process_one_batch(first_batch)

            # Process the remaining samples (no caching, read one by one)
            for batch in tqdm(data_iter, desc=f"Trajectory {traj_id} infer (all clips)"):
                process_one_batch(batch)



if __name__ == "__main__":
    main()
