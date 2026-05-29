"""
Qualitative demo evaluation for HMPDM.

Samples a handful of clips from a dataset and runs N denoising trajectories on
each, all sharing the same initial noise within a trajectory. Useful for
side-by-side comparison of stochastic samples.
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

        # Take the last `sample_frames` frames by default (matches the previous `selected_frames = frames[-sample_frames:]`)
        selected_frames = frames[-self.sample_frames:]

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
    p = argparse.ArgumentParser(description="Qualitative demo evaluation for HMPDM.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint dir containing unet/ and ctx_encoder.pt.")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root directory of clips to sample from (one subfolder per clip).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write generated mp4s.")
    p.add_argument("--scheduler_dir", type=str,
                   default="stabilityai/stable-video-diffusion-img2vid-xt")
    p.add_argument("--vae_path", type=str,
                   default="stabilityai/stable-video-diffusion-img2vid-xt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--F_hist", type=int, default=2)
    p.add_argument("--F_future", type=int, default=28)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--num_clips", type=int, default=1,
                   help="How many clips to sample from the dataset for the demo.")
    p.add_argument("--num_trajectories", type=int, default=1,
                   help="Trajectories per clip; same clip gets re-rendered with different initial noise.")
    p.add_argument("--base_noise_seed", type=int, default=1000)
    p.add_argument("--master_seed", type=int, default=9402)
    p.add_argument("--fps_out", type=int, default=17)
    return p.parse_args()


########################################
# Main inference loop
########################################
def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 1. Load models
    # -------------------------
    unet = TaLCUNet.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float32,
        subfolder="unet",
        low_cpu_mem_usage=False,
    ).to(device)

    ctx = MaPE(input_size=args.height // 8, num_frames=args.F_hist, n_pairs=3).to(device)
    state = torch.load(os.path.join(args.checkpoint, "ctx_encoder.pt"), map_location="cpu")
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
    # 2. Fix random seeds and sample N clips
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

    all_indices = list(range(len(base_dataset)))
    chosen_indices = random.sample(all_indices, min(args.num_clips, len(all_indices)))
    subset_dataset = Subset(base_dataset, chosen_indices)

    test_dataloader = DataLoader(
        subset_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )

    fps_out = args.fps_out
    num_inference_steps = args.num_inference_steps
    OUT_ROOT_BASE = args.output_dir
    os.makedirs(OUT_ROOT_BASE, exist_ok=True)

    # -------------------------
    # 3. Generate trajectories; clips within one trajectory share the same initial noise
    # -------------------------
    base_noise_seed = args.base_noise_seed

    with torch.no_grad():
        # Pre-fetch the entire dataloader once to avoid potential inconsistencies from re-reading disk
        # This also lets us use the first sample to determine the noise shape
        cached_batches = []
        for batch in tqdm(test_dataloader, desc="Caching 256 clips"):
            # batch: {"pixel_values": (1,T,3,H,W) after collate, "video_name": [str]}
            cached_batches.append(batch)

        # Iterate over the 10 trajectories
        for traj_id in range(args.num_trajectories):
            print(f"\n=== Generating trajectory {traj_id} ===")

            # Create the output folder for this trajectory
            OUT_ROOT = os.path.join(OUT_ROOT_BASE, f"traj_{traj_id}")
            os.makedirs(OUT_ROOT, exist_ok=True)

            # ----
            # 4.1 Fix the noise seed using traj_id.
            #     We need future_latents.shape before we can sample the initial noise,
            #     so probe the shape with the first batch (no actual inference / saving).
            # ----
            probe_batch = cached_batches[0]
            pixel_values_probe = probe_batch["pixel_values"].to(torch.float32).to(device)
            bsz_probe = pixel_values_probe.shape[0]  # should be 1
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

            # block: take the first F_hist + F_future = 20 frames
            block_probe = pixel_values_probe[:, 0:total_required_frames]
            total_latents_probe = tensor_to_vae_latent(block_probe, vae)  # [B,20,C,H',W']
            # Future portion
            future_latents_probe = total_latents_probe[:, F_hist:]        # [B,14,C,H',W']

            # Set this trajectory's random seed -> generate the initial noise shared across the trajectory
            traj_seed = base_noise_seed + traj_id
            g = torch.Generator(device=device)
            g.manual_seed(traj_seed)

            # scheduler.init_noise_sigma is a scalar (float or tensor)
            # Original code: noisy_future = torch.randn_like(future_latents) * scheduler.init_noise_sigma
            noise_shape = future_latents_probe.shape  # (B, F_future, C, H', W')

            initial_noise = torch.randn(
                noise_shape,
                dtype=future_latents_probe.dtype,
                device=device,
                generator=g,
            ) * scheduler.init_noise_sigma

            # -------------------------
            # 5. Run inference per clip and save the video
            # -------------------------
            for batch in tqdm(cached_batches, desc=f"Trajectory {traj_id} infer"):
                pixel_values = batch["pixel_values"].to(torch.float32).to(device)
                bsz = pixel_values.shape[0]  # should be 1
                assert bsz == 1, "this script assumes batch_size = 1"

                # Scheduler timestep reset for each clip
                scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

                # block: take the 20 frames we need
                block = pixel_values[:, 0:total_required_frames]  # shape [1,20,3,H,W]

                # -> VAE latent
                total_latents = tensor_to_vae_latent(block, vae)  # [1,20,C,H',W']
                hist_latents  = total_latents[:, :F_hist]         # [1,6,C,H',W']
                future_latents = total_latents[:, F_hist:]        # [1,14,C,H',W']

                # Copy this trajectory's shared noise to the current clip
                noisy_future = initial_noise.clone().to(device)

                # Original behavior: use the 6th frame (index=5) as the conditioning image
                conditional_pixel_values = block[:, F_hist-1:F_hist, :, :, :]  # [1,1,3,H,W]
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor  # [1,C,H',W']
                conditional_latents = conditional_latents.unsqueeze(1).repeat(
                    1, total_latents.shape[1], 1, 1, 1
                )  # [1,20,C,H',W']

                added_time_ids = _get_add_time_ids(
                    17,    # fps
                    127,   # motion_bucket_id fixed at 127
                    0,     # noise_aug_strength == cond_sigmas == 0
                    unet.dtype,
                    bsz,
                    unet,
                ).to(total_latents.device)

                # cond_mask: history frame = 1, future frame = 0
                cond_mask = torch.cat(
                    [
                        torch.ones (bsz, F_hist,   dtype=torch.float32, device=device),
                        torch.zeros(bsz, F_future, dtype=torch.float32, device=device),
                    ],
                    dim=1
                )  # [1,20]

                # Start the diffusion reverse process
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

                # After the reverse process, noisy_future is the synthesized 14-frame future latent
                totaldinal = x0_future  # [1,14,C,H',W']

                # decode
                all_latents = totaldinal / vae.config.scaling_factor  # match the scale used during training
                all_latents = rearrange(all_latents, "b f c h w -> (b f) c h w")
                decoded = vae.decode(all_latents, F_future).sample  # [14*C,H,W]? actually returns (B*F,3,H,W)

                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                video = rearrange(decoded, '(b f) c h w -> b f c h w', b=bsz)  # [1,14,3,H,W]
                video_uint8 = (video * 255).round().to(torch.uint8)           # [1,14,3,H,W]

                # Save mp4
                vid_names = batch["video_name"]
                if not isinstance(vid_names, (list, tuple)):
                    vid_names = [vid_names]

                B, F, C, H, W = video_uint8.shape
                for bb in range(B):
                    video_path = os.path.join(OUT_ROOT, f"{vid_names[bb]}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, fps_out, (W, H))

                    for f in range(F):
                        frame = video_uint8[bb, f].permute(1, 2, 0).cpu().numpy()  # RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)             # ->BGR
                        writer.write(frame)

                    writer.release()
                    #print(f"[traj {traj_id}] ✅ Saved video to: {video_path}")


if __name__ == "__main__":
    main()
