<div align="center">

# HMPDM: Historical Motion Priors-Informed Diffusion Model for Driving Video Prediction

**[Ke Li](https://kelisbu.github.io/)<sup>1</sup>, Tianjia Yang<sup>2</sup>, [Kaidi Liang](https://liangkd.github.io/)<sup>1</sup>, [Xianbiao Hu](https://sites.psu.edu/xbhu/xb-hu/)<sup>2</sup>, [Ruwen Qin](https://sites.google.com/stonybrook.edu/rqin/home)<sup>1</sup>\***

<sup>1</sup> Department of Civil Engineering, Stony Brook University  
<sup>2</sup> Department of Civil Engineering, Pennsylvania State University

[![preprint](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/abs/2603.27371)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

**🎉 Accepted by [IEEE IVS 2026] 🎉**
</div>

<div align="center">
  <img src="img/HMPDM.png" width="90%">
</div>






## 🚀 Quick Start
- **TaLC** (Temporal-aware Latent Conditioning) — concatenates clean latents of the past with noisy latents of the future, and uses two parallel time embeddings (one for clean history, one for noisy future) gated by a binary mask, so the UNet's spatio-temporal attention can attend to history without confusing the noise schedule.
- **MaPE** (Motion-aware Pyramid Encoder) — a hierarchical spatio-temporal transformer that turns the past-frame latents into three multi-scale token sequences `(M1, M2, M3)`. Each is injected as cross-attention memory at the matching depth in the UNet (down1 ↔ M1, down2 ↔ M2, mid / down3 ↔ M3, mirrored in the up path).
- **SC** (Self-Conditioning) — at training time, with probability `p_sc`, the model first runs a no-grad forward pass with the GT condition, then a second forward pass conditioned on its own detached prediction; only the second pass receives gradients. This lets the model "review" its previous estimate during the iterative denoising at test time.

Under the strictly RGB-only setting, HMPDM achieves 28.2% lower FVD than the previous best on the Cityscapes full test set (2 → 28 frames) and is competitive against multimodal methods that use depth or optical flow.

## Repository layout

```
.
├── models.py          # TaLCUNet, MaPE, HMPDM, plus PatchMerging2x2 / Attention / TransformerBlock
├── train.py           # Training script (TaLC + MaPE + SC), checkpoints UNet and MaPE together
├── evaluate.py        # Full-test-set evaluation, one mp4 per clip, argparse driven
├── evaluate_demo.py   # Qualitative demo: sample N clips, run K trajectories on the same noise
├── requirements.txt   # Pinned Python dependencies (PyTorch 2.6 / CUDA 12.4)
├── scheduler/         # EulerDiscreteScheduler config
└── README.md
```

## 1. Environment setup

The pinned versions in `requirements.txt` target CUDA 12.4. We recommend a fresh conda environment with Python 3.10:

```bash
conda create -n hmpdm python=3.10 -y
conda activate hmpdm

pip install -r requirements.txt
```

The first time the scripts run, they will download the base SVD weights from Hugging Face (`stabilityai/stable-video-diffusion-img2vid-xt`). For air-gapped nodes, pre-download the snapshot and pass the local path through `--svdpretrained_model_name_or_path` (training) or `--vae_path` / `--scheduler_dir` (evaluation).

Optional packages: `xformers` (memory-efficient attention, enable via `--enable_xformers_memory_efficient_attention`), `bitsandbytes` (8-bit Adam, `--use_8bit_adam`), `wandb` (`--report_to wandb`), `tensorboard` (default logger).

## 2. Datasets

The paper reports results on:

| Dataset | Train clips | Test clips | Frames per clip | Resolution | Split |
|---------|-------------|------------|-----------------|------------|-------|
| Cityscapes | 2,975 | 1,525 | 30 | 128 × 128 | 2 → 28 |
| KITTI | 759 | 150 | 9 | 128 × 128 | 4 → 5 |

Both training and evaluation expect data laid out as one subdirectory per clip, with sequentially numbered RGB frames inside:

```
<data_dir>/
├── clip_0001/
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── clip_0002/
│   └── ...
```

Each clip directory must contain at least `F_hist + F_future` frames. Frames are sorted by the numeric portion of the filename. `.png`, `.jpg`, and `.jpeg` are accepted.

## 3. Training

The training script jointly trains `TaLCUNet` (a UNet with TaLC) and `MaPE`, with optional EMA and self-conditioning. Following the paper, both are initialized from the SVD pretrained weights and fine-tuned together.

### 3.1 Cityscapes (2 → 28)

```bash
accelerate launch train.py \
    --base_folder /path/to/cityscapes/train \
    --output_dir ./outputs/hmpdm_cityscapes \
    --width 128 --height 128 \
    --F_hist 2 --F_future 28 --num_frames 30 \
    --num_samples 2975 \
    --per_gpu_batch_size 4 \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler cosine --lr_warmup_steps 3165 \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 2 \
    --mixed_precision fp16 \
    --p_sc 0.9 \
    --use_ema \
    --seed 123
```

### 3.2 KITTI (4 → 5)

```bash
accelerate launch train.py \
    --base_folder /path/to/kitti/train \
    --output_dir ./outputs/hmpdm_kitti \
    --width 128 --height 128 \
    --F_hist 4 --F_future 5 --num_frames 9 \
    --num_samples 759 \
    --per_gpu_batch_size 4 \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler cosine \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --mixed_precision fp16 \
    --p_sc 0.9 \
    --use_ema
```

### Key flags

| Flag | Description | Paper default |
|------|-------------|---------------|
| `--F_hist` | Number of past frames (P) | 2 (Cityscapes), 4 (KITTI) |
| `--F_future` | Number of future frames (F) | 28 (Cityscapes), 5 (KITTI) |
| `--num_frames` | Total frames per clip; must equal F_hist + F_future | 30 / 9 |
| `--p_sc` | Self-conditioning probability (W.A.L.T. recommends 0.9) | 0.9 |
| `--learning_rate` | LR after warmup | 2e-5 |
| `--per_gpu_batch_size` | Batch size per device | 4 (paper uses one L40S 48 GB) |
| `--max_train_steps` | Total optimization steps | 1e5 |
| `--use_ema` | Cross-device EMA over the trainable parameters | on |
| `--mixed_precision` | `no` / `fp16` / `bf16` | `fp16` |
| `--gradient_checkpointing` | Trade compute for memory; recommended on smaller GPUs | on |

Resume from the most recent checkpoint with `--resume_from_checkpoint latest`.

### Checkpoint contents

```
checkpoint-XXXXX/
├── unet/                  # diffusers-style TaLCUNet snapshot (config.json + safetensors)
├── ctx_encoder.pt         # MaPE state_dict
├── unet_ema.pt            # (--use_ema) EMA shadow for UNet
└── ctx_encoder_ema.pt     # (--use_ema) EMA shadow for MaPE
```

A `loss.csv` is written under `--output_dir` at the end of training for plotting.

## 4. Evaluation

Both evaluation scripts are argparse-driven; no hard-coded paths.

### 4.1 Full-test-set evaluation

Generates one mp4 per clip in the test set:

```bash
python evaluate.py \
    --checkpoint ./outputs/hmpdm_cityscapes/checkpoint-100000 \
    --data_dir /path/to/cityscapes/test \
    --output_dir ./outputs/hmpdm_cityscapes/eval_full \
    --device cuda:0 \
    --width 128 --height 128 \
    --F_hist 2 --F_future 28 \
    --num_inference_steps 50 \
    --num_trajectories 10
```

`--num_trajectories 10` reproduces the paper's `#T=10` setting (best-of-10 metrics).

### 4.2 Qualitative demo

Samples a small number of clips and renders multiple trajectories on each (different initial noise, same input):

```bash
python evaluate_demo.py \
    --checkpoint ./outputs/hmpdm_cityscapes/checkpoint-100000 \
    --data_dir /path/to/cityscapes/test \
    --output_dir ./outputs/hmpdm_cityscapes/demo \
    --device cuda:0 \
    --num_clips 4 \
    --num_trajectories 5
```

### 4.3 Metrics

The paper reports SSIM, PSNR, LPIPS, and FVD on (a) a randomly sampled subset of 256 clips and (b) the full test set, both across `#T=10` denoising trajectories. Computing these from the generated mp4s requires standard external implementations (e.g. `lpips`, `pytorch-fid`, `frechet-video-distance`); they are not bundled in this repo.

## 5. Implementation notes

- **MaPE input size constraint.** `MaPE` uses three /2 downsampling stages, so its `input_size` must be `height // 8` and divisible by 8. The training script and evaluation scripts wire this up automatically from `--height`.
- **Cross-attention dim.** `MaPE.hidden_size` must equal `UNet.cross_attention_dim` (1024 for SVD-XT). Don't change `hidden_size` unless you also adjust the UNet.
- **Dual time embedding.** `TaLCUNet.time_embedding_cond` is initialized as a deep copy of the pretrained `time_embedding`; both branches are fine-tuned during training, gated by the `cond_mask` derived from history vs. future positions.
- **Self-conditioning at inference.** During evaluation we always feed the model's previous-step `pred_original_sample` as the condition for the next step (concatenated along the channel dimension), so SC is implicit in the loop, not a flag.

## 6. Citation

If you use this code or build on HMPDM, please cite:

```bibtex
@inproceedings{li2026hmpdm,
  title     = {HMPDM: A Diffusion Model for Driving Video Prediction with Historical Motion Priors},
  author    = {Li, Ke and Yang, Tianjia and Liang, Kaidi and Hu, Xianbiao and Qin, Ruwen},
  booktitle = {IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and the SVD training scaffold from [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend). The MaPE block design draws on [Latte](https://github.com/Vchitect/Latte) for the alternating spatial–temporal transformer pattern. Self-conditioning follows the W.A.L.T. recipe.




