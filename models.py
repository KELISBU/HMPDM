"""
Model definitions for HMPDM (Historical Motion Priors-informed Diffusion Model).

Modules:
    TaLCUNet -- UNet that implements Temporal-aware Latent Conditioning (TaLC).
                Extends SVD's UNetSpatioTemporalConditionModel with a separate
                time-embedding branch for clean history vs. noisy future frames,
                plus per-stage gating for the multi-scale memory tokens.
    MaPE     -- Motion-aware Pyramid Encoder. A hierarchical spatio-temporal
                transformer that converts the latent of P past frames into
                three multi-scale token sequences (M1, M2, M3) used as cross-
                attention memory in the UNet.
    HMPDM    -- Top-level wrapper that combines TaLCUNet + MaPE.

Self-Conditioning (SC) is implemented in the training loop, not here.
"""

from copy import deepcopy
from typing import Optional, Union

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.utils import BaseOutput, logging
from timm.models.vision_transformer import PatchEmbed, Mlp



# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
logger = logging.get_logger(__name__) 
def _dbg(tag, x):
    # Avoid massive output: only print at rank==0 or in a debug environment
    if not torch.is_tensor(x):          # guard against non-tensor inputs
        print(tag, x)
        return
    print(f"{tag:12s}", tuple(x.shape), x.dtype,
          "min", float(x.min()), "max", float(x.max()),
          "finite", torch.isfinite(x).all().item())
class TaLCUNet(UNetSpatioTemporalConditionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Assumes model_channels == self.config.block_out_channels[0]
        self.time_embedding_cond = deepcopy(self.time_embedding)
        self.enc_norm = nn.LayerNorm(self.config.cross_attention_dim)
# Per-stage gates for enc1/enc2/enc3 (adjustable; suggested to decay across stages)
        self.enc_gate = nn.Parameter(torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32))
        # Attention "temperature" tau (>1 softens, <1 sharpens)
        self.tau = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def load_state_dict(self, state_dict, strict: bool = True):
        # Let the parent class load checkpoint weights into time_embedding first
        super().load_state_dict(state_dict, strict)
        # Then copy the weights to the auxiliary branch
        self.time_embedding_cond.load_state_dict(self.time_embedding.state_dict())
        return None
    def _prep_kv(self, enc: torch.Tensor, gate_idx: int):
    # LayerNorm + gate + tau scaling (equivalent to scaling the logits temperature)
        return self.enc_norm(enc) * (self.enc_gate[gate_idx] / self.tau.clamp_min(1e-3))
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        added_time_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,                 # single-input port (kept for backward compatibility)
        encoder_hidden_states_multi = None,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, tuple]:
        # match original signature and combine history/future time_proj
        # 1. Handle timesteps input and broadcast to [B, F]
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        
        cond_mask_flat = cond_mask.reshape(-1, 1).to(t_emb.device)
    
        
        emb = self.time_embedding_cond(t_emb) * cond_mask_flat +  self.time_embedding(t_emb) * (1 - cond_mask_flat) 
        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        aug_emb=aug_emb.repeat_interleave(num_frames, dim=0, output_size=aug_emb.shape[0] * num_frames)
        emb = emb + aug_emb
        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        if encoder_hidden_states_multi is not None:
            assert isinstance(encoder_hidden_states_multi, (list, tuple)) and len(encoder_hidden_states_multi) >= 3
            enc1, enc2, enc3 = encoder_hidden_states_multi[:3]
            # Move to the current device/dtype
            enc1 = enc1.to(device=sample.device, dtype=sample.dtype)
            enc2 = enc2.to(device=sample.device, dtype=sample.dtype)
            enc3 = enc3.to(device=sample.device, dtype=sample.dtype)
            # Repeat to [B*F, Li, D]
            enc1_bf = enc1.repeat_interleave(num_frames, dim=0, output_size=enc1.shape[0] * num_frames)
            enc2_bf = enc2.repeat_interleave(num_frames, dim=0, output_size=enc2.shape[0] * num_frames)
            enc3_bf = enc3.repeat_interleave(num_frames, dim=0, output_size=enc3.shape[0] * num_frames)
            enc1_bf = self._prep_kv(enc1_bf, 0)
            enc2_bf = self._prep_kv(enc2_bf, 1)
            enc3_bf = self._prep_kv(enc3_bf, 2)
        else:
            # Compatibility: if only one encoder output is provided, reuse it for all three stages
            if encoder_hidden_states is None:
                raise ValueError("Provide either `encoder_hidden_states_multi` or `encoder_hidden_states`.")
            enc_single = encoder_hidden_states.to(device=sample.device, dtype=sample.dtype)
            enc_single_bf = enc_single.repeat_interleave(num_frames, dim=0, output_size=enc_single.shape[0] * num_frames)
            enc1_bf = enc2_bf = enc3_bf = enc_single_bf
                # Repeat the embeddings num_video_frames times
                # emb: [batch, channels] -> [batch * frames, channels]
                # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        
        # 2. pre-process
        sample = self.conv_in(sample)
        
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            enc_for_block = enc3_bf
            if i == 0:
                enc_for_block = enc1_bf
            elif i == 1:
                enc_for_block = enc2_bf
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=enc_for_block,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=enc3_bf,
            image_only_indicator=image_only_indicator,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            if getattr(upsample_block, "has_cross_attention", False):
                if i == 1:
                    enc_for_block = enc3_bf
                elif i == 2:
                    enc_for_block = enc2_bf
                elif i == 3:
                    enc_for_block = enc1_bf
                else:
                    enc_for_block = None  # up0
            else:
                enc_for_block = None
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=enc_for_block,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)




# Must be available in scope:
# get_2d_sincos_pos_embed(embed_dim, grid_size)
# get_1d_sincos_temp_embed(embed_dim, length)
# PatchEmbed (your version; recommended flatten=True, with norm)

import torch
import torch.nn as nn
from einops import rearrange
class PatchMerging2x2(nn.Module):
    def __init__(self, dim_in: int, dim_out: Optional[int] = None, with_norm: bool = True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out or dim_in  # keep the dim unchanged so cross-attn can be applied directly
        self.norm = nn.LayerNorm(4 * dim_in) if with_norm else nn.Identity()
        self.reduction = nn.Linear(4 * dim_in, self.dim_out, bias=True)

    @torch.no_grad()
    def _check(self, g: int):
        assert g % 2 == 0, f"grid {g} must be even for 2x2 merging"

    def forward(self, tokens: torch.Tensor, B: int, F: int, g_in: int):
        # tokens: [B*F, N=g_in*g_in, D]
        self._check(g_in)
        BF, N, D = tokens.shape
        x = tokens.view(B*F, g_in, g_in, D)   # NHWC

        x0 = x[:, 0::2, 0::2, :]  # (BF, g/2, g/2, D)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)  # (BF, g/2, g/2, 4D)

        x_cat = self.norm(x_cat)
        x_red = self.reduction(x_cat)                # (BF, g/2, g/2, D_out)

        g_out = g_in // 2
        out = x_red.view(B*F, g_out * g_out, -1)     # (BF, N/4, D_out)
        return out, g_out

class MaPE(nn.Module):
    """
    Input:   x_hist: (B, F, C, H, W)
    Output:  [enc1, enc2, enc3]
             enc1: (B, F*N1, D)  # H/2,  W/2
             enc2: (B, F*N2, D)  # H/4,  W/4
             enc3: (B, F*N3, D)  # H/8,  W/8
    Constraint: D == UNet.cross_attention_dim
    """
    def __init__(
        self,
        input_size: int = 32,          # per-frame H = W; must match x_hist resolution and be divisible by 8
        in_channels: int = 4,
        hidden_size: int = 1024,       # = D (must equal UNet.cross_attention_dim)
        num_frames: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        n_pairs=3,
        attention_mode: str = "math",
        train_patch: bool = True,
        use_pos_scale: bool = False,    # optional: learnable position/time scale gate, more stable
    ):
        super().__init__()
        assert input_size % 8 == 0, "Three rounds of /2 downsampling require H,W divisible by 8"

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_frames  = num_frames
        self.n_pairs=n_pairs
        # Stage 1: enter from pixels (patch=2)
       
        self.patch1 = PatchEmbed(img_size=input_size, patch_size=2,
                                 in_chans=in_channels, embed_dim=hidden_size, bias=True)

        # Stage 2 / 3: use 2x2 Patch Merging
        self.merge2 = PatchMerging2x2(dim_in=hidden_size, dim_out=hidden_size, with_norm=True)
        self.merge3 = PatchMerging2x2(dim_in=hidden_size, dim_out=hidden_size, with_norm=True)

        if not train_patch:
            for p in self.patch1.parameters(): p.requires_grad = False
            for p in self.merge2.parameters(): p.requires_grad = False
            for p in self.merge3.parameters(): p.requires_grad = False

        # Grid sizes and token counts
        self.g1 = input_size // 2
        self.g2 = input_size // 4
        self.g3 = input_size // 8
        self.N1, self.N2, self.N3 = self.g1*self.g1, self.g2*self.g2, self.g3*self.g3

        # Frozen 2D / 1D sin-cos positional embeddings
        self.pos1 = nn.Parameter(torch.zeros(1, self.N1, hidden_size), requires_grad=False)
        self.pos2 = nn.Parameter(torch.zeros(1, self.N2, hidden_size), requires_grad=False)
        self.pos3 = nn.Parameter(torch.zeros(1, self.N3, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)

        # (Optional) position/time scale gate to prevent the positional signal from being too strong
        if use_pos_scale:
            self.pos_scale  = nn.Parameter(torch.tensor(0.1))
            self.time_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer("pos_scale",  torch.tensor(1.0), persistent=False)
            self.register_buffer("time_scale", torch.tensor(1.0), persistent=False)

        # Each stage: 3 x (Spatial + Temporal)
        def make_stage_blocks(n_pairs):
            blocks = nn.ModuleList()
            for _ in range(n_pairs):
                blocks.append(TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                                               attention_mode=attention_mode))  # spatial
                blocks.append(TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                                               attention_mode=attention_mode))  # temporal
            return blocks

        self.blocks1 = make_stage_blocks(self.n_pairs)
        self.blocks2 = make_stage_blocks(self.n_pairs)
        self.blocks3 = make_stage_blocks(self.n_pairs)

        self.initialize_weights()

        # Extra safety checks (run once before training)
        self._sanity_assert = True

    # ----- Initialize the frozen sin-cos embeddings and patch1 weights -----
    def initialize_weights(self):
       
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ----- Single stage: + pos -> (Spatial -> Temporal) x 3 -----
    def _forward_stage(self, x_tokens, B, F, pos_embed, grid, blocks):
        # x_tokens: (B*F, N, D)
        x = x_tokens #+ self.pos_scale * pos_embed.to(x_tokens.dtype)

        for i in range(0, len(blocks), 2):
            spatial_block, temp_block = blocks[i:i+2]

            # Spatial (within a frame)
            x = spatial_block(x)  # (B*F, N, D)

            # Temporal (across frames): treat N as the sequence, F as the batch grouping
            x = rearrange(x, '(b f) n d -> (b n) f d', b=B, f=F)
            if i == 0:  # only add the 1D time position before the first temporal block
                x = x #+ self.time_scale * self.temp_embed.to(x.dtype)
            x = temp_block(x)
            x = rearrange(x, '(b n) f d -> (b f) n d', b=B, n=grid*grid)

        x_feat = rearrange(x, '(b f) n d -> b f n d', b=B, f=F)  # (B, F, N, D)
        return x, x_feat

    def forward(self, x_hist: torch.Tensor):
        """
        x_hist: (B, F, C, H, W)
        return: [enc1, enc2, enc3], each enc*: (B, F*N*, D)
        """
        B, F, C, H, W = x_hist.shape
        assert F == self.num_frames, f"num_frames mismatch: {F} vs {self.num_frames}"
        assert H == self.input_size and W == self.input_size, \
            f"input_size mismatch: got {(H,W)}, expect {(self.input_size, self.input_size)}"

        # Stage 1
        x = rearrange(x_hist, 'b f c h w -> (b f) c h w')
        x = self.patch1(x)  # (B*F, N1, D)
        x, x_feat = self._forward_stage(x, B, F, self.pos1, self.g1, self.blocks1)
        enc1 = rearrange(x_feat, 'b f n d -> b (f n) d')  # (B, F*N1, D)

        # Stage 2: 2x2 merge
        x, g2 = self.merge2(x, B=B, F=F, g_in=self.g1)   # (B*F, N2, D)
        assert g2 == self.g2 or not self._sanity_assert
        x, x_feat = self._forward_stage(x, B, F, self.pos2, g2, self.blocks2)
        enc2 = rearrange(x_feat, 'b f n d -> b (f n) d')  # (B, F*N2, D)

        # Stage 3: merge again
        x, g3 = self.merge3(x, B=B, F=F, g_in=g2)        # (B*F, N3, D)
        assert g3 == self.g3 or not self._sanity_assert
        x, x_feat = self._forward_stage(x, B, F, self.pos3, g3, self.blocks3)
        enc3 = rearrange(x_feat, 'b f n d -> b (f n) d')  # (B, F*N3, D)

        return [enc1, enc2, enc3]

    
class ResidualFlowFusion3D(nn.Module):
    """
    Concat([x, flow]) -> 1x1x1 Conv -> residual add:
        x_tilde = x + alpha * Conv1x1x1([x; flow])
    Input/output tensor layout is [B, F, C, H, W] (matching the current training code).
    Set use_flow_only=True if you want to project only the flow (without x).
    """
    def __init__(self, c_latent: int = 8, c_flow: int = 4,
                 alpha_init: float = 0.2, use_flow_only: bool = False):
        super().__init__()
        self.use_flow_only = use_flow_only
        in_ch = c_flow if use_flow_only else (c_latent + c_flow)

        # 3D 1x1x1 projection (zero-init so it does not perturb the pretrained path initially)
        self.proj = nn.Conv3d(in_ch, c_latent, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # alpha gate (learnable). Also supports overriding via alpha_override at forward time.
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=True)

    @torch.no_grad()
    def set_alpha(self, value: float):
        self.alpha.fill_(float(value))

    def forward(self,
                x_bfchw: torch.Tensor,          # [B, F, C, H, W]      -- noisy latents
                flow_bfchw: torch.Tensor,       # [B, F, C_flow, H, W] -- flow latents (same scale/channels as x)
                alpha_override: float | None = None,
                p_drop: float | None = None,
                training: bool = True) -> torch.Tensor:
        """
        p_drop: conditioning dropout probability (if provided and training=True, zeroes the flow with this probability)
        alpha_override: if provided, overrides the learnable alpha (useful for annealing or inference control)
        """
        assert x_bfchw.ndim == 5 and flow_bfchw.ndim == 5, "expect [B,F,C,H,W]"
        B, F, C, H, W = x_bfchw.shape
        Cf = flow_bfchw.shape[2]
        assert flow_bfchw.shape[:2] == (B, F) and flow_bfchw.shape[3:] == (H, W), "flow and x must have matching shape/frame count"

        flow = flow_bfchw
        if (p_drop is not None) and training:
            # Per-batch random drop of the entire flow branch (could also be per-sample)
            keep = (torch.rand((), device=flow.device) > p_drop).float()
            flow = keep * flow

        # Conv3d expects [B,C,T,H,W]: swap F and C so F becomes the temporal dim
        x_bcthw    = x_bfchw.permute(0, 2, 1, 3, 4).contiguous()
        flow_bcthw = flow.permute(0, 2, 1, 3, 4).contiguous()

        if self.use_flow_only:
            z = flow_bcthw                                   # [B, Cf, T, H, W]
        else:
            z = torch.cat([x_bcthw, flow_bcthw], dim=1)      # [B, C+Cf, T, H, W]

        delta = self.proj(z)                                  # [B, C, T, H, W]
        alpha = self.alpha if alpha_override is None else torch.as_tensor(alpha_override, dtype=delta.dtype, device=delta.device)
        alpha = torch.clamp(alpha, min=0.0)                  # prevent negative gating

        x_tilde = x_bcthw + alpha * delta
        return x_tilde.permute(0, 2, 1, 3, 4).contiguous()    # back to [B, F, C, H, W]


'''class HMPDM(nn.Module):
    def __init__(self,
                 unet,
                 ctx_encoder,                 # MaPE
                 # New: residual projection config (adjust c_latent to match your VAE latent channels)
                 c_latent: int = 4,
                 c_flow: int = 4,
                 alpha_init: float = 0.2,
                 use_flow_only: bool = False):
        super().__init__()
        self.unet = unet
        self.ctx  = ctx_encoder  # saved/loaded together and managed together by the optimizer

        # New: residual flow projection module (zero-init + alpha gate)
        self.flow_fuser = ResidualFlowFusion3D(
            c_latent=c_latent, c_flow=c_flow,
            alpha_init=alpha_init, use_flow_only=use_flow_only
        )

    @torch.no_grad()
    def set_flow_alpha(self, value: float):
        """External setter for alpha (e.g. training annealing / inference toggle)."""
        self.flow_fuser.set_alpha(value)

    def forward(self,
                inp_noisy_latents,            # [B, F, C, H, W] -- noisy latents (hist + future) from the training loop
                timesteps,
                added_time_ids,
                cond_mask,
                x_hist,                       # [B, F_hist, C, H, W]
                # New: optional flow latent (same frame count/scale/channels as inp_noisy_latents)
                flow_latents: torch.Tensor | None = None,    # [B, F, C_flow, H, W]
                # New: optional gate and dropout (default to internal settings if not provided)
                flow_alpha: float | None = None,
                flow_p_drop: float | None = None,
                **extra):
        """
        When flow_latents=None or flow_alpha=0, the model falls back to the no-flow path;
        when flow_latents is provided and flow_alpha>0, the residual projection fusion is applied.
        """
        # 1) History frames -> spatio-temporal context
        enc_states = self.ctx(x_hist)  # (B, L, D_ctx) -- MaPE output

        # 2) Residual projection (if flow is provided)
        if (flow_latents is not None) and (flow_alpha is None or flow_alpha > 0):
            sample = self.flow_fuser(
                x_bfchw=inp_noisy_latents,
                flow_bfchw=flow_latents,
                alpha_override=flow_alpha,             # external annealing override
                p_drop=flow_p_drop,
                training=self.training
            )
        else:
            sample = inp_noisy_latents

        # 3) Call the underlying UNet
        out = self.unet(
            sample=sample,                      # still pass [B, F, C, H, W] (matches the existing layout)
            timestep=timesteps,
            encoder_hidden_states=None,
            encoder_hidden_states_multi=enc_states,
            added_time_ids=added_time_ids,
            cond_mask=cond_mask,
            **extra
        )
        return out'''
class HMPDM(nn.Module):
    def __init__(self, unet, ctx_encoder: MaPE, cross_attention_dim=None):
        super().__init__()
        self.unet = unet
        self.ctx  = ctx_encoder  # registered as a submodule => saved/loaded and optimized together



    def forward(self, inp_noisy_latents, timesteps,added_time_ids,cond_mask, x_hist, **extra):
        # 1) History frames -> spatio-temporal pos tokens
        enc_states = self.ctx(x_hist)  # (B, L, D_ctx)


        out = self.unet(
            sample=inp_noisy_latents,                # (B, C_lat, F, H, W)
            timestep=timesteps,                   # diffusion step
            encoder_hidden_states=None,
            encoder_hidden_states_multi=enc_states,
            added_time_ids=added_time_ids,
            cond_mask=cond_mask,    # history context (K/V)
            **extra
        )
        # diffusers usually returns an object; the caller takes .sample
        return out


def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class TransformerBlock(nn.Module):
    """
    A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.gamma1 = nn.Parameter(torch.ones(1) * 1e-5)
        self.gamma2 = nn.Parameter(torch.ones(1) * 1e-5)

    def forward(self, x):
        
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x
    


