import copy

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def infer_patch_size(dark3r_model, fallback=16):
    patch_embed = getattr(dark3r_model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", fallback)
    if isinstance(patch_size, tuple):
        return int(patch_size[0])
    return int(patch_size)


class EventPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, event_voxel):
        tokens = self.proj(event_voxel).flatten(2).transpose(1, 2)
        return self.norm(tokens)


class Dark3REventTokenEncoder(nn.Module):
    """Event encoder that reuses the Dark3R encoder token space.

    Event voxels are patchified into Dark3R-sized tokens and then processed by a
    copy of the Dark3R image encoder blocks. This follows the EventVGGT-style
    idea of using an event student encoder with the same token architecture as
    the RGB teacher, instead of reconstructing images with E2VID first.
    """

    def __init__(
        self,
        dark3r_model,
        in_channels,
        token_dim,
        patch_size=None,
        copy_encoder=True,
        freeze_copied_encoder=False,
    ):
        super().__init__()
        if not hasattr(dark3r_model, "enc_blocks") or not hasattr(dark3r_model, "enc_norm"):
            raise ValueError("dark3r_model must expose enc_blocks and enc_norm")
        self.in_channels = int(in_channels)
        self.out_dim = int(token_dim)
        self.patch_size = infer_patch_size(dark3r_model) if patch_size is None else int(patch_size)
        self.patch_embed = EventPatchEmbed(self.in_channels, self.out_dim, self.patch_size)

        if copy_encoder:
            self.enc_blocks = copy.deepcopy(dark3r_model.enc_blocks)
            self.enc_norm = copy.deepcopy(dark3r_model.enc_norm)
        else:
            self.enc_blocks = dark3r_model.enc_blocks
            self.enc_norm = dark3r_model.enc_norm

        if freeze_copied_encoder:
            self.enc_blocks.requires_grad_(False)
            self.enc_norm.requires_grad_(False)

    def _should_checkpoint(self, block, *args):
        if not getattr(self, "gradient_checkpointing", False):
            return False
        if not torch.is_grad_enabled():
            return False
        if any(torch.is_tensor(arg) and arg.requires_grad for arg in args):
            return True
        return any(param.requires_grad for param in block.parameters())

    def _forward_block(self, block, *args):
        if self._should_checkpoint(block, *args):
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def _positions(self, event_voxel):
        b, _, h, w = event_voxel.shape
        hp = h // self.patch_size
        wp = w // self.patch_size
        ys, xs = torch.meshgrid(
            torch.arange(hp, device=event_voxel.device, dtype=torch.long),
            torch.arange(wp, device=event_voxel.device, dtype=torch.long),
            indexing="ij",
        )
        pos = torch.stack([ys, xs], dim=-1).view(1, hp * wp, 2)
        return pos.repeat(b, 1, 1)

    def forward(self, event_voxel):
        if event_voxel is None:
            raise ValueError("Dark3REventTokenEncoder should not be called with event_voxel=None")
        h, w = event_voxel.shape[-2:]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"event voxel spatial size {(h, w)} must be divisible by patch_size={self.patch_size}"
            )
        x = self.patch_embed(event_voxel)
        pos = self._positions(event_voxel)
        for block in self.enc_blocks:
            x = self._forward_block(block, x, pos)
        return self.enc_norm(x)
