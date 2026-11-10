import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.event_voxel import build_event_voxel
from dataset.lowlight_augmentation import ContinuousLowLightAug
from losses.distillation_losses import (
    feature_distillation_loss,
    structure_consistency_loss,
)
from losses.event_geometry_losses import (
    event_depth_edge_loss,
    supervised_depth_gt_loss,
)
from model.event_dark3r_student import EventDark3RStudent


class TinyPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, token_dim=32, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, token_dim, patch_size, stride=patch_size)

    def forward(self, img, true_shape=None):
        b, _, h, w = img.shape
        tokens = self.proj(img).flatten(2).transpose(1, 2)
        hp = h // self.patch_size
        wp = w // self.patch_size
        ys, xs = torch.meshgrid(
            torch.arange(hp, device=img.device),
            torch.arange(wp, device=img.device),
            indexing="ij",
        )
        pos = torch.stack([ys, xs], dim=-1).view(1, hp * wp, 2).repeat(b, 1, 1)
        return tokens, pos


class TinyEncoderBlock(nn.Module):
    def __init__(self, token_dim=32, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, x, pos):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class TinyDark3R(nn.Module):
    def __init__(self, token_dim=32):
        super().__init__()
        self.patch_embed = TinyPatchEmbed(token_dim=token_dim)
        self.enc_blocks = nn.ModuleList([TinyEncoderBlock(token_dim)])
        self.enc_norm = nn.LayerNorm(token_dim)
        self.decoder_embed = nn.Linear(token_dim, token_dim)
        self.dec_norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 4)

    def _encode_image_pairs(self, img1, img2, shape1, shape2):
        feat1 = self._encode(img1)
        feat2 = self._encode(img2)
        pos1 = torch.zeros_like(feat1[..., :2])
        pos2 = torch.zeros_like(feat2[..., :2])
        return feat1, feat2, pos1, pos2

    def _encode(self, img):
        tokens, pos = self.patch_embed(img)
        for block in self.enc_blocks:
            tokens = block(tokens, pos)
        return self.enc_norm(tokens)

    def _decoder(self, feat1, pos1, feat2, pos2):
        dec1 = self.dec_norm(self.decoder_embed(feat1 + feat2.mean(dim=1, keepdim=True)))
        dec2 = self.dec_norm(self.decoder_embed(feat2 + feat1.mean(dim=1, keepdim=True)))
        return [feat1, dec1], [feat2, dec2]

    def _downstream_head(self, head_num, decout, img_shape):
        tokens = decout[-1]
        b, n, _ = tokens.shape
        side = int(n**0.5)
        raw = self.head(tokens).view(b, side, side, 4)
        pts = torch.cat([raw[..., :2], raw[..., 2:3].abs() + 0.1], dim=-1)
        return {
            "pts3d": pts,
            "conf": raw[..., 3].sigmoid() + 1.0,
            "desc": tokens,
            "feature": tokens,
        }

    def forward(self, view1, view2):
        feat1, feat2, pos1, pos2 = self._encode_image_pairs(
            view1["img"], view2["img"], view1["true_shape"], view2["true_shape"]
        )
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        res1 = self._downstream_head(1, dec1, view1["true_shape"])
        res2 = self._downstream_head(2, dec2, view2["true_shape"])
        res2["pts3d_in_other_view"] = res2.pop("pts3d")
        return res1, res2


def main():
    torch.manual_seed(0)
    b, h, w = 2, 32, 32
    images = torch.rand(b, 2, 3, h, w)
    aug = ContinuousLowLightAug(return_metadata=True)
    images_dark, metadata = aug(images)

    events = torch.tensor(
        [
            [0, 2, 3, 0.0, 1],
            [0, 4, 5, 0.3, -1],
            [0, 6, 7, 1.0, 1],
            [1, 8, 9, 0.2, 0],
            [1, 10, 11, 0.8, 1],
        ],
        dtype=torch.float32,
    )
    event_voxel = build_event_voxel(events, h, w)
    assert event_voxel.shape == (b, 8, h, w), event_voxel.shape

    batch = {
        "images": images,
        "images_dark": images_dark,
        "event_voxel": event_voxel,
        "depth_gt": torch.rand(b, h // 4, w // 4) + 0.5,
        "depth_source": "real",
    }

    teacher = TinyDark3R()
    teacher.eval().requires_grad_(False)

    view1 = {
        "img": images[:, 0],
        "true_shape": torch.tensor([[h, w]], dtype=torch.int32).repeat(b, 1),
    }
    view2 = {
        "img": images[:, 1],
        "true_shape": torch.tensor([[h, w]], dtype=torch.int32).repeat(b, 1),
    }
    with torch.no_grad():
        teacher_out = teacher(view1, view2)
    baseline_student = EventDark3RStudent(TinyDark3R(), config={"event": {"num_bins": 4}})
    student_out = baseline_student(batch)

    loss = student_out[0]["pts3d"].mean()
    loss_dict = {}
    feat_loss, loss_dict = feature_distillation_loss(
        student_out[0].get("feature", student_out[0]["desc"]),
        teacher_out[0].get("feature", teacher_out[0]["desc"]),
        loss_dict=loss_dict,
    )
    struct_loss, loss_dict = structure_consistency_loss(
        student_out, teacher_out, loss_dict=loss_dict
    )
    edge_loss, loss_dict = event_depth_edge_loss(
        student_out[0], event_voxel, loss_dict=loss_dict
    )
    depth_loss, loss_dict = supervised_depth_gt_loss(
        student_out[0], batch, loss_dict=loss_dict
    )
    total = loss + feat_loss + struct_loss + edge_loss + depth_loss
    total.backward()

    full_cfg = {
        "event": {"num_bins": 4, "polarity_separate": True},
        "event_norm": {"enabled": True, "use_log_count": True, "use_nonzero_norm": True},
        "event_stat": {"enabled": True, "stat_dim": 3},
        "event_film": {"enabled": True, "hidden_dim": 16},
        "fusion": {
            "gate_type": "tensor",
            "use_event_stat": True,
            "hidden_dim": 32,
            "event_scale_init": 0.1,
            "gate_bias_init": -2.0,
        },
    }
    full_student = EventDark3RStudent(TinyDark3R(), config=full_cfg)
    full_out = full_student(batch)
    full_loss = full_out[0]["pts3d"].mean() + full_out[1]["pts3d_in_other_view"].mean()
    full_loss.backward()

    gate_summary = full_out[0]["event_gate_summary"]
    stat_summary = full_out[0]["event_stat_summary"]
    gate_tensor = full_out[0]["event_gate_tensor"]
    gate_map = full_out[0]["event_gate_map"]
    event_stat = full_out[0]["event_stat"]
    assert full_out[0]["event_fusion_type"] == "tensor"
    assert full_out[0]["event_film_enabled"] is True
    assert full_out[0]["event_norm_enabled"] is True
    assert gate_tensor.ndim == 3 and gate_tensor.shape[:2] == event_stat.shape[:2]
    assert event_stat.ndim == 3 and event_stat.shape[-1] == 3
    assert gate_map.ndim == 3 and gate_map.shape[0] == gate_tensor.shape[0]
    assert gate_summary["min"] >= 0.0 and gate_summary["max"] <= 1.0
    assert "event_density_mean" in stat_summary
    assert torch.isfinite(gate_tensor).all()
    assert torch.isfinite(gate_map).all()
    assert torch.isfinite(event_stat).all()
    assert torch.isfinite(torch.tensor(list(gate_summary.values()))).all()
    assert torch.isfinite(torch.tensor(list(stat_summary.values()))).all()
    assert full_student.event_film.mlp[-1].weight.grad is not None
    assert full_student.fusion_adapter.gate[-1].weight.grad is not None

    print("event_voxel_shape", tuple(event_voxel.shape))
    print("lowlight_metadata", metadata)
    print("event_enabled", student_out[0]["event_enabled"])
    print("event_gate", student_out[0]["event_gate"])
    print("full_event_gate_summary", gate_summary)
    print("full_event_stat_summary", stat_summary)
    print("full_event_scale", full_out[0]["event_scale"])
    print("loss_dict", loss_dict)
    print("backward_ok", True)
    print("film_tensor_gate_backward_ok", True)


if __name__ == "__main__":
    main()
