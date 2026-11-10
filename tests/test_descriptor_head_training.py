from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.event_dark3r_student import EventDark3RStudent
from train_event_dark3r import (
    compute_losses,
    save_checkpoint,
    validate_descriptor_head_only_trainability,
)
from training.event_teacher_student import build_event_optimizer, set_event_training_stage
from utils.smoke_event_dark3r import TinyDark3R


class TinySplitHead(nn.Module):
    def __init__(self, token_dim=32, descriptor_dim=8):
        super().__init__()
        self.patch_size = 16
        self.local_feat_dim = descriptor_dim
        self.two_confs = False
        self.head_pts3d = nn.Linear(token_dim, 4)
        self.head_local_features = nn.Linear(
            2 * token_dim, descriptor_dim * self.patch_size**2
        )

    def forward(self, tokens):
        raw = self.head_pts3d(tokens)
        points = torch.cat([raw[..., :2], raw[..., 2:3].abs() + 0.1], dim=-1)
        descriptor = F.normalize(self.head_local_features(tokens), dim=-1)
        return {
            "pts3d": points,
            "conf": raw[..., 3].sigmoid() + 1.0,
            "desc": descriptor,
        }


class TinyDescriptorDark3R(TinyDark3R):
    def __init__(self, token_dim=32):
        super().__init__(token_dim=token_dim)
        self.downstream_head1 = TinySplitHead(token_dim)
        self.downstream_head2 = TinySplitHead(token_dim)

    def _downstream_head(self, head_num, decout, img_shape):
        head = self.downstream_head1 if head_num == 1 else self.downstream_head2
        return head(decout[-1])


def descriptor_config():
    return {
        "event": {"num_bins": 4, "polarity_separate": True},
        "training": {
            "descriptor_head_only": True,
            "checkpoint_mode": "descriptor_head_only",
        },
        "loss": {
            "descriptor_distill": 1.0,
            "normalize_by_active_weight": True,
        },
    }


def test_descriptor_only_step_preserves_geometry_modules():
    torch.manual_seed(41)
    config = descriptor_config()
    student = EventDark3RStudent(TinyDescriptorDark3R(), config=config)
    set_event_training_stage(student, stage=0, partial_finetune_blocks=0)
    validate_descriptor_head_only_trainability(student)

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if not parameter.requires_grad
    }
    descriptor_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }

    batch_size, height, width = 2, 32, 32
    images = torch.rand(batch_size, 2, 3, height, width) * 2.0 - 1.0
    batch = {
        "images": images,
        "images_dark": images,
        "event_voxel": torch.randn(batch_size, 2, 8, height, width),
        "dataset_source": ["dl3dv"] * batch_size,
    }
    student.train()
    student_out = student(batch)
    teacher_out = tuple(
        {"desc": F.normalize(torch.randn_like(pred["desc"]), dim=-1)}
        for pred in student_out
    )
    loss, diagnostics = compute_losses(student_out, teacher_out, batch, config["loss"])
    assert "descriptor_distill_view1" in diagnostics
    assert "descriptor_distill_view2" in diagnostics

    optimizer = build_event_optimizer(student, lr_head=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    for name, parameter in student.named_parameters():
        if name in descriptor_before:
            assert parameter.grad is not None
        else:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, frozen_before[name], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(parameter.detach(), descriptor_before[name])
        for name, parameter in student.named_parameters()
        if name in descriptor_before
    )


def test_descriptor_only_checkpoint_is_compact_and_complete(tmp_path):
    config = descriptor_config()
    student = EventDark3RStudent(TinyDescriptorDark3R(), config=config)
    set_event_training_stage(student, stage=0, partial_finetune_blocks=0)
    optimizer = build_event_optimizer(student, lr_head=1.0e-3)
    args = SimpleNamespace(
        no_save=False,
        init_event_checkpoint="base_night3r.pth",
    )
    save_checkpoint(
        tmp_path,
        student,
        optimizer,
        config,
        args,
        step=200,
        rank=0,
    )

    path = tmp_path / "last_descriptor_head.pth"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["checkpoint_kind"] == "descriptor_head_only"
    assert payload["step"] == 200
    assert "model" not in payload
    assert "optimizer" not in payload
    assert payload["descriptor_head"]
    assert all(
        ".head_local_features." in key for key in payload["descriptor_head"]
    )
