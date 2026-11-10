import copy
from types import SimpleNamespace

import torch

from dataset.dsec_event_dataset import relative_pose_camera2_from_camera1
from losses.event_geometry_losses import pose_gt_loss
from model.event_dark3r_student import EventDark3RStudent
from training.event_teacher_student import (
    build_event_optimizer,
    set_event_training_stage,
)
from train_event_dark3r import save_checkpoint, validate_pose_head_only_trainability
from utils.eval_night3r_trajectory import load_pose_head_checkpoint
from utils.smoke_event_dark3r import TinyDark3R


def _pose_head_config():
    return {
        "event": {"num_bins": 4, "polarity_separate": True},
        "pose_head": {
            "enabled": True,
            "hidden_dim": 48,
            "pool_hidden_dim": 16,
            "dropout": 0.0,
        },
        "training": {"pose_head_only": True},
    }


def test_relative_pose_direction_and_dl3dv_axis_conversion():
    t_world_camera1 = torch.eye(4)
    t_world_camera1[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    t_world_camera2 = torch.eye(4)
    angle = torch.tensor(0.2)
    c, s = torch.cos(angle), torch.sin(angle)
    t_world_camera2[:3, :3] = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    t_world_camera2[:3, 3] = torch.tensor([-0.5, 0.25, 1.0])

    native = relative_pose_camera2_from_camera1(
        t_world_camera1, t_world_camera2
    )
    point_camera1 = torch.tensor([0.2, -0.1, 4.0, 1.0])
    point_world = t_world_camera1 @ point_camera1
    point_camera2 = torch.linalg.inv(t_world_camera2) @ point_world
    torch.testing.assert_close(native @ point_camera1, point_camera2)

    opencv = relative_pose_camera2_from_camera1(
        t_world_camera1,
        t_world_camera2,
        source="dl3dv",
        output_convention="opencv",
    )
    axis_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    torch.testing.assert_close(opencv, axis_flip @ native @ axis_flip)


def test_pose_loss_ignores_uninformative_tiny_translation():
    pred_pose = torch.eye(4).unsqueeze(0).requires_grad_(True)
    pred_pose = pred_pose.clone()
    pred_pose[:, :3, 3] = torch.tensor([0.2, -0.1, 0.3])
    pose_gt = torch.eye(4).unsqueeze(0)
    pose_gt[:, 0, 3] = 1.0e-5
    loss, diagnostics = pose_gt_loss(
        {"relative_pose": pred_pose},
        {"relative_pose_gt": pose_gt},
        cycle_weight=0.0,
        min_translation_norm=1.0e-3,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert diagnostics["pose_translation_valid_fraction"] == 0.0


def test_pose_chordal_rotation_constraint_has_expected_scale_and_gradient():
    angle = torch.tensor(0.2)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    pred_pose = torch.eye(4).unsqueeze(0)
    pred_pose[:, :3, :3] = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    pred_pose.requires_grad_(True)
    pose_gt = torch.eye(4).unsqueeze(0)

    loss, diagnostics = pose_gt_loss(
        {"relative_pose": pred_pose},
        {"relative_pose_gt": pose_gt},
        cycle_weight=0.0,
        rotation_weight=0.0,
        rotation_chordal_weight=1.0,
        translation_weight=0.0,
    )

    expected = 2.0 * torch.sin(angle / 2.0)
    torch.testing.assert_close(loss, expected)
    assert diagnostics["pose_rotation_chordal"] == expected.item()
    loss.backward()
    assert pred_pose.grad is not None
    assert torch.isfinite(pred_pose.grad).all()


def test_pose_only_step_changes_only_pose_head():
    torch.manual_seed(5)
    student = EventDark3RStudent(TinyDark3R(), config=_pose_head_config())
    set_event_training_stage(student, stage=0, partial_finetune_blocks=0)
    validate_pose_head_only_trainability(student)
    frozen_before = {
        name: param.detach().clone()
        for name, param in student.named_parameters()
        if not name.startswith("pose_head.")
    }
    pose_before = {
        name: param.detach().clone()
        for name, param in student.pose_head.named_parameters()
    }

    batch_size, height, width = 2, 32, 32
    images = torch.rand(batch_size, 2, 3, height, width) * 2.0 - 1.0
    pose_gt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_gt[:, 0, 3] = torch.tensor([0.1, -0.15])
    batch = {
        "images": images,
        "images_dark": images,
        "event_voxel": torch.randn(batch_size, 2, 8, height, width),
        "relative_pose_gt": pose_gt,
    }

    optimizer = build_event_optimizer(student, lr_pose_head=1.0e-3)
    student.train()
    pred1, pred2 = student(batch)
    assert "relative_pose" in pred1
    assert "pts3d" not in pred1
    assert "pts3d_in_other_view" not in pred2
    loss, _ = pose_gt_loss(
        pred1, batch, cycle_weight=0.0, min_translation_norm=1.0e-3
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    for name, param in student.named_parameters():
        if name.startswith("pose_head."):
            assert param.grad is not None
        else:
            assert param.grad is None
            torch.testing.assert_close(param, frozen_before[name], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(param.detach(), pose_before[name])
        for name, param in student.pose_head.named_parameters()
    )


def test_pose_head_only_checkpoint_round_trip(tmp_path):
    torch.manual_seed(11)
    config = _pose_head_config()
    config["training"]["checkpoint_mode"] = "pose_head_only"
    student = EventDark3RStudent(TinyDark3R(), config=config)
    set_event_training_stage(student, stage=0, partial_finetune_blocks=0)
    optimizer = build_event_optimizer(student, lr_pose_head=1.0e-3)
    args = SimpleNamespace(
        no_save=False,
        init_event_checkpoint="base_night3r.pth",
    )
    filename = "pose_head_step00000200.pth"
    save_checkpoint(
        tmp_path,
        student,
        optimizer,
        config,
        args,
        step=200,
        rank=0,
        name=filename,
    )

    checkpoint_path = tmp_path / filename
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["checkpoint_kind"] == "pose_head_only"
    assert payload["step"] == 200
    assert "pose_head" in payload
    assert "model" not in payload
    assert "optimizer" not in payload

    torch.manual_seed(23)
    restored = EventDark3RStudent(TinyDark3R(), config=config)
    load_info = load_pose_head_checkpoint(restored, checkpoint_path)
    assert load_info["step"] == 200
    assert load_info["missing_keys"] == []
    assert load_info["unexpected_keys"] == []
    for key, value in student.pose_head.state_dict().items():
        torch.testing.assert_close(restored.pose_head.state_dict()[key], value)
