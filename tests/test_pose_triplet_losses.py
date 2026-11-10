import math

import torch

from losses.pose_triplet_losses import triplet_pose_loss


def make_transform(angle_deg, translation):
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transform = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    transform[0, :3, :3] = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[0, :3, 3] = torch.tensor(translation, dtype=torch.float32)
    return transform


def test_triplet_loss_is_zero_for_consistent_ground_truth():
    target_01 = make_transform(12.0, (1.0, 0.2, 0.0))
    target_12 = make_transform(-8.0, (0.5, -0.1, 0.0))
    target_02 = target_12 @ target_01
    loss, details = triplet_pose_loss(
        target_01,
        target_12,
        target_02,
        target_01,
        target_12,
        target_02,
        rotation_chordal_weight=1.0,
    )
    assert loss.item() < 1.0e-6
    assert details["cycle"].item() < 1.0e-6


def test_triplet_loss_penalizes_inconsistent_direct_transform():
    target_01 = make_transform(12.0, (1.0, 0.2, 0.0))
    target_12 = make_transform(-8.0, (0.5, -0.1, 0.0))
    target_02 = target_12 @ target_01
    inconsistent_02 = target_02.clone()
    inconsistent_02[:, :3, 3] += torch.tensor([0.5, 0.0, 0.0])
    loss, details = triplet_pose_loss(
        target_01,
        target_12,
        inconsistent_02,
        target_01,
        target_12,
        target_02,
        rotation_chordal_weight=1.0,
    )
    assert loss.item() > 0.0
    assert details["cycle"].item() > 0.0
