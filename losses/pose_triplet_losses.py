"""Losses for temporally consistent relative-pose triplets."""

import torch


def rotation_geodesic(prediction, target):
    """Mean SO(3) geodesic error between [B,4,4] transforms."""
    relative = prediction[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    trace = relative.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine).mean()


def rotation_chordal(prediction, target):
    relative = prediction[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    identity = torch.eye(3, device=relative.device, dtype=relative.dtype)
    return (
        torch.linalg.matrix_norm(relative - identity, ord="fro", dim=(-2, -1))
        / (2.0**0.5)
    ).mean()


def normalized_translation_l1(prediction, target, min_translation_norm=1.0e-3):
    target_translation = target[..., :3, 3]
    valid = target_translation.norm(dim=-1) >= float(min_translation_norm)
    if not valid.any():
        return prediction[..., :3, 3].sum() * 0.0
    scale = target_translation[valid].norm(dim=-1, keepdim=True)
    return ((prediction[..., :3, 3][valid] - target_translation[valid]) / scale).abs().mean()


def transform_supervision_loss(
    prediction,
    target,
    rotation_weight=1.0,
    rotation_chordal_weight=0.0,
    translation_weight=1.0,
    min_translation_norm=1.0e-3,
):
    rotation = rotation_geodesic(prediction, target)
    chordal = rotation_chordal(prediction, target)
    translation = normalized_translation_l1(
        prediction,
        target,
        min_translation_norm=min_translation_norm,
    )
    total = (
        float(rotation_weight) * rotation
        + float(rotation_chordal_weight) * chordal
        + float(translation_weight) * translation
    )
    return total, {
        "rotation_rad": rotation,
        "rotation_chordal": chordal,
        "translation_relative_l1": translation,
    }


def triplet_pose_loss(
    predicted_01,
    predicted_12,
    predicted_02,
    target_01,
    target_12,
    target_02,
    pair_weight=1.0,
    cycle_weight=1.0,
    rotation_weight=1.0,
    rotation_chordal_weight=0.0,
    translation_weight=1.0,
    min_translation_norm=1.0e-3,
):
    """Supervise a three-frame chain and its direct 0->2 transform.

    Every transform follows the dataset convention x_j = T_j_from_i x_i.
    The transitive prediction therefore is T_2_from_0 = T_2_from_1 @ T_1_from_0.
    """
    edge_losses = []
    edge_details = []
    for prediction, target in (
        (predicted_01, target_01),
        (predicted_12, target_12),
        (predicted_02, target_02),
    ):
        loss, details = transform_supervision_loss(
            prediction,
            target,
            rotation_weight=rotation_weight,
            rotation_chordal_weight=rotation_chordal_weight,
            translation_weight=translation_weight,
            min_translation_norm=min_translation_norm,
        )
        edge_losses.append(loss)
        edge_details.append(details)

    pair_loss = torch.stack(edge_losses).mean()
    composed_02 = predicted_12 @ predicted_01
    cycle_loss, cycle_details = transform_supervision_loss(
        composed_02,
        predicted_02,
        rotation_weight=rotation_weight,
        rotation_chordal_weight=rotation_chordal_weight,
        translation_weight=translation_weight,
        min_translation_norm=min_translation_norm,
    )
    total = float(pair_weight) * pair_loss + float(cycle_weight) * cycle_loss

    def average_detail(key):
        return torch.stack([details[key] for details in edge_details]).mean()

    return total, {
        "pair": pair_loss,
        "pair_rotation_rad": average_detail("rotation_rad"),
        "pair_rotation_chordal": average_detail("rotation_chordal"),
        "pair_translation_relative_l1": average_detail("translation_relative_l1"),
        "cycle": cycle_loss,
        "cycle_rotation_rad": cycle_details["rotation_rad"],
        "cycle_rotation_chordal": cycle_details["rotation_chordal"],
        "cycle_translation_relative_l1": cycle_details["translation_relative_l1"],
    }
