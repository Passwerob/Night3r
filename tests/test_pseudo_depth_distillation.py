import torch

from losses.event_geometry_losses import pseudo_depth_distillation_loss


def test_pseudo_depth_loss_is_invariant_to_global_scale():
    target = torch.linspace(1.0, 4.0, 64).reshape(1, 8, 8)
    prediction = {"depth": (target * 3.0).unsqueeze(1)}
    batch = {
        "depth_pseudo": target,
        "depth_pseudo_valid": torch.ones_like(target, dtype=torch.bool),
        "depth_pseudo_conf": torch.ones_like(target),
    }

    loss, diagnostics = pseudo_depth_distillation_loss(prediction, batch, weight=1.0)

    assert loss.item() < 1e-6
    assert abs(diagnostics["pseudo_depth_median_scale"] - (1.0 / 3.0)) < 1e-5


def test_pseudo_depth_loss_penalizes_shape_error():
    target = torch.linspace(1.0, 4.0, 64).reshape(1, 8, 8)
    distorted = target.clone()
    distorted[:, :, 4:] *= 1.8
    prediction = {"depth": distorted.unsqueeze(1)}
    batch = {
        "depth_pseudo": target,
        "depth_pseudo_valid": torch.ones_like(target, dtype=torch.bool),
        "depth_pseudo_conf": torch.ones_like(target),
    }

    loss, _ = pseudo_depth_distillation_loss(prediction, batch, weight=1.0)

    assert loss.item() > 0.01


def test_pseudo_depth_loss_skips_missing_teacher_depth():
    prediction = {"depth": torch.ones(1, 1, 8, 8)}

    loss, diagnostics = pseudo_depth_distillation_loss(prediction, {}, weight=1.0)

    assert loss.item() == 0.0
    assert diagnostics["skipped pseudo_depth"] == "missing pseudo depth"
