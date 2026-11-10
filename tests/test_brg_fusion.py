import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.event_fusion import (
    AdditiveEventFusion,
    ConcatMLPEventFusion,
    Night3RBoundedEventFusion,
    ResidualIdentityDensityFiLM,
)


def _cfg(**overrides):
    cfg = {
        "stat_dim": 3,
        "num_sources": 3,
        "use_source_bias": True,
        "use_source_stat_norm": True,
        "use_channel_modulation": False,
        "use_gate_regularization": True,
        "event_scale_init": 0.05,
        "event_scale_max": 0.2,
        "gate_bias_init": -3.5,
        "gate_cap_start": 0.2,
        "gate_cap_final": 1.0,
        "gate_cap_warmup_steps": 5000,
        "gate_mean_cap_start": 0.05,
        "gate_mean_cap_final": 0.30,
        "gate_mean_cap_warmup_steps": 10000,
    }
    cfg.update(overrides)
    return cfg


def _inputs(dtype=torch.float32):
    torch.manual_seed(7)
    b, n, c = 2, 16, 32
    rgb = torch.randn(b, n, c, dtype=dtype)
    event = torch.randn(b, n, c, dtype=dtype)
    event_stat = torch.zeros(b, n, 3, dtype=dtype)
    event_stat[..., 0] = torch.rand(b, n, dtype=dtype)
    event_stat[..., 1] = 1.0
    event_stat[..., 2] = torch.rand(b, n, dtype=dtype)
    return rgb, event, event_stat


def test_brg_shapes_and_initial_gate():
    rgb, event, event_stat = _inputs()
    fusion = Night3RBoundedEventFusion(32, cfg=_cfg())
    fused, diag, aux = fusion(
        rgb,
        event,
        event_stat,
        source_id=torch.tensor([0, 2]),
        step=0,
        return_diag=True,
    )
    assert fused.shape == rgb.shape
    assert aux["total"].ndim == 0
    assert math.isclose(float(diag["event_scale"]), 0.05, rel_tol=1e-4, abs_tol=1e-4)
    assert 0.005 <= float(diag["gate_mean"]) <= 0.03
    assert float(diag["gate_sat_high"]) == 0.0
    assert float(diag["channel_dev"]) == 0.0
    assert torch.isfinite(diag["residual_ratio"])


def test_brg_no_event_mask_identity():
    rgb, event, event_stat = _inputs()
    event_stat[..., 1] = 0.0
    fusion = Night3RBoundedEventFusion(32, cfg=_cfg())
    fused, diag, _ = fusion(rgb, event, event_stat, source_id=None, step=0, return_diag=True)
    assert torch.allclose(fused, rgb, atol=0.0, rtol=0.0)
    assert float(diag["gate_mean"]) == 0.0
    assert float(diag["gate_max"]) == 0.0


def test_brg_source_id_optional_and_channel_modulation():
    rgb, event, event_stat = _inputs()
    fusion = Night3RBoundedEventFusion(
        32,
        cfg=_cfg(
            use_channel_modulation=True,
            channel_alpha_start=0.0,
            channel_alpha_final=0.10,
            channel_alpha_max=0.10,
        ),
    )
    fused_none, _, _ = fusion(rgb, event, event_stat, source_id=None, step=None, return_diag=True)
    fused_src, diag_src, _ = fusion(
        rgb,
        event,
        event_stat,
        source_id=torch.tensor([0, 1]),
        step=None,
        return_diag=True,
    )
    assert fused_none.shape == rgb.shape
    assert fused_src.shape == rgb.shape
    assert float(diag_src["channel_dev"]) == 0.0


def test_brg_without_event_stats_accepts_none():
    rgb, event, _ = _inputs()
    fusion = Night3RBoundedEventFusion(32, cfg=_cfg(use_event_stat=False))
    fused, diag, _ = fusion(rgb, event, None, source_id=None, step=0, return_diag=True)
    assert fused.shape == rgb.shape
    assert torch.isfinite(fused).all()
    assert float(diag["gate_mean"]) > 0.0


def test_brg_replace_mode_removes_rgb_identity_path():
    rgb, event, event_stat = _inputs()
    fusion = Night3RBoundedEventFusion(32, cfg=_cfg(output_mode="replace"))
    fused, _, _ = fusion(rgb, event, event_stat, source_id=None, step=0, return_diag=True)
    assert fused.shape == rgb.shape
    assert not torch.allclose(fused, rgb)
    assert fused.norm() < rgb.norm()


def test_brg_event_stat_component_mask():
    rgb, event, event_stat = _inputs()
    fusion = Night3RBoundedEventFusion(
        32,
        cfg=_cfg(event_stat_mask=[1.0, 0.0, 0.0]),
    )
    fused, diag, _ = fusion(rgb, event, event_stat, source_id=None, step=0, return_diag=True)
    assert torch.equal(fusion.event_stat_mask.flatten(), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.isfinite(fused).all()
    assert float(diag["gate_mean"]) > 0.0


def test_brg_bf16_forward_backward_finite():
    rgb, event, event_stat = _inputs(dtype=torch.bfloat16)
    rgb = rgb.requires_grad_(True)
    event = event.requires_grad_(True)
    fusion = Night3RBoundedEventFusion(32, cfg=_cfg())
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        fused, _, aux = fusion(rgb, event, event_stat, source_id=None, step=0, return_diag=True)
        loss = fused.float().pow(2).mean() + aux["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(rgb.grad.float()).all()
    assert torch.isfinite(event.grad.float()).all()


def test_residual_identity_film_initial_identity():
    rgb, _, event_stat = _inputs()
    film = ResidualIdentityDensityFiLM(32, stat_dim=3, eta_init=0.0, eta_max=0.1)
    out = film(rgb, event_stat)
    assert torch.allclose(out, rgb, atol=0.0, rtol=0.0)


def test_concat_mlp_fusion_shape_and_event_dependence():
    rgb, event, _ = _inputs()
    fusion = ConcatMLPEventFusion(32, hidden_dim=48, event_scale_init=0.035)
    fused, scale = fusion(rgb, event, return_gate=True)
    fused_zero = fusion(rgb, torch.zeros_like(event))
    assert fused.shape == rgb.shape
    assert scale.shape == (rgb.shape[0], rgb.shape[1], 1)
    assert torch.isfinite(fused).all()
    assert not torch.allclose(fused, fused_zero)


def test_additive_fusion_shape_and_zero_event_behavior():
    rgb, event, _ = _inputs()
    fusion = AdditiveEventFusion(32, event_scale_init=0.035)
    fused, scale = fusion(rgb, event, return_gate=True)
    fused_zero = fusion(rgb, torch.zeros_like(event))
    assert fused.shape == rgb.shape
    assert scale.shape == (rgb.shape[0], rgb.shape[1], 1)
    assert torch.isfinite(fused).all()
    assert torch.allclose(fused_zero, rgb, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_brg_shapes_and_initial_gate()
    test_brg_no_event_mask_identity()
    test_brg_source_id_optional_and_channel_modulation()
    test_brg_without_event_stats_accepts_none()
    test_brg_replace_mode_removes_rgb_identity_path()
    test_brg_event_stat_component_mask()
    test_brg_bf16_forward_backward_finite()
    test_residual_identity_film_initial_identity()
    test_concat_mlp_fusion_shape_and_event_dependence()
    test_additive_fusion_shape_and_zero_event_behavior()
