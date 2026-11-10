import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sigmoid(value, eps=1e-6):
    value = float(max(min(value, 1.0 - eps), eps))
    return math.log(value / (1.0 - value))


def _stat_buffer(value, stat_dim, default, rows=None):
    if value is None:
        if rows is None:
            tensor = torch.full((stat_dim,), float(default), dtype=torch.float32)
        else:
            tensor = torch.full((rows, stat_dim), float(default), dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if rows is None:
            if tensor.numel() != stat_dim:
                raise ValueError(f"Expected {stat_dim} event-stat values, got {tensor.numel()}")
            tensor = tensor.reshape(stat_dim)
        else:
            if tensor.ndim == 1:
                if tensor.numel() == rows and stat_dim == 1:
                    tensor = tensor.reshape(rows, 1)
                elif tensor.numel() == stat_dim:
                    tensor = tensor.reshape(1, stat_dim).expand(rows, stat_dim).clone()
                else:
                    raise ValueError(
                        f"Expected {stat_dim} values or {rows} source values, got {tensor.numel()}"
                    )
            elif tensor.shape != (rows, stat_dim):
                raise ValueError(
                    f"Expected source event-stat values shaped {(rows, stat_dim)}, "
                    f"got {tuple(tensor.shape)}"
                )
    return tensor


def normalize_event_voxel(
    event_voxel,
    use_log_count=True,
    use_nonzero_norm=True,
    eps=1e-6,
):
    """Optionally compress and normalize an event voxel without changing shape."""
    if event_voxel is None:
        return None
    if use_log_count:
        event_voxel = torch.sign(event_voxel) * torch.log1p(event_voxel.abs())
    if not use_nonzero_norm:
        return event_voxel

    if event_voxel.ndim == 3:
        mask = event_voxel != 0
        if mask.any():
            values = event_voxel[mask]
            mean = values.mean()
            std = values.std(unbiased=False)
            event_voxel = torch.where(mask, (event_voxel - mean) / (std + eps), event_voxel)
        return event_voxel

    if event_voxel.ndim != 4:
        raise ValueError("event_voxel must be [C,H,W] or [B,C,H,W]")

    out = event_voxel.clone()
    for batch_idx in range(out.shape[0]):
        value = out[batch_idx]
        mask = value != 0
        if not mask.any():
            continue
        nonzero = value[mask]
        mean = nonzero.mean()
        std = nonzero.std(unbiased=False)
        out[batch_idx] = torch.where(mask, (value - mean) / (std + eps), value)
    return out


def compute_event_stats(
    event_voxel,
    target_hw,
    dt=None,
    eps=1e-6,
    polarity_separate=True,
):
    """Return token-level event statistics shaped [B, Ht*Wt, D]."""
    if event_voxel is None:
        return None
    if event_voxel.ndim == 3:
        event_voxel = event_voxel.unsqueeze(0)
    if event_voxel.ndim != 4:
        raise ValueError("event_voxel must be [C,H,W] or [B,C,H,W]")

    if len(target_hw) != 2:
        raise ValueError("target_hw must be a pair")
    target_hw = (int(target_hw[0]), int(target_hw[1]))
    if target_hw[0] <= 0 or target_hw[1] <= 0:
        raise ValueError(f"Invalid target_hw={target_hw}")

    abs_count = event_voxel.abs().sum(dim=1, keepdim=True)
    density = torch.log1p(abs_count)
    occupancy = (abs_count > 0).to(dtype=event_voxel.dtype)

    if polarity_separate and event_voxel.shape[1] % 2 == 0:
        half = event_voxel.shape[1] // 2
        pos = event_voxel[:, :half].abs().sum(dim=1, keepdim=True)
        neg = event_voxel[:, half:].abs().sum(dim=1, keepdim=True)
    else:
        pos = torch.clamp(event_voxel, min=0).sum(dim=1, keepdim=True)
        neg = torch.clamp(-event_voxel, min=0).sum(dim=1, keepdim=True)
    polarity_balance = (pos - neg).abs() / (pos + neg + eps)

    stats = torch.cat([density, occupancy, polarity_balance], dim=1)
    stats = F.adaptive_avg_pool2d(stats, target_hw)
    stats = stats.flatten(2).transpose(1, 2).contiguous()

    if dt is not None:
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=event_voxel.device, dtype=event_voxel.dtype)
        dt = dt.to(device=event_voxel.device, dtype=event_voxel.dtype)
        if dt.ndim == 0:
            dt = dt.view(1, 1, 1).expand(event_voxel.shape[0], stats.shape[1], 1)
        elif dt.ndim == 1:
            dt = dt.view(-1, 1, 1).expand(-1, stats.shape[1], 1)
        elif dt.ndim == 2:
            dt = dt.unsqueeze(-1)
        if dt.shape[0] == 1 and event_voxel.shape[0] > 1:
            dt = dt.expand(event_voxel.shape[0], -1, -1)
        if dt.shape[1] == 1 and stats.shape[1] > 1:
            dt = dt.expand(-1, stats.shape[1], -1)
        stats = torch.cat([stats, dt], dim=-1)
    return stats


def compute_visibility_confidence(
    image,
    target_hw,
    dark_thr=0.08,
    sat_thr=0.95,
    exposure_scale=0.08,
    contrast_tau=0.06,
    texture_tau=0.03,
    artifact_tau=2.0,
    smooth_kernel=5,
    eps=1e-6,
):
    """Return token-level low-light visibility confidence shaped [B, Ht*Wt, 1].

    The score is a bounded RGB reliability cue, not a signal-to-noise ratio. It
    combines exposure health, local contrast, gradient visibility, and a simple
    high-frequency artifact penalty. Inputs may be in [0, 1] or MASt3R-style
    [-1, 1]; the result is detached by the caller when used as a gate cap.
    """
    if image is None:
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("image must be [C,H,W] or [B,C,H,W]")
    if len(target_hw) != 2:
        raise ValueError("target_hw must be a pair")
    target_hw = (int(target_hw[0]), int(target_hw[1]))
    if target_hw[0] <= 0 or target_hw[1] <= 0:
        raise ValueError(f"Invalid target_hw={target_hw}")

    rgb = image.float()
    sample_min = rgb.amin(dim=(1, 2, 3), keepdim=True)
    rgb = torch.where(sample_min < -0.05, 0.5 * (rgb + 1.0), rgb).clamp(0.0, 1.0)
    if rgb.shape[1] >= 3:
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    else:
        gray = rgb.mean(dim=1, keepdim=True)

    mean = F.adaptive_avg_pool2d(gray, target_hw)
    mean2 = F.adaptive_avg_pool2d(gray.pow(2), target_hw)
    std = torch.sqrt((mean2 - mean.pow(2)).clamp_min(0.0) + eps)

    grad_x = gray[..., :, 1:] - gray[..., :, :-1]
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = gray[..., 1:, :] - gray[..., :-1, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
    grad = F.adaptive_avg_pool2d(grad_mag, target_hw)

    kernel = max(int(smooth_kernel), 1)
    if kernel % 2 == 0:
        kernel += 1
    local_mean = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
    high_freq = (gray - local_mean).abs()
    high_freq = F.adaptive_avg_pool2d(high_freq, target_hw)
    artifact_proxy = high_freq / (std + 0.02)

    exposure = torch.sigmoid((mean - float(dark_thr)) / float(exposure_scale))
    exposure = exposure * torch.sigmoid((float(sat_thr) - mean) / float(exposure_scale))
    contrast = 1.0 - torch.exp(-std / max(float(contrast_tau), eps))
    texture = 1.0 - torch.exp(-grad / max(float(texture_tau), eps))
    artifact_ok = torch.exp(-artifact_proxy / max(float(artifact_tau), eps))
    visibility = exposure * torch.maximum(contrast, texture) * artifact_ok
    visibility = visibility.clamp(0.0, 1.0)
    return visibility.flatten(2).transpose(1, 2).contiguous()


class DensityFiLM(nn.Module):
    """Legacy density-conditioned FiLM for event tokens.

    This is kept for ablations. Its initial output is LN(x), not x, so it is
    intentionally not the default Night3R FiLM path.
    """

    def __init__(self, dim, stat_dim=3, hidden_dim=128):
        super().__init__()
        self.dim = int(dim)
        self.stat_dim = int(stat_dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.stat_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, event_token, event_stat):
        if event_stat is None:
            raise ValueError("event_stat is required for DensityFiLM")
        if event_token.ndim != 3 or event_stat.ndim != 3:
            raise ValueError("event_token and event_stat must be [B,N,C/D]")
        if event_token.shape[:2] != event_stat.shape[:2]:
            raise ValueError(
                f"event_token/event_stat token mismatch: {event_token.shape} vs {event_stat.shape}"
            )
        if event_stat.shape[-1] != self.stat_dim:
            raise ValueError(f"Expected event_stat dim {self.stat_dim}, got {event_stat.shape[-1]}")
        event_token = self.norm(event_token)
        gamma, beta = self.mlp(event_stat).chunk(2, dim=-1)
        return (1.0 + gamma) * event_token + beta


class ResidualIdentityDensityFiLM(nn.Module):
    """Density-FiLM variant with identity initialization.

    The residual form keeps the event token distribution unchanged at init:
    out = x + eta * (gamma * LN(x) + beta).  The gamma/beta head is zero-init,
    and eta is bounded to avoid a large hidden rewrite of the event branch.
    """

    def __init__(
        self,
        dim,
        stat_dim=3,
        hidden_dim=128,
        eta_init=0.0,
        eta_max=0.1,
    ):
        super().__init__()
        self.dim = int(dim)
        self.stat_dim = int(stat_dim)
        self.eta_max = float(eta_max)
        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.stat_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.eta = nn.Parameter(torch.tensor(float(eta_init), dtype=torch.float32))

    def forward(self, event_token, event_stat):
        if event_stat is None:
            raise ValueError("event_stat is required for ResidualIdentityDensityFiLM")
        if event_token.ndim != 3 or event_stat.ndim != 3:
            raise ValueError("event_token and event_stat must be [B,N,C/D]")
        if event_token.shape[:2] != event_stat.shape[:2]:
            raise ValueError(
                f"event_token/event_stat token mismatch: {event_token.shape} vs {event_stat.shape}"
            )
        if event_stat.shape[-1] != self.stat_dim:
            raise ValueError(f"Expected event_stat dim {self.stat_dim}, got {event_stat.shape[-1]}")
        dtype = event_token.dtype
        with torch.amp.autocast(device_type=event_token.device.type, enabled=False):
            x = event_token.float()
            stat = event_stat.float()
            gamma, beta = self.mlp(stat).chunk(2, dim=-1)
            eta = torch.clamp(self.eta.float(), min=0.0, max=self.eta_max)
            out = x + eta * (gamma * self.norm(x) + beta)
        return out.to(dtype)


class EventStatNorm(nn.Module):
    """Feature-wise event-stat z-score normalization with optional source stats."""

    def __init__(
        self,
        stat_dim=3,
        num_sources=None,
        init_mean=None,
        init_std=None,
        source_init_mean=None,
        source_init_std=None,
        clip=5.0,
        learnable_affine=True,
    ):
        super().__init__()
        self.stat_dim = int(stat_dim)
        self.num_sources = None if num_sources is None else int(num_sources)
        self.clip = None if clip is None else float(clip)

        global_mean = _stat_buffer(init_mean, self.stat_dim, default=0.0).view(1, 1, self.stat_dim)
        global_std = _stat_buffer(init_std, self.stat_dim, default=1.0).view(1, 1, self.stat_dim)
        self.register_buffer("mean", global_mean)
        self.register_buffer("std", torch.clamp(global_std, min=1e-4))

        if self.num_sources is not None:
            source_mean = _stat_buffer(
                source_init_mean if source_init_mean is not None else init_mean,
                self.stat_dim,
                default=0.0,
                rows=self.num_sources,
            ).view(self.num_sources, 1, self.stat_dim)
            source_std = _stat_buffer(
                source_init_std if source_init_std is not None else init_std,
                self.stat_dim,
                default=1.0,
                rows=self.num_sources,
            ).view(self.num_sources, 1, self.stat_dim)
            self.register_buffer("source_mean", source_mean)
            self.register_buffer("source_std", torch.clamp(source_std, min=1e-4))
        else:
            self.source_mean = None
            self.source_std = None

        if learnable_affine:
            self.affine = nn.Parameter(torch.ones(self.stat_dim, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(self.stat_dim, dtype=torch.float32))
        else:
            self.register_buffer("affine", torch.ones(self.stat_dim, dtype=torch.float32))
            self.register_buffer("bias", torch.zeros(self.stat_dim, dtype=torch.float32))

    def _source_stats(self, event_stat, source_id):
        batch_size = event_stat.shape[0]
        mean = self.mean.expand(batch_size, -1, -1)
        std = self.std.expand(batch_size, -1, -1)
        if self.num_sources is None or source_id is None:
            return mean, std

        ids = torch.as_tensor(source_id, device=event_stat.device).long().view(-1)
        if ids.numel() != batch_size:
            return mean, std
        valid = (ids >= 0) & (ids < self.num_sources)
        if not valid.any():
            return mean, std

        mean = mean.clone()
        std = std.clone()
        mean[valid] = self.source_mean[ids[valid]]
        std[valid] = self.source_std[ids[valid]]
        return mean, std

    def forward(self, event_stat, source_id=None):
        if event_stat is None:
            raise ValueError("event_stat is required for EventStatNorm")
        if event_stat.ndim != 3 or event_stat.shape[-1] != self.stat_dim:
            raise ValueError(f"event_stat must be [B,N,{self.stat_dim}], got {event_stat.shape}")
        event_stat = event_stat.float()
        mean, std = self._source_stats(event_stat, source_id)
        z = (event_stat - mean.float()) / (std.float() + 1e-4)
        if self.clip is not None and self.clip > 0:
            z = torch.clamp(z, -self.clip, self.clip)
        return z * self.affine.float().view(1, 1, -1) + self.bias.float().view(1, 1, -1)


class Night3RBoundedEventFusion(nn.Module):
    """MASt3R-preserving bounded event residual adapter.

    R_raw is never layer-normalized on the output path: the adapter only adds a
    small bounded residual to preserve the pretrained MASt3R token distribution.
    The token gate is [B,N,1] so event reliability is spatial, not a free
    per-channel rewrite.  Optional channel modulation is explicitly bounded.
    Gate/stat computation runs in fp32 to avoid mixed-precision gate saturation.
    """

    def __init__(self, dim, event_dim=None, cfg=None):
        super().__init__()
        cfg = {} if cfg is None else dict(cfg)
        self.dim = int(dim)
        self.event_dim = int(event_dim if event_dim is not None else dim)
        self.stat_dim = int(cfg.get("stat_dim", 3))
        self.use_event_stat = bool(cfg.get("use_event_stat", True))
        self.output_mode = str(cfg.get("output_mode", "residual")).lower()
        if self.output_mode not in {"residual", "replace", "event_only"}:
            raise ValueError(f"Unsupported BRG output_mode: {self.output_mode}")
        stat_mask = cfg.get("event_stat_mask", None)
        if stat_mask is None:
            stat_mask = torch.ones(self.stat_dim, dtype=torch.float32)
        else:
            stat_mask = torch.as_tensor(stat_mask, dtype=torch.float32).reshape(-1)
            if stat_mask.numel() != self.stat_dim:
                raise ValueError(
                    f"event_stat_mask must have {self.stat_dim} values, got {stat_mask.numel()}"
                )
        self.register_buffer(
            "event_stat_mask",
            stat_mask.reshape(1, 1, self.stat_dim),
            persistent=False,
        )
        self.use_channel_modulation = bool(cfg.get("use_channel_modulation", False))
        self.use_source_bias = bool(cfg.get("use_source_bias", True))
        self.use_source_stat_norm = bool(cfg.get("use_source_stat_norm", self.use_source_bias))
        self.num_sources = int(cfg.get("num_sources", 3))
        self.gate_bias_init = float(cfg.get("gate_bias_init", -3.5))
        self.logit_bound = float(cfg.get("gate_logit_bound", 6.0))
        self.gate_temperature = float(cfg.get("gate_temperature", 1.0))
        self.gate_cap_start = float(cfg.get("gate_cap_start", 0.2))
        self.gate_cap_final = float(cfg.get("gate_cap_final", 1.0))
        self.gate_cap_warmup_steps = int(cfg.get("gate_cap_warmup_steps", 5000))
        self.occupancy_eps = float(cfg.get("occupancy_eps", 0.0))
        self.event_scale_init = float(cfg.get("event_scale_init", 0.05))
        self.event_scale_max = float(cfg.get("event_scale_max", 0.2))
        self.force_event_scale_zero = bool(cfg.get("force_event_scale_zero", False))
        self.channel_alpha_start = float(cfg.get("channel_alpha_start", 0.0))
        self.channel_alpha_final = float(
            cfg.get("channel_alpha_final", cfg.get("channel_alpha", cfg.get("channel_alpha_max", 0.0)))
        )
        self.channel_alpha_final = min(self.channel_alpha_final, float(cfg.get("channel_alpha_max", 0.1)))
        self.channel_alpha_warmup_steps = int(cfg.get("channel_alpha_warmup_steps", 0))
        self.use_gate_regularization = bool(cfg.get("use_gate_regularization", True))
        self.gate_mean_cap_start = float(cfg.get("gate_mean_cap_start", 0.05))
        self.gate_mean_cap_final = float(cfg.get("gate_mean_cap_final", 0.30))
        self.gate_mean_cap_warmup_steps = int(cfg.get("gate_mean_cap_warmup_steps", 10000))
        self.low_event_occupancy_thr = float(cfg.get("low_event_occupancy_thr", 0.01))
        self.low_event_density_thr = cfg.get("low_event_density_thr", None)
        self.low_event_density_thr = (
            None if self.low_event_density_thr is None else float(self.low_event_density_thr)
        )
        self.logit_reg_bound = float(cfg.get("logit_reg_bound", 4.0))
        self.gate_mean_reg_weight = float(cfg.get("gate_mean_reg_weight", 1e-3))
        self.gate_low_event_reg_weight = float(cfg.get("gate_low_event_reg_weight", 1e-2))
        self.gate_logit_reg_weight = float(cfg.get("gate_logit_reg_weight", 1e-4))
        self.channel_dev_reg_weight = float(cfg.get("channel_dev_reg_weight", 1e-4))
        visibility_cfg = cfg.get("visibility_gate", {})
        if not isinstance(visibility_cfg, dict):
            visibility_cfg = {}
        self.visibility_gate_enabled = bool(
            cfg.get("visibility_gate_enabled", visibility_cfg.get("enabled", False))
        )
        self.visibility_floor = float(
            cfg.get("visibility_floor", visibility_cfg.get("floor", 0.10))
        )
        self.visibility_gamma = float(
            cfg.get("visibility_gamma", visibility_cfg.get("gamma", 1.5))
        )
        self.visibility_detach = bool(
            cfg.get("visibility_detach", visibility_cfg.get("detach", True))
        )
        self.visibility_low_thr = float(
            cfg.get("visibility_low_thr", visibility_cfg.get("low_thr", 0.35))
        )
        self.visibility_high_thr = float(
            cfg.get("visibility_high_thr", visibility_cfg.get("high_thr", 0.70))
        )
        self.high_visibility_gate_reg_weight = float(
            cfg.get(
                "high_visibility_gate_reg_weight",
                visibility_cfg.get("high_visibility_gate_reg_weight", 0.0),
            )
        )
        structural_cfg = cfg.get("structural_gate", {})
        if not isinstance(structural_cfg, dict):
            structural_cfg = {}
        self.structural_gate_enabled = bool(
            cfg.get("structural_gate_enabled", structural_cfg.get("enabled", False))
        )
        self.structural_floor = float(
            cfg.get("structural_floor", structural_cfg.get("floor", 0.25))
        )
        self.structural_density_low = float(
            cfg.get("structural_density_low", structural_cfg.get("density_low", 0.03))
        )
        self.structural_density_tau = float(
            cfg.get("structural_density_tau", structural_cfg.get("density_tau", 0.05))
        )
        self.structural_occupancy_low = float(
            cfg.get("structural_occupancy_low", structural_cfg.get("occupancy_low", 0.02))
        )
        self.structural_occupancy_tau = float(
            cfg.get("structural_occupancy_tau", structural_cfg.get("occupancy_tau", 0.03))
        )
        self.structural_density_high = float(
            cfg.get("structural_density_high", structural_cfg.get("density_high", 2.5))
        )
        self.structural_density_high_tau = float(
            cfg.get("structural_density_high_tau", structural_cfg.get("density_high_tau", 1.0))
        )
        self.structural_detach = bool(
            cfg.get("structural_detach", structural_cfg.get("detach", True))
        )
        self.structural_low_thr = float(
            cfg.get("structural_low_thr", structural_cfg.get("low_thr", 0.35))
        )
        self.structural_high_thr = float(
            cfg.get("structural_high_thr", structural_cfg.get("high_thr", 0.70))
        )

        self.rgb_norm = nn.LayerNorm(self.dim)
        self.event_norm = nn.LayerNorm(self.event_dim)
        self.stat_norm = EventStatNorm(
            stat_dim=self.stat_dim,
            num_sources=self.num_sources if self.use_source_stat_norm else None,
            init_mean=cfg.get("event_stat_mean", cfg.get("event_stat_norm_mean", None)),
            init_std=cfg.get("event_stat_std", cfg.get("event_stat_norm_std", None)),
            source_init_mean=cfg.get("source_event_stat_mean", None),
            source_init_std=cfg.get("source_event_stat_std", None),
            clip=cfg.get("event_stat_clip", 5.0),
            learnable_affine=bool(cfg.get("event_stat_learnable_affine", True)),
        )
        hidden_event = int(cfg.get("event_proj_hidden_dim", 4 * self.dim))
        self.event_proj = nn.Sequential(
            nn.Linear(self.event_dim, hidden_event),
            nn.GELU(),
            nn.Linear(hidden_event, self.dim),
        )
        nn.init.xavier_uniform_(self.event_proj[-1].weight, gain=0.5)
        nn.init.zeros_(self.event_proj[-1].bias)

        gate_hidden = int(cfg.get("token_gate_hidden_dim", 64))
        self.token_gate = nn.Sequential(
            nn.Linear(self.stat_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.token_gate[-1].weight)
        nn.init.constant_(self.token_gate[-1].bias, self.gate_bias_init)

        if self.use_source_bias:
            self.source_bias = nn.Embedding(self.num_sources, 1)
            nn.init.zeros_(self.source_bias.weight)
            initial_bias = cfg.get("source_initial_gate_bias", None)
            if initial_bias is not None:
                bias_tensor = _stat_buffer(initial_bias, 1, default=0.0, rows=self.num_sources)
                with torch.no_grad():
                    self.source_bias.weight.copy_(bias_tensor)
        else:
            self.source_bias = None

        if self.use_channel_modulation:
            channel_hidden = int(cfg.get("channel_hidden_dim", max(self.dim // 4, 64)))
            self.channel_mlp = nn.Sequential(
                nn.Linear(self.dim + self.event_dim + self.stat_dim, channel_hidden),
                nn.GELU(),
                nn.Linear(channel_hidden, self.dim),
            )
            nn.init.zeros_(self.channel_mlp[-1].weight)
            nn.init.zeros_(self.channel_mlp[-1].bias)
        else:
            self.channel_mlp = None

        scale_ratio = self.event_scale_init / max(self.event_scale_max, 1e-6)
        self.event_scale_logit = nn.Parameter(
            torch.tensor(_inverse_sigmoid(scale_ratio), dtype=torch.float32)
        )

    @property
    def event_scale(self):
        if self.force_event_scale_zero:
            return self.event_scale_logit.new_zeros(())
        return self.event_scale_max * torch.sigmoid(self.event_scale_logit.float())

    def _linear_schedule(self, start, final, warmup_steps, step):
        if step is None:
            return float(final)
        if warmup_steps <= 0:
            return float(final)
        progress = min(max(float(step), 0.0), float(warmup_steps)) / float(warmup_steps)
        return float(start) + progress * (float(final) - float(start))

    def get_gate_cap(self, step=None):
        return self._linear_schedule(
            self.gate_cap_start,
            self.gate_cap_final,
            self.gate_cap_warmup_steps,
            step,
        )

    def get_gate_mean_cap(self, step=None):
        return self._linear_schedule(
            self.gate_mean_cap_start,
            self.gate_mean_cap_final,
            self.gate_mean_cap_warmup_steps,
            step,
        )

    def get_channel_alpha(self, step=None):
        return self._linear_schedule(
            self.channel_alpha_start,
            self.channel_alpha_final,
            self.channel_alpha_warmup_steps,
            step,
        )

    def _flatten_events(self, event_features):
        if event_features.ndim == 4:
            return event_features.flatten(2).transpose(1, 2)
        if event_features.ndim == 3:
            return event_features
        raise ValueError("event_features must be [B,C,H,W] or [B,N,C]")

    def _masked_mean(self, value, mask):
        mask = mask.to(dtype=value.dtype)
        denom = mask.sum()
        if denom.detach().item() <= 0:
            return value.new_zeros(())
        return (value * mask).sum() / denom.clamp_min(1.0)

    def forward(
        self,
        rgb_tokens,
        event_features,
        event_stat=None,
        rgb_visibility=None,
        source_id=None,
        step=None,
        return_diag=True,
        return_gate=False,
    ):
        if event_features is None:
            if return_diag or return_gate:
                empty_diag = {}
                empty_aux = {}
                gate = None
                if return_diag:
                    return rgb_tokens, empty_diag, empty_aux
                return rgb_tokens, gate
            return rgb_tokens
        if rgb_tokens.ndim != 3:
            raise ValueError("rgb_tokens must be [B,N,C]")
        event_tokens = self._flatten_events(event_features)
        if event_tokens.shape[:2] != rgb_tokens.shape[:2]:
            raise ValueError(
                f"BRG requires same token grid, got {rgb_tokens.shape} and {event_tokens.shape}"
            )
        if event_tokens.shape[-1] != self.event_dim:
            raise ValueError(f"Expected event dim {self.event_dim}, got {event_tokens.shape[-1]}")
        if self.use_event_stat:
            if event_stat is None:
                raise ValueError("event_stat is required when BRG use_event_stat=True")
            if event_stat.shape[:2] != rgb_tokens.shape[:2] or event_stat.shape[-1] != self.stat_dim:
                raise ValueError(
                    f"event_stat must be [B,N,{self.stat_dim}], got {event_stat.shape}"
                )

        dtype = rgb_tokens.dtype
        with torch.amp.autocast(device_type=rgb_tokens.device.type, enabled=False):
            rgb_raw = rgb_tokens.float()
            event_tokens = event_tokens.float()
            r_ln = self.rgb_norm(rgb_raw)
            e_ln = self.event_norm(event_tokens)
            if self.use_event_stat:
                event_stat_input = event_stat.float()
                event_stat_raw = event_stat_input * self.event_stat_mask
                stat = self.stat_norm(event_stat_raw, source_id=source_id)
                valid_event = (event_stat_input[..., 1:2] > self.occupancy_eps).float()
            else:
                event_stat_input = None
                event_stat_raw = rgb_raw.new_zeros(
                    rgb_raw.shape[0], rgb_raw.shape[1], self.stat_dim
                )
                stat = event_stat_raw
                valid_event = rgb_raw.new_ones(rgb_raw.shape[0], rgb_raw.shape[1], 1)

            delta_e = self.event_proj(e_ln)
            raw_logit = self.token_gate(stat)
            if self.source_bias is not None and source_id is not None:
                ids = torch.as_tensor(source_id, device=rgb_tokens.device).long().view(-1)
                if ids.numel() == rgb_tokens.shape[0]:
                    valid = (ids >= 0) & (ids < self.num_sources)
                    if valid.all():
                        raw_logit = raw_logit + self.source_bias(ids).view(-1, 1, 1)

            bounded_logit = self.logit_bound * torch.tanh(raw_logit / self.logit_bound)
            gate_cap = self.get_gate_cap(step)
            g_token = gate_cap * torch.sigmoid(bounded_logit / self.gate_temperature)
            g_token = g_token * valid_event
            visibility = None
            event_allow = None
            structural_reliability = None
            structural_allow = None
            if self.visibility_gate_enabled:
                if rgb_visibility is None:
                    raise ValueError("rgb_visibility is required when visibility_gate is enabled")
                visibility = rgb_visibility.float()
                if visibility.ndim == 2:
                    visibility = visibility.unsqueeze(-1)
                if visibility.shape[:2] != rgb_tokens.shape[:2] or visibility.shape[-1] != 1:
                    raise ValueError(
                        f"rgb_visibility must be [B,N,1], got {visibility.shape}"
                    )
                visibility = visibility.clamp(0.0, 1.0)
                if self.visibility_detach:
                    visibility = visibility.detach()
                floor = max(min(float(self.visibility_floor), 1.0), 0.0)
                low_visibility = (1.0 - visibility).clamp(0.0, 1.0)
                event_allow = floor + (1.0 - floor) * low_visibility.pow(
                    max(float(self.visibility_gamma), 1e-6)
                )
                g_token = g_token * event_allow
            if self.structural_gate_enabled:
                density = event_stat_raw[..., 0:1].clamp_min(0.0)
                occupancy = event_stat_raw[..., 1:2].clamp(0.0, 1.0)
                density_ok = torch.sigmoid(
                    (density - self.structural_density_low)
                    / max(self.structural_density_tau, 1e-6)
                )
                occupancy_ok = torch.sigmoid(
                    (occupancy - self.structural_occupancy_low)
                    / max(self.structural_occupancy_tau, 1e-6)
                )
                hotspot_ok = torch.exp(
                    -torch.relu(density - self.structural_density_high)
                    / max(self.structural_density_high_tau, 1e-6)
                )
                structural_reliability = (density_ok * occupancy_ok * hotspot_ok).clamp(0.0, 1.0)
                if self.structural_detach:
                    structural_reliability = structural_reliability.detach()
                floor = max(min(float(self.structural_floor), 1.0), 0.0)
                structural_allow = floor + (1.0 - floor) * structural_reliability
                g_token = g_token * structural_allow

            if self.channel_mlp is None:
                g_channel = torch.ones_like(delta_e)
                channel_dev = delta_e.new_zeros(())
            else:
                channel_in = torch.cat([r_ln, e_ln, stat], dim=-1)
                channel_raw = self.channel_mlp(channel_in)
                alpha = self.get_channel_alpha(step)
                g_channel = 1.0 + alpha * torch.tanh(channel_raw)
                channel_dev = (g_channel - 1.0).abs().mean()

            event_scale = self.event_scale
            residual = event_scale * g_token * g_channel * delta_e
            fused = rgb_raw + residual if self.output_mode == "residual" else residual

            low_event_mask = (
                event_stat_input[..., 1:2] < self.low_event_occupancy_thr
                if self.use_event_stat
                else torch.zeros_like(valid_event, dtype=torch.bool)
            )
            if self.use_event_stat and self.low_event_density_thr is not None:
                low_event_mask = low_event_mask | (
                    event_stat_input[..., 0:1] < self.low_event_density_thr
                )
            high_event_mask = valid_event > 0
            if visibility is not None:
                low_visibility_mask = visibility < self.visibility_low_thr
                high_visibility_mask = visibility > self.visibility_high_thr
                gate_high_visibility_reg = self._masked_mean(
                    g_token.pow(2), high_visibility_mask
                )
                visibility_mean = visibility.mean()
                event_allow_mean = event_allow.mean()
                low_visibility_gate_mean = self._masked_mean(g_token, low_visibility_mask)
                high_visibility_gate_mean = self._masked_mean(g_token, high_visibility_mask)
                gate_centered = g_token - g_token.mean()
                visibility_centered = visibility - visibility_mean
                gate_visibility_corr = (gate_centered * visibility_centered).mean() / (
                    gate_centered.pow(2).mean().sqrt()
                    * visibility_centered.pow(2).mean().sqrt()
                    + 1e-6
                )
            else:
                gate_high_visibility_reg = g_token.new_zeros(())
                visibility_mean = g_token.new_tensor(float("nan"))
                event_allow_mean = g_token.new_tensor(float("nan"))
                low_visibility_gate_mean = g_token.new_tensor(float("nan"))
                high_visibility_gate_mean = g_token.new_tensor(float("nan"))
                gate_visibility_corr = g_token.new_tensor(float("nan"))
            if structural_allow is not None:
                structural_mean = structural_reliability.mean()
                structural_allow_mean = structural_allow.mean()
                low_structural_mask = structural_reliability < self.structural_low_thr
                high_structural_mask = structural_reliability > self.structural_high_thr
                low_structural_gate_mean = self._masked_mean(g_token, low_structural_mask)
                high_structural_gate_mean = self._masked_mean(g_token, high_structural_mask)
            else:
                structural_mean = g_token.new_tensor(float("nan"))
                structural_allow_mean = g_token.new_tensor(float("nan"))
                low_structural_gate_mean = g_token.new_tensor(float("nan"))
                high_structural_gate_mean = g_token.new_tensor(float("nan"))
            gate_mean = g_token.mean()
            gate_mean_cap = self.get_gate_mean_cap(step)
            gate_mean_reg = torch.relu(gate_mean - gate_mean_cap).pow(2)
            gate_low_event_reg = self._masked_mean(g_token.pow(2), low_event_mask)
            gate_logit_reg = torch.relu(raw_logit.abs() - self.logit_reg_bound).pow(2).mean()
            channel_dev_reg = (g_channel - 1.0).pow(2).mean() if self.channel_mlp is not None else delta_e.new_zeros(())
            weight_sum = 0.0
            if self.use_gate_regularization:
                total_aux = (
                    self.gate_mean_reg_weight * gate_mean_reg
                    + self.gate_low_event_reg_weight * gate_low_event_reg
                    + self.gate_logit_reg_weight * gate_logit_reg
                    + self.channel_dev_reg_weight * channel_dev_reg
                    + self.high_visibility_gate_reg_weight * gate_high_visibility_reg
                )
                weight_sum = (
                    self.gate_mean_reg_weight
                    + self.gate_low_event_reg_weight
                    + self.gate_logit_reg_weight
                    + self.channel_dev_reg_weight
                    + self.high_visibility_gate_reg_weight
                )
            else:
                total_aux = gate_mean_reg.new_zeros(())

            rgb_norm = rgb_raw.norm(dim=-1).mean().clamp_min(1e-6)
            residual_ratio = residual.norm(dim=-1).mean() / rgb_norm
            diag = {
                "event_scale": event_scale.detach(),
                "gate_mean": gate_mean.detach(),
                "gate_std": g_token.std(unbiased=False).detach(),
                "gate_min": g_token.min().detach(),
                "gate_max": g_token.max().detach(),
                "gate_sat_low": (g_token < 0.01).float().mean().detach(),
                "gate_sat_high": (g_token > 0.95).float().mean().detach(),
                "channel_dev": channel_dev.detach(),
                "residual_ratio": residual_ratio.detach(),
                "gate_logit_abs": raw_logit.abs().mean().detach(),
                "low_event_gate_mean": self._masked_mean(g_token, low_event_mask).detach(),
                "high_event_gate_mean": self._masked_mean(g_token, high_event_mask).detach(),
                "visibility_mean": visibility_mean.detach(),
                "event_allow_mean": event_allow_mean.detach(),
                "structural_reliability_mean": structural_mean.detach(),
                "structural_allow_mean": structural_allow_mean.detach(),
                "low_visibility_gate_mean": low_visibility_gate_mean.detach(),
                "high_visibility_gate_mean": high_visibility_gate_mean.detach(),
                "low_structural_gate_mean": low_structural_gate_mean.detach(),
                "high_structural_gate_mean": high_structural_gate_mean.detach(),
                "gate_visibility_corr": gate_visibility_corr.detach(),
                "gate_cap": torch.tensor(gate_cap, device=rgb_tokens.device),
                "gate_mean_cap": torch.tensor(gate_mean_cap, device=rgb_tokens.device),
                "gate_tensor": g_token.detach(),
            }
            aux = {
                "gate_mean_reg": gate_mean_reg,
                "gate_low_event_reg": gate_low_event_reg,
                "gate_logit_reg": gate_logit_reg,
                "channel_dev_reg": channel_dev_reg,
                "gate_high_visibility_reg": gate_high_visibility_reg,
                "total": total_aux,
                "weight_sum": total_aux.new_tensor(weight_sum),
            }

        fused = fused.to(dtype)
        if return_diag:
            return fused, diag, aux
        if return_gate:
            return fused, g_token.to(dtype)
        return fused


class GatedCrossAttentionFusion(nn.Module):
    """Fuse event features into Dark3R RGB tokens with a small learnable gate."""

    def __init__(
        self,
        token_dim,
        event_dim=None,
        num_heads=4,
        dropout=0.0,
        init_gate=0.001,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.event_dim = int(event_dim if event_dim is not None else token_dim)
        self.event_proj = (
            nn.Identity()
            if self.event_dim == self.token_dim
            else nn.Linear(self.event_dim, self.token_dim)
        )
        self.rgb_norm = nn.LayerNorm(self.token_dim)
        self.event_norm = nn.LayerNorm(self.token_dim)
        self.attn = nn.MultiheadAttention(
            self.token_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.proj = nn.Linear(self.token_dim, self.token_dim)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def _flatten_events(self, event_features):
        if event_features.ndim == 4:
            return event_features.flatten(2).transpose(1, 2)
        if event_features.ndim == 3:
            return event_features
        raise ValueError("event_features must be [B,C,H,W] or [B,M,C]")

    def forward(self, rgb_tokens, event_features, event_stat=None, return_gate=False):
        if event_features is None:
            return (rgb_tokens, None) if return_gate else rgb_tokens
        event_tokens = self._flatten_events(event_features)
        event_tokens = self.event_proj(event_tokens)
        if rgb_tokens.shape[:2] != event_tokens.shape[:2]:
            raise ValueError(
                f"RGB/event token mismatch: {rgb_tokens.shape} vs {event_tokens.shape}"
            )
        attn_out, _ = self.attn(
            self.rgb_norm(rgb_tokens),
            self.event_norm(event_tokens),
            self.event_norm(event_tokens),
            need_weights=False,
        )
        fused = rgb_tokens + self.gate * self.proj(attn_out)
        if return_gate:
            return fused, self.gate.reshape(1, 1, 1).expand_as(fused)
        return fused


class AdditiveEventFusion(nn.Module):
    """Project same-grid event tokens and add them to the RGB token stream."""

    def __init__(self, dim, event_dim=None, event_scale_init=0.035):
        super().__init__()
        self.dim = int(dim)
        self.event_dim = int(event_dim if event_dim is not None else dim)
        self.event_norm = nn.LayerNorm(self.event_dim)
        self.projection = nn.Linear(self.event_dim, self.dim)
        self.event_scale = nn.Parameter(torch.tensor(float(event_scale_init)))
        nn.init.xavier_uniform_(self.projection.weight, gain=0.5)
        nn.init.zeros_(self.projection.bias)

    def _flatten_events(self, event_features):
        if event_features.ndim == 4:
            return event_features.flatten(2).transpose(1, 2)
        if event_features.ndim == 3:
            return event_features
        raise ValueError("event_features must be [B,C,H,W] or [B,N,C]")

    def forward(self, rgb_tokens, event_features, event_stat=None, return_gate=False):
        if event_features is None:
            return (rgb_tokens, None) if return_gate else rgb_tokens
        if rgb_tokens.ndim != 3:
            raise ValueError("rgb_tokens must be [B,N,C]")
        event_tokens = self._flatten_events(event_features)
        if event_tokens.shape[:2] != rgb_tokens.shape[:2]:
            raise ValueError(
                f"RGB/event token mismatch: {rgb_tokens.shape} vs {event_tokens.shape}"
            )
        residual = self.projection(self.event_norm(event_tokens))
        fused = rgb_tokens + self.event_scale * residual
        if return_gate:
            gate = self.event_scale.reshape(1, 1, 1).expand(
                rgb_tokens.shape[0], rgb_tokens.shape[1], 1
            )
            return fused, gate
        return fused


class ConcatMLPEventFusion(nn.Module):
    """Concatenate aligned RGB/event tokens and predict an RGB residual."""

    def __init__(self, dim, event_dim=None, hidden_dim=None, event_scale_init=0.035):
        super().__init__()
        self.dim = int(dim)
        self.event_dim = int(event_dim if event_dim is not None else dim)
        hidden_dim = int(hidden_dim if hidden_dim is not None else 2 * self.dim)
        self.rgb_norm = nn.LayerNorm(self.dim)
        self.event_norm = nn.LayerNorm(self.event_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.dim + self.event_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
        )
        self.event_scale = nn.Parameter(torch.tensor(float(event_scale_init)))
        nn.init.xavier_uniform_(self.fusion_mlp[-1].weight, gain=0.5)
        nn.init.zeros_(self.fusion_mlp[-1].bias)

    def _flatten_events(self, event_features):
        if event_features.ndim == 4:
            return event_features.flatten(2).transpose(1, 2)
        if event_features.ndim == 3:
            return event_features
        raise ValueError("event_features must be [B,C,H,W] or [B,N,C]")

    def forward(self, rgb_tokens, event_features, event_stat=None, return_gate=False):
        if event_features is None:
            return (rgb_tokens, None) if return_gate else rgb_tokens
        if rgb_tokens.ndim != 3:
            raise ValueError("rgb_tokens must be [B,N,C]")
        event_tokens = self._flatten_events(event_features)
        if event_tokens.shape[:2] != rgb_tokens.shape[:2]:
            raise ValueError(
                f"RGB/event token mismatch: {rgb_tokens.shape} vs {event_tokens.shape}"
            )
        fusion_input = torch.cat(
            [self.rgb_norm(rgb_tokens), self.event_norm(event_tokens)], dim=-1
        )
        residual = self.fusion_mlp(fusion_input)
        fused = rgb_tokens + self.event_scale * residual
        if return_gate:
            gate = self.event_scale.reshape(1, 1, 1).expand(
                rgb_tokens.shape[0], rgb_tokens.shape[1], 1
            )
            return fused, gate
        return fused


class TensorGatedFusion(nn.Module):
    """Fuse same-grid RGB and event tokens with a channel-wise tensor gate."""

    def __init__(
        self,
        dim,
        event_dim=None,
        stat_dim=3,
        hidden_dim=256,
        use_event_stat=True,
        event_scale_init=0.1,
        gate_bias_init=-2.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.event_dim = int(event_dim if event_dim is not None else dim)
        self.stat_dim = int(stat_dim)
        self.use_event_stat = bool(use_event_stat)
        self.event_proj = nn.Linear(self.event_dim, self.dim)
        gate_in_dim = self.dim * 2 + (self.stat_dim if self.use_event_stat else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.dim),
        )
        self.event_scale = nn.Parameter(torch.tensor(float(event_scale_init)))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))

    def _flatten_events(self, event_features):
        if event_features.ndim == 4:
            return event_features.flatten(2).transpose(1, 2)
        if event_features.ndim == 3:
            return event_features
        raise ValueError("event_features must be [B,C,H,W] or [B,N,C]")

    def forward(self, rgb_tokens, event_features, event_stat=None, return_gate=False):
        if event_features is None:
            return (rgb_tokens, None) if return_gate else rgb_tokens
        if rgb_tokens.ndim != 3:
            raise ValueError("rgb_tokens must be [B,N,C]")
        event_tokens = self._flatten_events(event_features)
        event_tokens = self.event_proj(event_tokens)
        if rgb_tokens.shape != event_tokens.shape:
            raise ValueError(
                f"TensorGatedFusion requires same token shape, got {rgb_tokens.shape} and {event_tokens.shape}"
            )
        if self.use_event_stat:
            if event_stat is None:
                raise ValueError("event_stat is required when TensorGatedFusion.use_event_stat=True")
            if event_stat.shape[:2] != rgb_tokens.shape[:2]:
                raise ValueError(
                    f"event_stat token mismatch: {event_stat.shape} vs {rgb_tokens.shape}"
                )
            if event_stat.shape[-1] != self.stat_dim:
                raise ValueError(f"Expected event_stat dim {self.stat_dim}, got {event_stat.shape[-1]}")
            gate_input = torch.cat([rgb_tokens, event_tokens, event_stat], dim=-1)
        else:
            gate_input = torch.cat([rgb_tokens, event_tokens], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused = rgb_tokens + self.event_scale * gate * event_tokens
        if return_gate:
            return fused, gate
        return fused


class FiLMEventFusion(nn.Module):
    """Simple FiLM event fusion kept for ablations; V0 defaults to cross-attention."""

    def __init__(self, token_dim, event_dim=None, init_gate=0.001):
        super().__init__()
        self.token_dim = int(token_dim)
        self.event_dim = int(event_dim if event_dim is not None else token_dim)
        self.event_proj = nn.Linear(self.event_dim, 2 * self.token_dim)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, rgb_tokens, event_features):
        if event_features is None:
            return rgb_tokens
        if event_features.ndim == 4:
            event_tokens = event_features.flatten(2).transpose(1, 2)
        elif event_features.ndim == 3:
            event_tokens = event_features
        else:
            raise ValueError("event_features must be [B,C,H,W] or [B,M,C]")
        pooled = event_tokens.mean(dim=1)
        gamma, beta = self.event_proj(pooled).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return rgb_tokens + self.gate * (rgb_tokens * gamma + beta)
