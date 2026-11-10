import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.event_voxel import build_event_voxel
from model.event_encoder import Dark3REventTokenEncoder
from model.event_fusion import (
    AdditiveEventFusion,
    ConcatMLPEventFusion,
    DensityFiLM,
    GatedCrossAttentionFusion,
    Night3RBoundedEventFusion,
    ResidualIdentityDensityFiLM,
    TensorGatedFusion,
    compute_event_stats,
    compute_visibility_confidence,
    normalize_event_voxel,
)
from model.relative_pose_head import RelativePoseHead


SOURCE_NAME_TO_ID = {
    "mvsec": 0,
    "dsec": 1,
    "dl3dv": 2,
}


def infer_dark3r_token_dim(dark3r_model, fallback=768):
    if hasattr(dark3r_model, "enc_norm") and hasattr(
        dark3r_model.enc_norm, "normalized_shape"
    ):
        return int(dark3r_model.enc_norm.normalized_shape[0])
    if hasattr(dark3r_model, "decoder_embed") and hasattr(
        dark3r_model.decoder_embed, "in_features"
    ):
        return int(dark3r_model.decoder_embed.in_features)
    for name in ("enc_embed_dim", "embed_dim"):
        if hasattr(dark3r_model, name):
            return int(getattr(dark3r_model, name))
    return int(fallback)


def infer_dark3r_decoder_dim(dark3r_model, fallback=768):
    if hasattr(dark3r_model, "dec_embed_dim"):
        return int(dark3r_model.dec_embed_dim)
    if hasattr(dark3r_model, "dec_norm") and hasattr(
        dark3r_model.dec_norm, "normalized_shape"
    ):
        return int(dark3r_model.dec_norm.normalized_shape[0])
    return int(fallback)


def infer_event_channels(num_bins=4, polarity_separate=True):
    polarity_factor = 2 if polarity_separate else 1
    return int(num_bins) * polarity_factor


def _clone_view(view):
    return dict(view)


def decode_local_descriptors(dark3r_model, head_num, decout, true_shape):
    """Run only MASt3R's local-feature MLP, bypassing the point-map DPT head."""
    module = getattr(dark3r_model, f"downstream_head{int(head_num)}", None)
    descriptor_head = getattr(module, "head_local_features", None)
    if not isinstance(descriptor_head, nn.Module):
        raise RuntimeError(
            f"downstream_head{head_num} does not expose head_local_features"
        )
    if not torch.is_tensor(true_shape):
        true_shape = torch.as_tensor(true_shape, device=decout[-1].device)
    shapes = true_shape.reshape(-1, 2)
    if not torch.equal(shapes, shapes[:1].expand_as(shapes)):
        raise ValueError("descriptor-only forward requires a uniform image shape per batch")
    height, width = (int(value) for value in shapes[0].detach().cpu().tolist())
    patch_size = int(module.patch_size)
    cat_output = torch.cat([decout[0], decout[-1]], dim=-1).float()
    local = descriptor_head(cat_output)
    batch_size = local.shape[0]
    local = local.transpose(-1, -2).reshape(
        batch_size,
        -1,
        height // patch_size,
        width // patch_size,
    )
    local = F.pixel_shuffle(local, patch_size).permute(0, 2, 3, 1)
    descriptor_dim = int(module.local_feat_dim)
    descriptor = F.normalize(local[..., :descriptor_dim], dim=-1, eps=1.0e-8)
    result = {"desc": descriptor}
    if bool(getattr(module, "two_confs", False)):
        result["desc_conf_raw"] = local[..., descriptor_dim]
    return result


def make_view_pair_from_batch(batch, prefer_dark=True):
    """Return Dark3R-compatible (view1, view2) without changing old datasets."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return _clone_view(batch[0]), _clone_view(batch[1])

    if "view1" in batch and "view2" in batch:
        return _clone_view(batch["view1"]), _clone_view(batch["view2"])

    image_key = "images_dark" if prefer_dark and "images_dark" in batch else "images"
    if image_key not in batch:
        raise KeyError("batch must contain view1/view2 or images/images_dark")

    images = batch[image_key]
    if images.ndim != 5 or images.shape[1] < 2:
        raise ValueError("images/images_dark must be shaped [B,2,C,H,W]")

    b, _, _, h, w = images.shape
    true_shape = torch.tensor([[h, w]], device=images.device, dtype=torch.int32).repeat(
        b, 1
    )
    view1 = {"img": images[:, 0], "true_shape": true_shape, "idx": 0, "instance": "0"}
    view2 = {"img": images[:, 1], "true_shape": true_shape, "idx": 1, "instance": "1"}
    return view1, view2


class EventDark3RStudent(nn.Module):
    """Dark3R student with an event encoder and a gated fusion adapter.

    The Dark3R backbone remains the only backbone. If no event input is provided,
    this wrapper falls back to RGB-only Dark3R behavior.
    """

    def __init__(self, dark3r_model, event_encoder=None, fusion_adapter=None, config=None):
        super().__init__()
        self.dark3r = dark3r_model
        self.config = config or {}
        token_dim = infer_dark3r_token_dim(dark3r_model)
        num_bins = int(self._cfg("event", "num_bins", 4))
        polarity_separate = bool(self._cfg("event", "polarity_separate", True))
        self.event_num_bins = num_bins
        self.event_polarity_separate = polarity_separate
        self.event_channels = infer_event_channels(num_bins, polarity_separate)
        num_heads = int(self._cfg("event_fusion", "num_heads", 4))
        init_gate = float(self._cfg("event_fusion", "init_gate", 0.001))
        dropout = float(self._cfg("event_fusion", "dropout", 0.0))
        fusion_cfg = self._section("fusion")
        event_fusion_cfg = self._section("event_fusion")
        self.fusion_type = str(
            fusion_cfg.get(
                "fusion_type",
                fusion_cfg.get("gate_type", event_fusion_cfg.get("gate_type", "scalar")),
            )
        ).lower()
        self.gate_type = self.fusion_type
        self.fusion_disabled = (
            not bool(event_fusion_cfg.get("enabled", True))
            or self.fusion_type in {"none", "disabled", "rgb_only", "rgb-only"}
        )
        self.event_norm_enabled = bool(self._cfg("event_norm", "enabled", False))
        self.event_norm_use_log_count = bool(self._cfg("event_norm", "use_log_count", True))
        self.event_norm_use_nonzero_norm = bool(
            self._cfg("event_norm", "use_nonzero_norm", True)
        )
        self.event_norm_eps = float(self._cfg("event_norm", "eps", 1e-6))
        self.event_stat_enabled = bool(self._cfg("event_stat", "enabled", False))
        self.event_stat_include_dt = bool(self._cfg("event_stat", "include_dt", False))
        self.event_stat_dim = int(
            self._cfg("event_stat", "stat_dim", 4 if self.event_stat_include_dt else 3)
        )
        self.event_stat_eps = float(self._cfg("event_stat", "eps", 1e-6))
        visibility_cfg = self._section("visibility_gate")
        brg_visibility_cfg = self._section("brg").get("visibility_gate", {})
        if not isinstance(brg_visibility_cfg, dict):
            brg_visibility_cfg = {}
        self.visibility_gate_enabled = bool(
            visibility_cfg.get("enabled", brg_visibility_cfg.get("enabled", False))
        )
        self.visibility_dark_thr = float(
            visibility_cfg.get("dark_thr", brg_visibility_cfg.get("dark_thr", 0.08))
        )
        self.visibility_sat_thr = float(
            visibility_cfg.get("sat_thr", brg_visibility_cfg.get("sat_thr", 0.95))
        )
        self.visibility_exposure_scale = float(
            visibility_cfg.get(
                "exposure_scale", brg_visibility_cfg.get("exposure_scale", 0.08)
            )
        )
        self.visibility_contrast_tau = float(
            visibility_cfg.get(
                "contrast_tau", brg_visibility_cfg.get("contrast_tau", 0.06)
            )
        )
        self.visibility_texture_tau = float(
            visibility_cfg.get("texture_tau", brg_visibility_cfg.get("texture_tau", 0.03))
        )
        self.visibility_artifact_tau = float(
            visibility_cfg.get(
                "artifact_tau", brg_visibility_cfg.get("artifact_tau", 2.0)
            )
        )
        self.visibility_smooth_kernel = int(
            visibility_cfg.get(
                "smooth_kernel", brg_visibility_cfg.get("smooth_kernel", 5)
            )
        )
        temporal_cfg = self._section("event_temporal_support")
        self.event_temporal_support_enabled = bool(temporal_cfg.get("enabled", False))
        self.event_temporal_support_mode = str(temporal_cfg.get("mode", "soft")).lower()
        self.event_temporal_center_bins = int(
            temporal_cfg.get("center_bins", min(2, self.event_num_bins))
        )
        self.event_temporal_min_outer_weight = float(
            temporal_cfg.get("min_outer_weight", 0.1)
        )
        self.event_temporal_max_outer_weight = float(
            temporal_cfg.get("max_outer_weight", 1.0)
        )
        self.event_temporal_visibility_gamma = float(
            temporal_cfg.get("visibility_gamma", 1.0)
        )
        self.event_temporal_hard_threshold = float(
            temporal_cfg.get("hard_visibility_threshold", 0.5)
        )
        self.event_temporal_center_weight = float(temporal_cfg.get("center_weight", 1.0))
        self.event_temporal_detach_visibility = bool(
            temporal_cfg.get("detach_visibility", True)
        )
        event_film_cfg = self._section("event_film")
        self.event_film_enabled = bool(
            fusion_cfg.get("use_density_film", event_film_cfg.get("enabled", False))
        )
        self.density_film_type = str(
            fusion_cfg.get("density_film_type", event_film_cfg.get("type", "legacy"))
        ).lower()
        if self.density_film_type == "density":
            self.density_film_type = "legacy"
        patch_size = self._cfg("event", "patch_size", None)
        copy_encoder = bool(self._cfg("event", "copy_dark3r_encoder", True))
        freeze_copied = bool(self._cfg("event", "freeze_copied_encoder", False))
        self.event_encoder = event_encoder or Dark3REventTokenEncoder(
            dark3r_model,
            in_channels=self.event_channels,
            token_dim=token_dim,
            patch_size=patch_size,
            copy_encoder=copy_encoder,
            freeze_copied_encoder=freeze_copied,
        )
        if self.event_film_enabled and self.density_film_type not in {"none", "disabled"}:
            if self.density_film_type == "legacy":
                self.event_film = DensityFiLM(
                    token_dim,
                    stat_dim=self.event_stat_dim,
                    hidden_dim=int(self._cfg("event_film", "hidden_dim", 128)),
                )
            elif self.density_film_type in {"residual_identity", "identity_residual"}:
                self.event_film = ResidualIdentityDensityFiLM(
                    token_dim,
                    stat_dim=self.event_stat_dim,
                    hidden_dim=int(self._cfg("event_film", "hidden_dim", 128)),
                    eta_init=float(self._cfg("event_film", "eta_init", 0.0)),
                    eta_max=float(self._cfg("event_film", "eta_max", 0.1)),
                )
            else:
                raise ValueError(f"Unsupported density_film_type: {self.density_film_type}")
        else:
            self.event_film = None
            self.event_film_enabled = False

        if self.fusion_disabled:
            self.fusion_adapter = nn.Identity()
        elif fusion_adapter is not None:
            self.fusion_adapter = fusion_adapter
        elif self.fusion_type in {"brg", "bounded_reliability", "bounded_reliability_gate"}:
            brg_cfg = dict(self._section("brg"))
            for key in (
                "event_scale_init",
                "event_scale_max",
                "gate_bias_init",
                "hidden_dim",
                "use_channel_modulation",
                "use_source_bias",
                "use_gate_regularization",
                "force_event_scale_zero",
            ):
                if key in fusion_cfg and key not in brg_cfg:
                    brg_cfg[key] = fusion_cfg[key]
            brg_cfg.setdefault("stat_dim", self.event_stat_dim)
            self.fusion_adapter = Night3RBoundedEventFusion(
                dim=token_dim,
                event_dim=token_dim,
                cfg=brg_cfg,
            )
        elif self.fusion_type in {"tensor", "legacy_tensor"}:
            self.fusion_adapter = TensorGatedFusion(
                dim=token_dim,
                event_dim=token_dim,
                stat_dim=self.event_stat_dim,
                hidden_dim=int(fusion_cfg.get("hidden_dim", 256)),
                use_event_stat=bool(fusion_cfg.get("use_event_stat", self.event_stat_enabled)),
                event_scale_init=float(fusion_cfg.get("event_scale_init", init_gate)),
                gate_bias_init=float(fusion_cfg.get("gate_bias_init", -2.0)),
            )
        elif self.fusion_type in {"concat", "concat_mlp", "concat-mlp"}:
            self.fusion_adapter = ConcatMLPEventFusion(
                dim=token_dim,
                event_dim=token_dim,
                hidden_dim=int(fusion_cfg.get("hidden_dim", 2 * token_dim)),
                event_scale_init=float(fusion_cfg.get("event_scale_init", 0.035)),
            )
        elif self.fusion_type in {"add", "additive", "additive_projection"}:
            self.fusion_adapter = AdditiveEventFusion(
                dim=token_dim,
                event_dim=token_dim,
                event_scale_init=float(fusion_cfg.get("event_scale_init", 0.035)),
            )
        elif self.fusion_type in {"simple", "scalar", "gated_cross_attention", "cross_attention"}:
            self.fusion_adapter = GatedCrossAttentionFusion(
                token_dim=token_dim,
                event_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout,
                init_gate=init_gate,
            )
        else:
            raise ValueError(f"Unsupported fusion_type/gate_type: {self.fusion_type}")

        self.fusion_requires_event_stat = (
            not self.fusion_disabled
            and (
                (
                    self.fusion_type
                    in {"brg", "bounded_reliability", "bounded_reliability_gate"}
                    and getattr(self.fusion_adapter, "use_event_stat", True)
                )
                or (
                    self.fusion_type in {"tensor", "legacy_tensor"}
                    and getattr(self.fusion_adapter, "use_event_stat", False)
                )
            )
        )
        if (self.event_film_enabled or self.fusion_requires_event_stat) and not self.event_stat_enabled:
            raise ValueError(
                "event_stat.enabled must be true when DensityFiLM or stat-driven fusion is enabled"
            )
        if self.visibility_gate_enabled and not isinstance(self.fusion_adapter, Night3RBoundedEventFusion):
            raise ValueError("visibility_gate currently requires fusion_type='brg'")

        pose_cfg = self._section("pose_head")
        self.pose_head_enabled = bool(pose_cfg.get("enabled", False))
        self.pose_head_only = bool(self._section("training").get("pose_head_only", False))
        self.descriptor_head_only = bool(
            self._section("training").get("descriptor_head_only", False)
        )
        self.geometry_head_only = bool(
            self._section("training").get("geometry_head_only", False)
        )
        exclusive_modes = sum(
            int(value)
            for value in (
                self.pose_head_only,
                self.descriptor_head_only,
                self.geometry_head_only,
            )
        )
        if exclusive_modes > 1:
            raise ValueError(
                "pose_head_only, descriptor_head_only, and geometry_head_only "
                "are mutually exclusive"
            )
        if self.pose_head_enabled:
            decoder_dim = infer_dark3r_decoder_dim(dark3r_model)
            self.pose_head = RelativePoseHead(
                input_dim=decoder_dim,
                hidden_dim=int(pose_cfg.get("hidden_dim", 512)),
                pool_hidden_dim=int(pose_cfg.get("pool_hidden_dim", 128)),
                dropout=float(pose_cfg.get("dropout", 0.1)),
                separate_branches=bool(pose_cfg.get("separate_branches", False)),
            )
        else:
            self.pose_head = None

    def train(self, mode=True):
        super().train(mode)
        if self.pose_head_only:
            self.dark3r.eval()
            self.event_encoder.eval()
            self.fusion_adapter.eval()
            if isinstance(self.event_film, nn.Module):
                self.event_film.eval()
            if isinstance(self.pose_head, nn.Module):
                self.pose_head.train(mode)
        elif self.descriptor_head_only:
            self.dark3r.eval()
            self.event_encoder.eval()
            self.fusion_adapter.eval()
            if isinstance(self.event_film, nn.Module):
                self.event_film.eval()
            for module_name in ("downstream_head1", "downstream_head2"):
                module = getattr(self.dark3r, module_name, None)
                descriptor_head = getattr(module, "head_local_features", None)
                if isinstance(descriptor_head, nn.Module):
                    descriptor_head.train(mode)
        elif self.geometry_head_only:
            self.dark3r.eval()
            self.event_encoder.eval()
            self.fusion_adapter.eval()
            if isinstance(self.event_film, nn.Module):
                self.event_film.eval()
            for module_name in ("downstream_head1", "downstream_head2"):
                module = getattr(self.dark3r, module_name, None)
                if isinstance(module, nn.Module):
                    module.train(mode)
        return self

    def _cfg(self, section, key, default):
        section_cfg = self._section(section)
        return section_cfg.get(key, default)

    def _section(self, section):
        return self.config.get(section, {}) if isinstance(self.config, dict) else {}

    def set_train_step(self, step):
        self.current_step = None if step is None else int(step)

    def _current_step(self):
        return getattr(self, "current_step", None)

    def _batch_values(self, value, batch_size):
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().cpu().view(-1).tolist()
            return value
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, (list, tuple)):
            values = list(value)
            if len(values) == 1 and batch_size > 1:
                return values * batch_size
            return values
        return [value] * batch_size

    def _source_id_from_batch(self, batch, device, batch_size):
        if not isinstance(batch, dict):
            return None
        for key in ("source_id", "dataset_source_id"):
            if key in batch:
                ids = torch.as_tensor(batch[key], device=device).long().view(-1)
                if ids.numel() == 1 and batch_size > 1:
                    ids = ids.expand(batch_size)
                return ids if ids.numel() == batch_size else None

        source_value = None
        for key in ("dataset_source", "source_name", "source"):
            if key in batch:
                source_value = batch[key]
                break
        values = self._batch_values(source_value, batch_size)
        if not values:
            return None
        ids = []
        for value in values:
            source = str(value).strip().lower()
            if source not in SOURCE_NAME_TO_ID:
                return None
            ids.append(SOURCE_NAME_TO_ID[source])
        if len(ids) != batch_size:
            return None
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _event_from_batch(self, batch, view, view_index, height, width):
        if "event_voxel" in view:
            return view["event_voxel"]
        if "events" in view:
            return build_event_voxel(
                view["events"],
                height,
                width,
                num_bins=int(self._cfg("event", "num_bins", 4)),
                polarity_separate=bool(self._cfg("event", "polarity_separate", True)),
                normalize=bool(self._cfg("event", "normalize", True)),
            )
        if not isinstance(batch, dict):
            return None

        key = f"event_voxel{view_index + 1}"
        if key in batch:
            return batch[key]
        if "event_voxel" in batch:
            event_voxel = batch["event_voxel"]
            if torch.is_tensor(event_voxel) and event_voxel.ndim == 5:
                return event_voxel[:, view_index]
            if view_index == 0:
                return event_voxel

        key = f"events{view_index + 1}"
        if key in batch:
            return build_event_voxel(
                batch[key],
                height,
                width,
                num_bins=int(self._cfg("event", "num_bins", 4)),
                polarity_separate=bool(self._cfg("event", "polarity_separate", True)),
                normalize=bool(self._cfg("event", "normalize", True)),
            )
        if "events" in batch and view_index == 0:
            return build_event_voxel(
                batch["events"],
                height,
                width,
                num_bins=int(self._cfg("event", "num_bins", 4)),
                polarity_separate=bool(self._cfg("event", "polarity_separate", True)),
                normalize=bool(self._cfg("event", "normalize", True)),
            )
        return None

    def _prepare_event(self, event_voxel, img):
        if event_voxel is None:
            return None
        if not torch.is_tensor(event_voxel):
            event_voxel = torch.as_tensor(event_voxel)
        event_voxel = event_voxel.to(device=img.device, dtype=img.dtype)
        if event_voxel.ndim == 3:
            event_voxel = event_voxel.unsqueeze(0)
        if event_voxel.ndim != 4:
            raise ValueError(
                "event_voxel must be [C,H,W] or [B,C,H,W], where C matches the "
                "configured event channels"
            )
        if event_voxel.shape[1] != self.event_channels:
            raise ValueError(
                f"event_voxel must have {self.event_channels} channels for "
                "the configured num_bins/polarity setting, got "
                f"{event_voxel.shape[1]}"
            )
        if event_voxel.shape[0] == 1 and img.shape[0] > 1:
            event_voxel = event_voxel.expand(img.shape[0], -1, -1, -1)
        return event_voxel

    def _event_token_hw(self, event_token, event_voxel):
        if event_token.ndim != 3:
            raise ValueError(f"event_token must be [B,N,C], got {event_token.shape}")
        token_hw = self._event_token_hw_from_voxel(event_voxel)
        if token_hw[0] * token_hw[1] == event_token.shape[1]:
            return token_hw
        side = int(math.sqrt(event_token.shape[1]))
        if side * side == event_token.shape[1]:
            return side, side
        patch_size = int(getattr(self.event_encoder, "patch_size", 16))
        raise ValueError(
            f"Cannot align event stats: event token count={event_token.shape[1]}, "
            f"event voxel shape={tuple(event_voxel.shape)}, patch_size={patch_size}"
        )

    def _event_token_hw_from_voxel(self, event_voxel):
        patch_size = int(getattr(self.event_encoder, "patch_size", 16))
        h, w = event_voxel.shape[-2:]
        token_hw = (h // patch_size, w // patch_size)
        if token_hw[0] <= 0 or token_hw[1] <= 0:
            raise ValueError(
                f"Invalid event token grid for event voxel shape={tuple(event_voxel.shape)} "
                f"and patch_size={patch_size}"
            )
        return token_hw

    def _normalize_event_if_enabled(self, event_voxel):
        if event_voxel is None or not self.event_norm_enabled:
            return event_voxel
        return normalize_event_voxel(
            event_voxel,
            use_log_count=self.event_norm_use_log_count,
            use_nonzero_norm=self.event_norm_use_nonzero_norm,
            eps=self.event_norm_eps,
        )

    def _compute_event_stat_if_enabled(self, event_voxel_raw, event_token, batch=None, view_index=0):
        if event_voxel_raw is None or not self.event_stat_enabled:
            return None
        token_hw = self._event_token_hw(event_token, event_voxel_raw)
        dt = None
        if self.event_stat_include_dt and isinstance(batch, dict):
            key = f"event_dt{view_index + 1}"
            dt = batch.get(key, batch.get("event_dt", None))
            if torch.is_tensor(dt) and dt.ndim >= 2 and dt.shape[1] > view_index:
                dt = dt[:, view_index]
        stat = compute_event_stats(
            event_voxel_raw,
            token_hw,
            dt=dt,
            eps=self.event_stat_eps,
            polarity_separate=bool(self._cfg("event", "polarity_separate", True)),
        )
        if stat.shape[1] != event_token.shape[1]:
            raise ValueError(
                f"event_stat/event_token token mismatch: {stat.shape} vs {event_token.shape}"
            )
        if stat.shape[-1] != self.event_stat_dim:
            raise ValueError(f"Expected event_stat dim {self.event_stat_dim}, got {stat.shape[-1]}")
        return stat.to(dtype=event_token.dtype)

    def _compute_rgb_visibility_for_event(self, image, event_voxel_raw):
        if event_voxel_raw is None:
            return None
        token_hw = self._event_token_hw_from_voxel(event_voxel_raw)
        visibility = compute_visibility_confidence(
            image,
            token_hw,
            dark_thr=self.visibility_dark_thr,
            sat_thr=self.visibility_sat_thr,
            exposure_scale=self.visibility_exposure_scale,
            contrast_tau=self.visibility_contrast_tau,
            texture_tau=self.visibility_texture_tau,
            artifact_tau=self.visibility_artifact_tau,
            smooth_kernel=self.visibility_smooth_kernel,
            eps=self.event_stat_eps,
        )
        return visibility.to(device=event_voxel_raw.device, dtype=event_voxel_raw.dtype)

    def _compute_visibility_if_enabled(self, image, event_token, event_voxel_raw):
        if not self.visibility_gate_enabled:
            return None
        if event_voxel_raw is None:
            return None
        visibility = self._compute_rgb_visibility_for_event(image, event_voxel_raw)
        if visibility.shape[1] != event_token.shape[1]:
            raise ValueError(
                f"visibility/event_token token mismatch: {visibility.shape} vs {event_token.shape}"
            )
        return visibility.to(device=event_token.device, dtype=event_token.dtype)

    def _temporal_center_mask(self, event_voxel):
        num_bins = int(self.event_num_bins)
        center_bins = max(1, min(int(self.event_temporal_center_bins), num_bins))
        start = max((num_bins - center_bins) // 2, 0)
        mask = torch.zeros(num_bins, device=event_voxel.device, dtype=event_voxel.dtype)
        mask[start : start + center_bins] = 1.0
        if self.event_polarity_separate:
            mask = torch.cat([mask, mask], dim=0)
        if mask.numel() != event_voxel.shape[1]:
            raise ValueError(
                f"Temporal mask has {mask.numel()} channels, expected {event_voxel.shape[1]}"
            )
        return mask.view(1, event_voxel.shape[1], 1, 1)

    def _apply_event_temporal_support(self, event_voxel_raw, image):
        """Adapt event temporal support without changing channel semantics.

        Current event voxels are [B,C,H,W] with C=num_bins or 2*num_bins.
        Center temporal bins keep full weight. Outer bins are softly or hard
        down-weighted when RGB visibility is high, reducing long-window event
        trails while preserving the event encoder input shape.
        """
        if event_voxel_raw is None or not self.event_temporal_support_enabled:
            return event_voxel_raw, None, {}
        if self.event_temporal_support_mode in {"none", "disabled", "off"}:
            return event_voxel_raw, None, {}

        visibility = self._compute_rgb_visibility_for_event(image, event_voxel_raw)
        if visibility is None:
            return event_voxel_raw, None, {}
        b, _, h, w = event_voxel_raw.shape
        token_hw = self._event_token_hw_from_voxel(event_voxel_raw)
        vis_map = visibility.transpose(1, 2).reshape(b, 1, token_hw[0], token_hw[1])
        vis_map = F.interpolate(
            vis_map.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).to(device=event_voxel_raw.device, dtype=event_voxel_raw.dtype)
        if self.event_temporal_detach_visibility:
            vis_map = vis_map.detach()

        min_outer = max(0.0, min(float(self.event_temporal_min_outer_weight), 1.0))
        max_outer = max(min_outer, min(float(self.event_temporal_max_outer_weight), 1.0))
        if self.event_temporal_support_mode == "hard":
            outer_weight = torch.where(
                vis_map >= float(self.event_temporal_hard_threshold),
                torch.full_like(vis_map, min_outer),
                torch.full_like(vis_map, max_outer),
            )
        elif self.event_temporal_support_mode == "soft":
            gamma = max(float(self.event_temporal_visibility_gamma), 1e-6)
            darkness = (1.0 - vis_map.clamp(0.0, 1.0)).pow(gamma)
            outer_weight = min_outer + (max_outer - min_outer) * darkness
        else:
            raise ValueError(
                f"Unsupported event_temporal_support.mode={self.event_temporal_support_mode}"
            )

        center_mask = self._temporal_center_mask(event_voxel_raw)
        weights = outer_weight.expand(-1, event_voxel_raw.shape[1], -1, -1)
        weights = weights * (1.0 - center_mask) + float(self.event_temporal_center_weight) * center_mask
        event_weighted = event_voxel_raw * weights
        diag = {
            "temporal_enabled": 1.0,
            "temporal_visibility_mean": float(visibility.detach().float().mean().cpu()),
            "temporal_outer_weight_mean": float(outer_weight.detach().float().mean().cpu()),
            "temporal_outer_weight_min": float(outer_weight.detach().float().min().cpu()),
            "temporal_outer_weight_max": float(outer_weight.detach().float().max().cpu()),
            "temporal_center_bins": float(max(1, min(int(self.event_temporal_center_bins), int(self.event_num_bins)))),
        }
        return event_weighted, visibility, diag

    def _apply_event_film_if_enabled(self, event_token, event_stat):
        if self.event_film is None:
            return event_token
        return self.event_film(event_token, event_stat)

    def _fuse(self, rgb_token, event_token, event_stat, rgb_visibility=None, source_id=None):
        if self.fusion_disabled:
            return rgb_token, None, {}, {}
        if isinstance(self.fusion_adapter, Night3RBoundedEventFusion):
            fused, diag, aux_losses = self.fusion_adapter(
                rgb_token,
                event_token,
                event_stat=event_stat,
                rgb_visibility=rgb_visibility,
                source_id=source_id,
                step=self._current_step(),
                return_diag=True,
            )
            gate = diag.get("gate_tensor", None) if isinstance(diag, dict) else None
            return fused, gate, diag, aux_losses
        fused, gate = self.fusion_adapter(
            rgb_token,
            event_token,
            event_stat=event_stat,
            return_gate=True,
        )
        return fused, gate, {}, {}

    def _stats_summary(self, tensors):
        valid = [tensor for tensor in tensors if torch.is_tensor(tensor)]
        if not valid:
            return {}
        value = torch.cat([tensor.detach().float().reshape(-1) for tensor in valid])
        if value.numel() == 0:
            return {}
        return {
            "mean": float(value.mean().cpu()),
            "std": float(value.std(unbiased=False).cpu()),
            "min": float(value.min().cpu()),
            "max": float(value.max().cpu()),
        }

    def _scalar_dict_summary(self, dicts):
        out = {}
        keys = sorted({key for item in dicts if isinstance(item, dict) for key in item})
        for key in keys:
            if key == "gate_tensor":
                continue
            values = []
            for item in dicts:
                value = item.get(key, None) if isinstance(item, dict) else None
                if torch.is_tensor(value) and value.numel() == 1:
                    values.append(value.detach().float().reshape(()))
                elif isinstance(value, (int, float)):
                    values.append(torch.tensor(float(value)))
            if values:
                out[key] = float(torch.stack([v.cpu() for v in values]).mean())
        return out

    def _merge_aux_losses(self, aux_losses):
        keys = sorted({key for item in aux_losses if isinstance(item, dict) for key in item})
        out = {}
        for key in keys:
            values = [
                item[key]
                for item in aux_losses
                if isinstance(item, dict) and torch.is_tensor(item.get(key, None))
            ]
            if values:
                out[key] = torch.stack([value.float().reshape(()) for value in values]).mean()
        return out

    def _event_debug_payload(
        self,
        event_stats,
        gates,
        token_norms,
        fusion_diags=None,
        aux_losses=None,
        visibilities=None,
        temporal_diags=None,
    ):
        gate = getattr(self.fusion_adapter, "gate", None)
        scalar_gate_value = None
        if gate is not None and torch.is_tensor(gate):
            if gate.ndim == 0:
                scalar_gate_value = float(gate.detach().cpu())
        gate_summary = self._stats_summary(gates)
        stat_summary = {}
        valid_stats = [stat for stat in event_stats if torch.is_tensor(stat)]
        if valid_stats:
            stat_cat = torch.cat([stat.detach().float() for stat in valid_stats], dim=0)
            stat_summary = {
                "event_density_mean": float(stat_cat[..., 0].mean().cpu()),
                "event_density_std": float(stat_cat[..., 0].std(unbiased=False).cpu()),
                "event_occupancy_mean": float(stat_cat[..., 1].mean().cpu()),
                "event_polarity_balance_mean": float(stat_cat[..., 2].mean().cpu()),
            }
        visibility_summary = {}
        valid_visibility = [value for value in ([] if visibilities is None else visibilities) if torch.is_tensor(value)]
        if valid_visibility:
            visibility_cat = torch.cat([value.detach().float() for value in valid_visibility], dim=0)
            visibility_summary = {
                "visibility_mean": float(visibility_cat.mean().cpu()),
                "visibility_std": float(visibility_cat.std(unbiased=False).cpu()),
                "visibility_min": float(visibility_cat.min().cpu()),
                "visibility_max": float(visibility_cat.max().cpu()),
            }
        fusion_diags = [] if fusion_diags is None else fusion_diags
        diag_summary = self._scalar_dict_summary(fusion_diags)
        temporal_summary = self._scalar_dict_summary(
            [] if temporal_diags is None else temporal_diags
        )
        event_scale = getattr(self.fusion_adapter, "event_scale", None)
        event_scale_value = None if event_scale is None else float(event_scale.detach().cpu())
        if "event_scale" in diag_summary:
            event_scale_value = diag_summary["event_scale"]
        aux_summary = self._merge_aux_losses([] if aux_losses is None else aux_losses)
        payload = {
            "event_gate": scalar_gate_value if scalar_gate_value is not None else gate_summary.get("mean"),
            "event_gate_summary": gate_summary,
            "event_scale": event_scale_value,
            "event_stat_summary": stat_summary,
            "event_fusion_type": self.fusion_type,
            "event_norm_enabled": self.event_norm_enabled,
            "event_film_enabled": self.event_film_enabled,
            "density_film_type": self.density_film_type,
            "visibility_gate_enabled": self.visibility_gate_enabled,
        }
        payload.update(visibility_summary)
        for key, value in diag_summary.items():
            if key == "event_scale":
                continue
            payload[f"event_{key}"] = value
        for key, value in temporal_summary.items():
            payload[f"event_{key}"] = value
        if aux_summary:
            payload["event_aux_losses"] = aux_summary
        payload.update(token_norms)
        return payload

    def _gate_mean_map(self, gate, event_token, event_voxel):
        if not torch.is_tensor(gate):
            return None
        token_hw = self._event_token_hw(event_token, event_voxel)
        if token_hw[0] * token_hw[1] != gate.shape[1]:
            return None
        return gate.detach().float().mean(dim=-1).view(gate.shape[0], token_hw[0], token_hw[1])

    def _add_debug(self, pred, enabled, event_shapes, event_debug=None):
        pred["event_enabled"] = enabled
        pred["event_features_shape"] = event_shapes
        if event_debug:
            pred.update(event_debug)
        return pred

    def forward(self, batch):
        view1, view2 = make_view_pair_from_batch(batch, prefer_dark=True)
        img1, img2 = view1["img"], view2["img"]
        b = img1.shape[0]
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:], device=img1.device)[None].repeat(b, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:], device=img2.device)[None].repeat(b, 1)
        )
        if not torch.is_tensor(shape1):
            shape1 = torch.as_tensor(shape1, device=img1.device)
        if not torch.is_tensor(shape2):
            shape2 = torch.as_tensor(shape2, device=img2.device)
        shape1 = shape1.to(img1.device)
        shape2 = shape2.to(img2.device)

        event1 = self._event_from_batch(batch, view1, 0, img1.shape[-2], img1.shape[-1])
        event2 = self._event_from_batch(batch, view2, 1, img2.shape[-2], img2.shape[-1])
        event1_raw = self._prepare_event(event1, img1)
        event2_raw = self._prepare_event(event2, img2)
        event_temporal_diags = []
        event_temporal_visibility1 = None
        event_temporal_visibility2 = None
        event1_raw, event_temporal_visibility1, temporal_diag1 = self._apply_event_temporal_support(
            event1_raw, img1
        )
        event2_raw, event_temporal_visibility2, temporal_diag2 = self._apply_event_temporal_support(
            event2_raw, img2
        )
        event_temporal_diags.extend([temporal_diag1, temporal_diag2])
        event1 = self._normalize_event_if_enabled(event1_raw)
        event2 = self._normalize_event_if_enabled(event2_raw)

        feat1, feat2, pos1, pos2 = self.dark3r._encode_image_pairs(
            img1, img2, shape1, shape2
        )

        event_shapes = []
        event_stats = []
        event_gates = []
        event_diags = []
        event_aux_losses = []
        event_visibilities = []
        token_norms = {}
        event_debug1 = {}
        event_debug2 = {}
        source_id = self._source_id_from_batch(batch, img1.device, b)
        if event1 is not None:
            event_feat1 = self.event_encoder(event1)
            event_shapes.append(tuple(event_feat1.shape))
            event_stat1 = self._compute_event_stat_if_enabled(event1_raw, event_feat1, batch, 0)
            event_stats.append(event_stat1)
            visibility1 = (
                event_temporal_visibility1.to(dtype=event_feat1.dtype)
                if self.visibility_gate_enabled and torch.is_tensor(event_temporal_visibility1)
                else self._compute_visibility_if_enabled(img1, event_feat1, event1_raw)
            )
            event_visibilities.append(visibility1)
            token_norms["event_token_norm_before_film"] = float(
                event_feat1.detach().float().norm(dim=-1).mean().cpu()
            )
            event_feat1 = self._apply_event_film_if_enabled(event_feat1, event_stat1)
            token_norms["event_token_norm_after_film"] = float(
                event_feat1.detach().float().norm(dim=-1).mean().cpu()
            )
            token_norms["rgb_token_norm"] = float(feat1.detach().float().norm(dim=-1).mean().cpu())
            feat1, gate1, diag1, aux1 = self._fuse(
                feat1,
                event_feat1,
                event_stat1,
                rgb_visibility=visibility1,
                source_id=source_id,
            )
            event_gates.append(gate1)
            event_diags.append(diag1)
            event_aux_losses.append(aux1)
            event_debug1 = {
                "event_gate_tensor": gate1.detach() if torch.is_tensor(gate1) else None,
                "event_gate_map": self._gate_mean_map(gate1, event_feat1, event1_raw),
                "event_stat": event_stat1.detach() if torch.is_tensor(event_stat1) else None,
                "rgb_visibility": visibility1.detach() if torch.is_tensor(visibility1) else None,
            }
        if event2 is not None:
            event_feat2 = self.event_encoder(event2)
            event_shapes.append(tuple(event_feat2.shape))
            event_stat2 = self._compute_event_stat_if_enabled(event2_raw, event_feat2, batch, 1)
            event_stats.append(event_stat2)
            visibility2 = (
                event_temporal_visibility2.to(dtype=event_feat2.dtype)
                if self.visibility_gate_enabled and torch.is_tensor(event_temporal_visibility2)
                else self._compute_visibility_if_enabled(img2, event_feat2, event2_raw)
            )
            event_visibilities.append(visibility2)
            if "event_token_norm_before_film" not in token_norms:
                token_norms["event_token_norm_before_film"] = float(
                    event_feat2.detach().float().norm(dim=-1).mean().cpu()
                )
            event_feat2 = self._apply_event_film_if_enabled(event_feat2, event_stat2)
            if "event_token_norm_after_film" not in token_norms:
                token_norms["event_token_norm_after_film"] = float(
                    event_feat2.detach().float().norm(dim=-1).mean().cpu()
                )
            if "rgb_token_norm" not in token_norms:
                token_norms["rgb_token_norm"] = float(
                    feat2.detach().float().norm(dim=-1).mean().cpu()
                )
            feat2, gate2, diag2, aux2 = self._fuse(
                feat2,
                event_feat2,
                event_stat2,
                rgb_visibility=visibility2,
                source_id=source_id,
            )
            event_gates.append(gate2)
            event_diags.append(diag2)
            event_aux_losses.append(aux2)
            event_debug2 = {
                "event_gate_tensor": gate2.detach() if torch.is_tensor(gate2) else None,
                "event_gate_map": self._gate_mean_map(gate2, event_feat2, event2_raw),
                "event_stat": event_stat2.detach() if torch.is_tensor(event_stat2) else None,
                "rgb_visibility": visibility2.detach() if torch.is_tensor(visibility2) else None,
            }

        dec1, dec2 = self.dark3r._decoder(feat1, pos1, feat2, pos2)
        pose_only_inference = bool(
            isinstance(batch, dict) and batch.get("pose_only_inference", False)
        )
        skip_geometry_heads = self.pose_head_enabled and (
            (self.pose_head_only and self.training) or pose_only_inference
        )
        descriptor_only_training = self.descriptor_head_only and self.training
        with torch.amp.autocast(device_type=img1.device.type, enabled=False):
            if descriptor_only_training:
                res1 = decode_local_descriptors(self.dark3r, 1, dec1, shape1)
                res2 = decode_local_descriptors(self.dark3r, 2, dec2, shape2)
            elif skip_geometry_heads:
                res1, res2 = {}, {}
            else:
                res1 = self.dark3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = self.dark3r._downstream_head(2, [tok.float() for tok in dec2], shape2)
            if self.pose_head_enabled:
                res1["relative_pose"] = self.pose_head(dec1[-1].float(), dec2[-1].float())

        if "pts3d" in res2:
            res2["pts3d_in_other_view"] = res2.pop("pts3d")
        enabled = event1 is not None or event2 is not None
        event_debug = self._event_debug_payload(
            event_stats,
            event_gates,
            token_norms,
            fusion_diags=event_diags,
            aux_losses=event_aux_losses,
            visibilities=event_visibilities,
            temporal_diags=event_temporal_diags,
        )
        self._add_debug(res1, enabled, event_shapes, {**event_debug, **event_debug1})
        self._add_debug(res2, enabled, event_shapes, {**event_debug, **event_debug2})
        return res1, res2
