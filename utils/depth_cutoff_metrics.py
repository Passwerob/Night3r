"""Depth errors evaluated inside fixed metric-distance cutoffs."""

import torch
import torch.nn.functional as F


class DepthCutoffAccumulator:
    """Accumulate raw and per-sample median-aligned MAE at GT cutoffs."""

    def __init__(self, cutoffs=(10.0, 20.0, 30.0)):
        self.cutoffs = tuple(float(value) for value in cutoffs)
        self.stats = {
            variant: {
                cutoff: {"abs_sum": 0.0, "valid_px": 0}
                for cutoff in self.cutoffs
            }
            for variant in ("raw", "median_aligned")
        }

    def add_depth(self, pred_depth, gt_depth, gt_valid):
        pred_depth = pred_depth.float()
        gt_depth = gt_depth.float()
        gt_valid = gt_valid.bool()
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(
                pred_depth,
                size=gt_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        for batch_idx in range(pred_depth.shape[0]):
            pred_map = pred_depth[batch_idx]
            gt_map = gt_depth[batch_idx]
            valid = (
                gt_valid[batch_idx]
                & torch.isfinite(gt_map)
                & (gt_map > 0)
                & torch.isfinite(pred_map)
                & (pred_map > 0)
            )
            if not valid.any():
                continue

            scale = gt_map[valid].median() / pred_map[valid].median().clamp_min(1e-6)
            variants = {
                "raw": pred_map,
                "median_aligned": pred_map * scale,
            }
            for variant, prediction in variants.items():
                for cutoff in self.cutoffs:
                    cutoff_valid = valid & (gt_map <= cutoff)
                    count = int(cutoff_valid.sum().item())
                    if count <= 0:
                        continue
                    abs_error = (prediction[cutoff_valid] - gt_map[cutoff_valid]).abs()
                    stats = self.stats[variant][cutoff]
                    stats["abs_sum"] += float(abs_error.sum().detach().cpu())
                    stats["valid_px"] += count

    def finalize(self):
        output = {}
        for variant, cutoff_stats in self.stats.items():
            for cutoff, stats in cutoff_stats.items():
                label = f"{int(cutoff)}m" if cutoff.is_integer() else f"{cutoff:g}m"
                output[f"depth_{variant}_valid_px_at_{label}"] = stats["valid_px"]
                output[f"depth_{variant}_mae_at_{label}"] = (
                    stats["abs_sum"] / stats["valid_px"]
                    if stats["valid_px"] > 0
                    else None
                )
        return output
