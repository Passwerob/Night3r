import torch

from utils.depth_cutoff_metrics import DepthCutoffAccumulator


def test_cutoff_metrics_use_gt_distance_mask():
    pred = torch.tensor([[[[2.0, 4.0, 8.0, 16.0]]]])
    gt = torch.tensor([[[[1.0, 5.0, 15.0, 40.0]]]])
    valid = torch.ones_like(gt, dtype=torch.bool)
    accumulator = DepthCutoffAccumulator()
    accumulator.add_depth(pred, gt, valid)
    metrics = accumulator.finalize()

    assert metrics["depth_raw_valid_px_at_10m"] == 2
    assert metrics["depth_raw_mae_at_10m"] == 1.0
    assert metrics["depth_raw_valid_px_at_20m"] == 3
    assert metrics["depth_raw_mae_at_20m"] == 3.0


def test_cutoff_metrics_report_raw_and_median_aligned_mae():
    pred = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    gt = pred * 2.0
    valid = torch.ones_like(gt, dtype=torch.bool)
    accumulator = DepthCutoffAccumulator()
    accumulator.add_depth(pred, gt, valid)
    metrics = accumulator.finalize()

    assert metrics["depth_raw_valid_px_at_10m"] == 4
    assert metrics["depth_raw_mae_at_10m"] == 2.5
    assert metrics["depth_median_aligned_mae_at_10m"] == 0.0
