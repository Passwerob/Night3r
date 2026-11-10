import numpy as np
import torch


def _as_tensor(events):
    if events is None:
        return None
    if isinstance(events, np.ndarray):
        return torch.from_numpy(events)
    if torch.is_tensor(events):
        return events
    return torch.as_tensor(events)


def _normalize_nonzero(voxel):
    mask = voxel != 0
    if not mask.any():
        return voxel
    values = voxel[mask]
    std = values.std(unbiased=False)
    if torch.isfinite(std) and std > 1e-6:
        voxel = voxel.clone()
        voxel[mask] = (values - values.mean()) / std
    return voxel


def build_event_voxel(
    events,
    height,
    width,
    num_bins=4,
    polarity_separate=True,
    normalize=True,
):
    """Build a temporal event voxel grid.

    Args:
        events: Tensor/ndarray shaped [N, 4] with x,y,t,p or [N, 5] with
            batch_idx,x,y,t,p. Polarity p may be {0,1} or {-1,+1}.
        height: output image height.
        width: output image width.
        num_bins: temporal bins. Dark3R-event V0 defaults to 4.
        polarity_separate: if True, use positive and negative polarity channels.
        normalize: normalize non-zero voxel values to zero mean and unit std.

    Returns:
        None if events is empty. For [N, 4], returns [C,H,W], where C is 8 when
        polarity_separate=True and 4 otherwise. For [N, 5], returns [B,C,H,W].
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    events = _as_tensor(events)
    if events is None or events.numel() == 0:
        return None
    if events.ndim != 2 or events.shape[1] not in (4, 5):
        raise ValueError("events must have shape [N,4] or [N,5]")

    device = events.device
    dtype = torch.float32 if not events.is_floating_point() else events.dtype
    events = events.to(device=device, dtype=dtype)

    has_batch = events.shape[1] == 5
    if has_batch:
        batch_idx = events[:, 0].long()
        x, y, t, p = events[:, 1], events[:, 2], events[:, 3], events[:, 4]
        batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
    else:
        batch_idx = None
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        batch_size = 1

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not valid.any():
        return None

    if has_batch:
        batch_idx = batch_idx[valid]
    x = x[valid].long()
    y = y[valid].long()
    t = t[valid]
    p = p[valid]

    t_min = t.min()
    t_range = t.max() - t_min
    if torch.isfinite(t_range) and t_range > 1e-12:
        t_norm = (t - t_min) / t_range
    else:
        t_norm = torch.zeros_like(t)

    t_scaled = t_norm.clamp(0, 1) * (num_bins - 1)
    t_left = torch.floor(t_scaled).long()
    t_right = torch.clamp(t_left + 1, max=num_bins - 1)
    w_right = t_scaled - t_left.to(t_scaled.dtype)
    w_left = 1.0 - w_right

    channels = 2 * num_bins if polarity_separate else num_bins
    shape = (batch_size, channels, int(height), int(width))
    voxel = torch.zeros(shape, device=device, dtype=dtype)

    if polarity_separate:
        # Channel layout: [positive bins..., negative bins...].
        pol_offset = torch.where(p > 0, 0, num_bins)
        values = torch.ones_like(t_scaled)
    else:
        pol_offset = 0
        values = torch.where(p > 0, torch.ones_like(t_scaled), -torch.ones_like(t_scaled))

    if has_batch:
        b = batch_idx
    else:
        b = torch.zeros_like(x)

    c_left = pol_offset + t_left
    c_right = pol_offset + t_right
    voxel.index_put_((b, c_left, y, x), values * w_left, accumulate=True)
    voxel.index_put_((b, c_right, y, x), values * w_right, accumulate=True)

    if normalize:
        if has_batch:
            voxel = torch.stack([_normalize_nonzero(v) for v in voxel], dim=0)
        else:
            voxel = _normalize_nonzero(voxel[0]).unsqueeze(0)

    return voxel if has_batch else voxel[0]
