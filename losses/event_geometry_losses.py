import numpy as np
import torch
import torch.nn.functional as F

from losses.depth_loss import SILogLoss


def _anchor_tensor(*items):
    for item in items:
        if torch.is_tensor(item):
            return item
        if isinstance(item, dict):
            anchor = _anchor_tensor(*item.values())
            if anchor is not None:
                return anchor
        if isinstance(item, (list, tuple)):
            anchor = _anchor_tensor(*item)
            if anchor is not None:
                return anchor
    return None


def _zero(anchor):
    if anchor is None:
        return torch.tensor(0.0)
    return anchor.new_zeros(())


def _as_bchw(x):
    if x is None:
        return None
    if x.ndim == 2:
        return x[None, None]
    if x.ndim == 3:
        return x[:, None]
    if x.ndim == 4:
        if x.shape[1] == 1:
            return x
        return x[:, :1]
    return None


def _depth_from_pred(pred):
    if pred is None:
        return None
    if "depth" in pred:
        return _as_bchw(pred["depth"])
    key = "pts3d" if "pts3d" in pred else "pts3d_in_other_view"
    if key not in pred:
        return None
    pts = pred[key]
    if pts.ndim != 4 or pts.shape[-1] < 3:
        return None
    return pts[..., 2].unsqueeze(1)


def _event_activity(event_voxel):
    if event_voxel is None:
        return None
    if not torch.is_tensor(event_voxel):
        event_voxel = torch.as_tensor(event_voxel)
    if event_voxel.ndim == 3:
        event_voxel = event_voxel.unsqueeze(0)
    if event_voxel.ndim == 5:
        event_voxel = event_voxel[:, 0]
    if event_voxel.ndim != 4:
        return None
    return event_voxel.abs().sum(dim=1, keepdim=True)


def _gradient_magnitude(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx.abs(), (0, 1, 0, 0))
    dy = F.pad(dy.abs(), (0, 0, 0, 1))
    return dx + dy


def _znorm(x, mask=None, eps=1e-6):
    if mask is not None:
        mask = mask & torch.isfinite(x)
        values = x[mask]
        if values.numel() == 0:
            return torch.zeros_like(x)
        mean = values.mean()
        std = values.std(unbiased=False)
        if not torch.isfinite(mean) or not torch.isfinite(std):
            return torch.zeros_like(x)
        std = std.clamp_min(eps)
        out = torch.zeros_like(x)
        out[mask] = (x[mask] - mean) / std
        return out
    else:
        mask = torch.isfinite(x)
        values = x[mask]
        if values.numel() == 0:
            return torch.zeros_like(x)
        mean = values.mean()
        std = values.std(unbiased=False)
        if not torch.isfinite(mean) or not torch.isfinite(std):
            return torch.zeros_like(x)
        std = std.clamp_min(eps)
        out = torch.zeros_like(x)
        out[mask] = (x[mask] - mean) / std
        return out


def event_depth_edge_loss(
    pred,
    event_voxel,
    weight=0.02,
    loss_dict=None,
    prefix="event_depth_edge",
):
    """Weakly encourage depth/geometry edges near event activity."""
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(pred, event_voxel)
    depth = _depth_from_pred(pred)
    activity = _event_activity(event_voxel)
    if depth is None:
        loss_dict[f"skipped {prefix}"] = "missing depth/pts3d output"
        return _zero(anchor), loss_dict
    if activity is None:
        loss_dict[f"skipped {prefix}"] = "missing event_voxel"
        return _zero(anchor), loss_dict

    activity = activity.to(device=depth.device, dtype=depth.dtype)
    if activity.shape[-2:] != depth.shape[-2:]:
        activity = F.interpolate(activity, size=depth.shape[-2:], mode="bilinear", align_corners=False)
    depth_grad = _gradient_magnitude(depth)
    valid = torch.isfinite(activity) & torch.isfinite(depth_grad)
    if not valid.any():
        loss_dict[f"skipped {prefix}"] = "no finite overlap"
        return _zero(anchor), loss_dict

    a = _znorm(activity, valid)
    g = _znorm(depth_grad, valid)
    valid = valid & torch.isfinite(a) & torch.isfinite(g)
    if not valid.any():
        loss_dict[f"skipped {prefix}"] = "no finite normalized overlap"
        return _zero(anchor), loss_dict
    corr = (a[valid] * g[valid]).mean()
    if not torch.isfinite(corr):
        loss_dict[f"skipped {prefix}"] = "non-finite edge correlation"
        return _zero(anchor), loss_dict
    loss = (1.0 - corr.clamp(-1, 1)) * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    return loss, loss_dict


def _event_sum(event_voxel):
    if event_voxel is None:
        return None
    if event_voxel.ndim == 5:
        event_voxel = event_voxel[:, 0]
    if event_voxel.ndim == 3:
        event_voxel = event_voxel.unsqueeze(0)
    if event_voxel.ndim != 4:
        return None
    if event_voxel.shape[1] == 8:
        return event_voxel[:, :4].sum(dim=1, keepdim=True) - event_voxel[:, 4:].sum(dim=1, keepdim=True)
    return event_voxel.sum(dim=1, keepdim=True)


def _grayscale_log(images):
    if images.ndim != 5 or images.shape[1] < 2:
        return None, None
    img_i = images[:, 0].mean(dim=1, keepdim=True).clamp_min(1e-4)
    img_j = images[:, 1].mean(dim=1, keepdim=True).clamp_min(1e-4)
    return torch.log(img_i), torch.log(img_j)


def _warp_second_to_first(log_img_j, depth_i, intrinsics, relative_pose):
    b, _, h, w = depth_i.shape
    device = depth_i.device
    dtype = depth_i.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).view(1, 3, -1).repeat(b, 1, 1)
    k_inv = torch.inverse(intrinsics.to(device=device, dtype=dtype))
    xyz_i = k_inv @ pix * depth_i.view(b, 1, -1)
    xyz_i_h = torch.cat([xyz_i, torch.ones(b, 1, xyz_i.shape[-1], device=device, dtype=dtype)], dim=1)
    xyz_j = relative_pose.to(device=device, dtype=dtype) @ xyz_i_h
    z = xyz_j[:, 2:3].clamp_min(1e-6)
    proj = intrinsics.to(device=device, dtype=dtype) @ xyz_j[:, :3]
    uv = proj[:, :2] / z
    x_norm = 2.0 * uv[:, 0] / max(w - 1, 1) - 1.0
    y_norm = 2.0 * uv[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(b, h, w, 2)
    warped = F.grid_sample(log_img_j, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (z.view(b, 1, h, w) > 1e-6) & (grid[..., 0].abs().unsqueeze(1) <= 1) & (grid[..., 1].abs().unsqueeze(1) <= 1)
    return warped, valid


def event_photo_consistency_loss(
    batch,
    pred,
    weight=0.05,
    loss_dict=None,
    prefix="event_photo",
):
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(batch, pred)
    if not isinstance(batch, dict):
        loss_dict[f"skipped {prefix}"] = "batch is not dict"
        return _zero(anchor), loss_dict

    images = batch.get(
        "images_photo_dark",
        batch.get("images_photo", batch.get("images_dark", batch.get("images", None))),
    )
    event_voxel = batch.get("event_voxel", None)
    intrinsics = batch.get("intrinsics", batch.get("camera_intrinsics", None))
    relative_pose = batch.get("relative_pose", batch.get("pose", None))
    if images is None:
        loss_dict[f"skipped {prefix}"] = "missing adjacent images"
        return _zero(anchor), loss_dict
    if event_voxel is None:
        loss_dict[f"skipped {prefix}"] = "missing event_voxel"
        return _zero(anchor), loss_dict
    if intrinsics is None:
        loss_dict[f"skipped {prefix}"] = "missing intrinsics"
        return _zero(anchor), loss_dict
    if relative_pose is None:
        loss_dict[f"skipped {prefix}"] = "missing relative_pose"
        return _zero(anchor), loss_dict

    depth = _depth_from_pred(pred)
    if depth is None:
        loss_dict[f"skipped {prefix}"] = "missing predicted depth/pts3d"
        return _zero(anchor), loss_dict
    images = images.to(device=depth.device, dtype=depth.dtype)
    log_i, log_j = _grayscale_log(images)
    if log_i is None:
        loss_dict[f"skipped {prefix}"] = "images must be [B,2,C,H,W]"
        return _zero(anchor), loss_dict
    if depth.shape[-2:] != log_i.shape[-2:]:
        depth = F.interpolate(depth, size=log_i.shape[-2:], mode="bilinear", align_corners=False)

    warped_j, valid = _warp_second_to_first(log_j, depth, intrinsics, relative_pose)
    pred_log_delta = warped_j - log_i
    event_delta = _event_sum(event_voxel)
    if event_delta is None:
        loss_dict[f"skipped {prefix}"] = "invalid event_voxel"
        return _zero(anchor), loss_dict
    event_delta = event_delta.to(device=depth.device, dtype=depth.dtype)
    if event_delta.shape[-2:] != pred_log_delta.shape[-2:]:
        event_delta = F.interpolate(event_delta, size=pred_log_delta.shape[-2:], mode="bilinear", align_corners=False)

    valid = valid & torch.isfinite(pred_log_delta) & torch.isfinite(event_delta)
    if not valid.any():
        loss_dict[f"skipped {prefix}"] = "no valid warped pixels"
        return _zero(anchor), loss_dict
    a = _znorm(pred_log_delta, valid)
    b = _znorm(event_delta, valid)
    cosine = F.cosine_similarity(a[valid].flatten(), b[valid].flatten(), dim=0)
    loss = (1.0 - cosine) * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    return loss, loss_dict


def supervised_depth_gt_loss(
    pred,
    batch,
    weight=1.0,
    loss_dict=None,
    prefix="depth_gt",
):
    """Supervised depth loss. Only real sensor/dataset depth_gt is accepted."""
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(pred, batch)
    if not isinstance(batch, dict):
        loss_dict[f"skipped {prefix}"] = "batch is not dict"
        return _zero(anchor), loss_dict
    depth_gt = batch.get("depth_gt", batch.get("depth_gt1", None))
    if depth_gt is None:
        loss_dict[f"skipped {prefix}"] = "missing real depth_gt"
        return _zero(anchor), loss_dict
    depth_source = str(batch.get("depth_source", "real")).lower()
    if "pseudo" in depth_source or "teacher" in depth_source or "external" in depth_source:
        loss_dict[f"skipped {prefix}"] = f"invalid depth_source={depth_source}"
        return _zero(anchor), loss_dict

    pred_depth = _depth_from_pred(pred)
    if pred_depth is None:
        loss_dict[f"skipped {prefix}"] = "missing predicted depth/pts3d"
        return _zero(anchor), loss_dict
    depth_gt = _as_bchw(depth_gt.to(device=pred_depth.device, dtype=pred_depth.dtype))
    if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
        pred_depth = F.interpolate(pred_depth, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
    valid = batch.get("depth_valid", torch.isfinite(depth_gt) & (depth_gt > 0))
    valid = _as_bchw(valid.to(device=pred_depth.device, dtype=torch.bool))
    valid = valid & torch.isfinite(pred_depth) & torch.isfinite(depth_gt) & (pred_depth > 0) & (depth_gt > 0)
    if not valid.any():
        loss_dict[f"skipped {prefix}"] = "no valid real depth_gt pixels"
        return _zero(anchor), loss_dict

    loss = SILogLoss()(pred_depth, depth_gt, valid) * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    return loss, loss_dict


def pseudo_depth_distillation_loss(
    pred,
    batch,
    weight=0.25,
    confidence_percentile=40.0,
    huber_delta=0.1,
    min_valid_fraction=0.05,
    metric_scale_weight=0.0,
    loss_dict=None,
    prefix="pseudo_depth",
):
    """Scale-aligned, confidence-masked distillation from external pseudo depth."""
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(pred, batch)
    if not isinstance(batch, dict):
        loss_dict[f"skipped {prefix}"] = "batch is not dict"
        return _zero(anchor), loss_dict
    target = batch.get("depth_pseudo", batch.get("depth_pseudo1", None))
    if target is None:
        loss_dict[f"skipped {prefix}"] = "missing pseudo depth"
        return _zero(anchor), loss_dict

    pred_depth = _depth_from_pred(pred)
    if pred_depth is None:
        loss_dict[f"skipped {prefix}"] = "missing predicted depth/pts3d"
        return _zero(anchor), loss_dict
    target = _as_bchw(target.to(device=pred_depth.device, dtype=pred_depth.dtype))
    if pred_depth.shape[-2:] != target.shape[-2:]:
        pred_depth = F.interpolate(
            pred_depth, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    valid = batch.get(
        "depth_pseudo_valid", torch.isfinite(target) & (target > 0)
    )
    valid = _as_bchw(valid.to(device=pred_depth.device, dtype=torch.bool))
    valid = (
        valid
        & torch.isfinite(pred_depth)
        & torch.isfinite(target)
        & (pred_depth > 0)
        & (target > 0)
    )

    confidence = batch.get("depth_pseudo_conf", None)
    if confidence is not None:
        confidence = _as_bchw(
            confidence.to(device=pred_depth.device, dtype=pred_depth.dtype)
        )
        if confidence.shape[-2:] != target.shape[-2:]:
            confidence = F.interpolate(
                confidence, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

    losses = []
    scale_losses = []
    scales = []
    valid_fractions = []
    for item_index in range(pred_depth.shape[0]):
        item_valid = valid[item_index]
        if confidence is not None:
            item_conf = confidence[item_index]
            finite_conf = item_valid & torch.isfinite(item_conf)
            if finite_conf.any():
                quantile = torch.quantile(
                    item_conf[finite_conf].float(),
                    float(confidence_percentile) / 100.0,
                ).to(dtype=item_conf.dtype)
                item_valid = finite_conf & (item_conf >= quantile)
        valid_fraction = float(item_valid.float().mean().detach().cpu())
        valid_fractions.append(valid_fraction)
        if valid_fraction < float(min_valid_fraction):
            continue

        pred_log = torch.log(pred_depth[item_index][item_valid].float().clamp_min(1e-6))
        target_log = torch.log(target[item_index][item_valid].float().clamp_min(1e-6))
        log_scale = torch.median(target_log - pred_log).detach()
        residual = pred_log + log_scale - target_log
        losses.append(
            F.huber_loss(
                residual,
                torch.zeros_like(residual),
                delta=float(huber_delta),
                reduction="mean",
            )
        )
        if metric_scale_weight > 0:
            scale_residual = (pred_log - target_log).mean()
            scale_losses.append(
                F.huber_loss(
                    scale_residual,
                    torch.zeros_like(scale_residual),
                    delta=float(huber_delta),
                    reduction="mean",
                )
            )
        scales.append(torch.exp(log_scale))

    if not losses:
        loss_dict[f"skipped {prefix}"] = "insufficient confident pseudo-depth pixels"
        return _zero(anchor), loss_dict
    shape_loss = torch.stack(losses).mean()
    scale_loss = (
        torch.stack(scale_losses).mean()
        if scale_losses
        else shape_loss.new_zeros(())
    )
    loss = (shape_loss + float(metric_scale_weight) * scale_loss) * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    loss_dict[f"{prefix}_shape"] = float((shape_loss * weight).detach().cpu())
    loss_dict[f"{prefix}_metric_scale"] = float(
        (scale_loss * weight * float(metric_scale_weight)).detach().cpu()
    )
    loss_dict[f"{prefix}_valid_fraction"] = float(np.mean(valid_fractions))
    loss_dict[f"{prefix}_median_scale"] = float(
        torch.stack(scales).median().detach().cpu()
    )
    return loss, loss_dict


def _masked_gradient_l1(pred, target, valid, edge_weight=None):
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    if edge_weight is not None:
        weight_dx = edge_weight[..., :, 1:].maximum(edge_weight[..., :, :-1])
        weight_dy = edge_weight[..., 1:, :].maximum(edge_weight[..., :-1, :])
    else:
        weight_dx = None
        weight_dy = None

    losses = []
    for diff, mask, weight_map in (
        ((pred_dx - target_dx).abs(), valid_dx, weight_dx),
        ((pred_dy - target_dy).abs(), valid_dy, weight_dy),
    ):
        mask = mask & torch.isfinite(diff)
        if not mask.any():
            continue
        if weight_map is not None:
            weights = weight_map[mask].clamp_min(1.0)
            losses.append((diff[mask] * weights).sum() / weights.sum().clamp_min(1e-6))
        else:
            losses.append(diff[mask].mean())
    if not losses:
        return None
    return sum(losses) / len(losses)


def depth_gradient_gt_loss(
    pred,
    batch,
    weight=0.05,
    image_edge_weight=1.0,
    event_edge_weight=0.5,
    log_depth=True,
    loss_dict=None,
    prefix="depth_gradient_gt",
):
    """Align predicted and real depth gradients, emphasizing image/event edges."""
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(pred, batch)
    if not isinstance(batch, dict):
        loss_dict[f"skipped {prefix}"] = "batch is not dict"
        return _zero(anchor), loss_dict

    depth_gt = batch.get("depth_gt", batch.get("depth_gt1", None))
    if depth_gt is None:
        loss_dict[f"skipped {prefix}"] = "missing real depth_gt"
        return _zero(anchor), loss_dict
    pred_depth = _depth_from_pred(pred)
    if pred_depth is None:
        loss_dict[f"skipped {prefix}"] = "missing predicted depth/pts3d"
        return _zero(anchor), loss_dict

    depth_gt = _as_bchw(depth_gt.to(device=pred_depth.device, dtype=pred_depth.dtype))
    if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
        pred_depth = F.interpolate(pred_depth, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
    valid = batch.get("depth_valid", torch.isfinite(depth_gt) & (depth_gt > 0))
    valid = _as_bchw(valid.to(device=pred_depth.device, dtype=torch.bool))
    valid = valid & torch.isfinite(pred_depth) & torch.isfinite(depth_gt) & (pred_depth > 0) & (depth_gt > 0)
    if valid.sum() < 4:
        loss_dict[f"skipped {prefix}"] = "too few valid real depth_gt pixels"
        return _zero(anchor), loss_dict

    pred_for_grad = pred_depth.clamp_min(1e-6)
    gt_for_grad = depth_gt.clamp_min(1e-6)
    if log_depth:
        pred_for_grad = pred_for_grad.log()
        gt_for_grad = gt_for_grad.log()

    edge_weight = None
    if image_edge_weight > 0 and "images_photo" in batch:
        images = batch.get("images_photo_dark", batch.get("images_photo"))
        if images is not None and torch.is_tensor(images) and images.ndim == 5:
            image = images[:, 0].to(device=pred_depth.device, dtype=pred_depth.dtype).mean(dim=1, keepdim=True)
            if image.shape[-2:] != depth_gt.shape[-2:]:
                image = F.interpolate(image, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
            edge_weight = _gradient_magnitude(image) * float(image_edge_weight)
    if event_edge_weight > 0 and "event_voxel" in batch:
        activity = _event_activity(batch["event_voxel"])
        if activity is not None:
            activity = activity.to(device=pred_depth.device, dtype=pred_depth.dtype)
            if activity.shape[-2:] != depth_gt.shape[-2:]:
                activity = F.interpolate(activity, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
            event_edges = _gradient_magnitude(activity)
            edge_weight = event_edges * float(event_edge_weight) if edge_weight is None else edge_weight + event_edges * float(event_edge_weight)
    if edge_weight is not None:
        edge_weight = 1.0 + _znorm(edge_weight).clamp_min(0.0)

    grad_loss = _masked_gradient_l1(pred_for_grad, gt_for_grad, valid, edge_weight=edge_weight)
    if grad_loss is None or not torch.isfinite(grad_loss):
        loss_dict[f"skipped {prefix}"] = "no finite valid depth gradients"
        return _zero(anchor), loss_dict

    loss = grad_loss * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    return loss, loss_dict


def pose_gt_loss(
    pred,
    batch,
    weight=1.0,
    cycle_weight=0.1,
    rotation_weight=1.0,
    rotation_chordal_weight=0.0,
    translation_weight=1.0,
    min_translation_norm=1.0e-3,
    loss_dict=None,
):
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(pred, batch)
    if not isinstance(batch, dict):
        loss_dict["skipped pose_gt"] = "batch is not dict"
        return _zero(anchor), loss_dict
    pred_pose = pred.get("pose", pred.get("relative_pose", None)) if isinstance(pred, dict) else None
    pose_gt = batch.get("pose_gt", batch.get("relative_pose_gt", None))
    if pred_pose is None:
        loss_dict["skipped pose_gt"] = "missing predicted pose"
        return _zero(anchor), loss_dict
    if pose_gt is None:
        loss_dict["skipped pose_gt"] = "missing pose_gt"
        return _zero(anchor), loss_dict

    pred_pose = pred_pose.to(dtype=torch.float32)
    pose_gt = pose_gt.to(device=pred_pose.device, dtype=torch.float32)
    r_rel = pred_pose[..., :3, :3].transpose(-1, -2) @ pose_gt[..., :3, :3]
    trace = r_rel.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            r_rel[..., 2, 1] - r_rel[..., 1, 2],
            r_rel[..., 0, 2] - r_rel[..., 2, 0],
            r_rel[..., 1, 0] - r_rel[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    rot_loss = torch.atan2(sine, cosine).mean()
    identity = torch.eye(3, device=r_rel.device, dtype=r_rel.dtype)
    chordal_loss = (
        torch.linalg.matrix_norm(r_rel - identity, ord="fro", dim=(-2, -1))
        / (2.0**0.5)
    ).mean()
    t_pred = pred_pose[..., :3, 3]
    t_gt = pose_gt[..., :3, 3]
    translation_norm = t_gt.norm(dim=-1)
    valid_translation = translation_norm >= float(min_translation_norm)
    if valid_translation.any():
        scale = translation_norm[valid_translation, None]
        trans_loss = (
            (t_pred[valid_translation] - t_gt[valid_translation]) / scale
        ).abs().mean()
    else:
        trans_loss = t_pred.sum() * 0.0
    loss = (
        float(rotation_weight) * rot_loss
        + float(rotation_chordal_weight) * chordal_loss
        + float(translation_weight) * trans_loss
    ) * weight
    loss_dict["pose_gt"] = float(loss.detach().cpu())
    loss_dict["pose_rotation_rad"] = float(rot_loss.detach().cpu())
    loss_dict["pose_rotation_chordal"] = float(chordal_loss.detach().cpu())
    loss_dict["pose_translation_relative_l1"] = float(trans_loss.detach().cpu())
    loss_dict["pose_translation_valid_fraction"] = float(
        valid_translation.float().mean().detach().cpu()
    )

    pred_seq = pred.get("poses", None) if isinstance(pred, dict) else None
    if cycle_weight > 0 and pred_seq is not None and pred_seq.shape[-3] >= 3:
        cycle = pred_seq[..., 0, :, :] @ pred_seq[..., 1, :, :] @ pred_seq[..., 2, :, :]
        eye = torch.eye(4, device=cycle.device, dtype=cycle.dtype).expand_as(cycle)
        cycle_loss = F.smooth_l1_loss(cycle, eye) * cycle_weight
        loss = loss + cycle_loss
        loss_dict["pose_cycle"] = float(cycle_loss.detach().cpu())
    elif cycle_weight > 0:
        loss_dict["skipped pose_cycle"] = "need predicted pose cycle with >=3 transforms"
    return loss, loss_dict
