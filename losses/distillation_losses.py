import torch
import torch.nn.functional as F


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


def _last_tensor(features):
    if features is None:
        return None
    if torch.is_tensor(features):
        return features
    if isinstance(features, (list, tuple)):
        for item in reversed(features):
            out = _last_tensor(item)
            if out is not None:
                return out
    if isinstance(features, dict):
        for key in ("feature", "features", "tokens", "feat", "desc"):
            if key in features:
                return _last_tensor(features[key])
    return None


def _flatten_feature(x):
    if x is None:
        return None
    if x.ndim == 4:
        return x.flatten(2).transpose(1, 2)
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x.unsqueeze(1)
    return None


def feature_distillation_loss(
    student_features,
    teacher_features,
    weight=0.05,
    loss_dict=None,
    prefix="feat_distill",
):
    """Weakly align student features to frozen RGB-only teacher features."""
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(student_features, teacher_features)
    student = _flatten_feature(_last_tensor(student_features))
    teacher = _flatten_feature(_last_tensor(teacher_features))
    if student is None or teacher is None:
        loss_dict[f"skipped {prefix}"] = "missing features"
        return _zero(anchor), loss_dict
    if student.shape != teacher.shape:
        loss_dict[f"skipped {prefix}"] = f"shape mismatch {student.shape} vs {teacher.shape}"
        return _zero(anchor), loss_dict

    student = student.float()
    teacher = teacher.detach().float()
    finite = torch.isfinite(student).all(dim=-1) & torch.isfinite(teacher).all(dim=-1)
    if not finite.any():
        loss_dict[f"skipped {prefix}"] = "no finite feature tokens"
        return _zero(anchor), loss_dict

    student = F.normalize(student[finite], dim=-1, eps=1e-6)
    teacher = F.normalize(teacher[finite], dim=-1, eps=1e-6)
    cosine = (student * teacher).sum(dim=-1).clamp(-1.0, 1.0)
    loss = (1.0 - cosine).mean() * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    return loss, loss_dict


def _iter_pred_pairs(student_outputs, teacher_outputs):
    if isinstance(student_outputs, dict) and isinstance(teacher_outputs, dict):
        yield student_outputs, teacher_outputs
        return
    if isinstance(student_outputs, (list, tuple)) and isinstance(teacher_outputs, (list, tuple)):
        for student_pred, teacher_pred in zip(student_outputs, teacher_outputs):
            if isinstance(student_pred, dict) and isinstance(teacher_pred, dict):
                yield student_pred, teacher_pred


def _shared_points(student_pred, teacher_pred):
    for key in ("pts3d", "pts3d_in_other_view"):
        if key in student_pred and key in teacher_pred:
            return key, student_pred[key], teacher_pred[key]
    return None, None, None


def _confidence_mask(teacher_pred, points, conf_thr):
    conf = teacher_pred.get("conf", None)
    if conf is None:
        return torch.ones(points.shape[:-1], device=points.device, dtype=torch.bool)
    if conf.ndim == points.ndim and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 4 and conf.shape[1] == 1:
        conf = conf[:, 0]
    while conf.ndim > points.ndim - 1:
        if conf.shape[-1] == 1:
            conf = conf.squeeze(-1)
        elif conf.shape[1] == 1:
            conf = conf.squeeze(1)
        else:
            break
    if conf.shape != points.shape[:-1]:
        return torch.zeros(points.shape[:-1], device=points.device, dtype=torch.bool)
    return conf > conf_thr


def structure_consistency_loss(
    student_outputs,
    teacher_outputs,
    weight=0.1,
    conf_thr=1.0,
    min_scale=1.0,
    max_scale=1000.0,
    max_point_norm=10000.0,
    max_residual=50.0,
    min_valid_fraction=0.01,
    loss_dict=None,
    prefix="structure_consistency",
):
    """Weak confidence-aware structure consistency.

    The teacher prediction is treated as a frozen consistency target, not as
    ground-truth depth supervision.
    """
    loss_dict = {} if loss_dict is None else loss_dict
    anchor = _anchor_tensor(student_outputs, teacher_outputs)
    total = None
    used = 0
    valid_points = 0
    total_points = 0
    scales = []

    for student_pred, teacher_pred in _iter_pred_pairs(student_outputs, teacher_outputs):
        _, student_pts, teacher_pts = _shared_points(student_pred, teacher_pred)
        if student_pts is None or teacher_pts is None:
            continue
        if student_pts.shape != teacher_pts.shape:
            continue

        student_pts = student_pts.float()
        teacher_pts = teacher_pts.detach().float()
        finite = torch.isfinite(student_pts).all(dim=-1) & torch.isfinite(teacher_pts).all(dim=-1)
        conf_mask = _confidence_mask(teacher_pred, student_pts, conf_thr).to(student_pts.device)
        mask = finite & conf_mask
        candidate_points = int(mask.sum().detach().cpu())
        total_points += int(mask.numel())
        if max_point_norm is not None and max_point_norm > 0:
            student_norm = torch.linalg.vector_norm(student_pts.detach(), dim=-1)
            teacher_norm = torch.linalg.vector_norm(teacher_pts, dim=-1)
            norm_mask = (
                torch.isfinite(student_norm)
                & torch.isfinite(teacher_norm)
                & (student_norm <= max_point_norm)
                & (teacher_norm <= max_point_norm)
            )
            mask = mask & norm_mask
        view_valid_points = int(mask.sum().detach().cpu())
        if view_valid_points == 0:
            continue
        if candidate_points > 0 and view_valid_points / candidate_points < float(min_valid_fraction):
            continue

        teacher_norm = torch.linalg.vector_norm(teacher_pts[mask], dim=-1)
        scale = teacher_norm.median()
        if not torch.isfinite(scale):
            continue
        scale = scale.clamp(min=float(min_scale), max=float(max_scale))
        residual = (student_pts[mask] - teacher_pts[mask]) / scale
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if max_residual is not None and max_residual > 0:
            residual = residual.clamp(min=-float(max_residual), max=float(max_residual))

        # log1p keeps the loss and gradient bounded for severe 3D outliers.
        view_loss = torch.log1p(residual.abs()).mean()
        total = view_loss if total is None else total + view_loss
        used += 1
        valid_points += view_valid_points
        scales.append(float(scale.detach().cpu()))

    if used == 0:
        loss_dict[f"skipped {prefix}"] = "missing compatible pts3d/confidence"
        return _zero(anchor), loss_dict

    loss = total / used * weight
    loss_dict[prefix] = float(loss.detach().cpu())
    if total_points > 0:
        loss_dict[f"{prefix}_valid_fraction"] = valid_points / max(total_points, 1)
    if scales:
        loss_dict[f"{prefix}_scale"] = sum(scales) / len(scales)
    return loss, loss_dict
