#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from evo.core import metrics
from evo.core.trajectory import PosePath3D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DUST3R_ROOT = REPO_ROOT / "dust3r"
if str(DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(DUST3R_ROOT))

from dust3r.post_process import estimate_focal_knowing_depth  # noqa: E402
from mast3r.fast_nn import extract_correspondences_nonsym  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    build_dataset,
    resolve_amp_config,
    load_config,
    move_batch_to_device,
    safe_torch_load,
)
from utils.eval_event_dark3r_variant import (  # noqa: E402
    apply_event_mode,
    build_model,
    cfg_without_lora,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Recover Night3R trajectories with dense PnP and evaluate ATE/RPE"
    )
    parser.add_argument(
        "--variant",
        choices=("mast3r", "dark3r_lora", "event_dark3r"),
        default="event_dark3r",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--pose-head-checkpoint",
        default=None,
        help=(
            "Optional pose-head-only checkpoint loaded on top of --checkpoint. "
            "This is used for lightweight checkpoint sweeps."
        ),
    )
    parser.add_argument("--dark3r-lora-checkpoint", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--event-mode", choices=("real", "zero", "shuffle"), default="real")
    parser.add_argument(
        "--pose-recovery",
        choices=("direct", "pointmap_sim3", "essential", "pnp"),
        default="pointmap_sim3",
        help="Use the direct pose head or recover pose from descriptors/point maps.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--max-frames-per-scene", type=int, default=32)
    parser.add_argument(
        "--include-scenes",
        nargs="+",
        default=None,
        help=(
            "Evaluate only these exact scene names. Comma-separated values are "
            "also accepted. This filter is applied after dataset discovery so it "
            "does not depend on lexical scene ordering."
        ),
    )
    parser.add_argument(
        "--export-trajectories",
        action="store_true",
        help=(
            "Store the composed predicted and GT camera-to-world trajectories "
            "for each evaluated scene in the JSON output."
        ),
    )
    parser.add_argument("--conf-quantile", type=float, default=0.70)
    parser.add_argument(
        "--max-pointmap-residual",
        type=float,
        default=0.25,
        help=(
            "Reject point-map Sim(3) registrations whose trimmed 3D RMSE, "
            "normalized by the target point-map extent, exceeds this value."
        ),
    )
    parser.add_argument("--max-pnp-points", type=int, default=20000)
    parser.add_argument("--pnp-reproj-error", type=float, default=4.0)
    parser.add_argument("--pnp-iterations", type=int, default=200)
    parser.add_argument("--min-pnp-inliers", type=int, default=64)
    parser.add_argument(
        "--pnp-intrinsics",
        choices=("search", "predicted", "gt"),
        default="search",
        help=(
            "Search the Dust3R focal range, estimate focal from the model point "
            "map, or use the dataset calibration for PnP."
        ),
    )
    parser.add_argument("--match-stride", type=int, default=8)
    parser.add_argument("--max-matches", type=int, default=4096)
    parser.add_argument("--essential-reproj-error", type=float, default=1.5)
    parser.add_argument("--essential-iterations", type=int, default=2000)
    parser.add_argument("--min-essential-inliers", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--lowlight-exposure",
        type=float,
        default=None,
        help=(
            "Apply a fixed synthetic exposure in the linearized RGB domain. "
            "When omitted, stored RGB is evaluated without test-time darkening."
        ),
    )
    parser.add_argument("--lowlight-read-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-gamma", type=float, default=2.2)
    parser.add_argument(
        "--lowlight-seed",
        type=int,
        default=20260716,
        help="Base seed for sample-keyed low-light noise; this is not a model seed sweep.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.lowlight_exposure is not None and not 0.0 < args.lowlight_exposure <= 1.0:
        parser.error("--lowlight-exposure must be in (0, 1]")
    if args.lowlight_read_noise_std < 0.0 or args.lowlight_shot_noise_std < 0.0:
        parser.error("low-light noise standard deviations must be non-negative")
    if args.lowlight_gamma <= 0.0:
        parser.error("--lowlight-gamma must be positive")
    if args.max_pointmap_residual <= 0.0:
        parser.error("--max-pointmap-residual must be positive")
    return args


def make_train_args(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent),
        model_name_or_path=args.model_name_or_path,
        dark3r_lora_checkpoint=args.dark3r_lora_checkpoint,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=1,
        max_batches=None,
        num_epochs=None,
        max_steps=None,
        max_scenes=args.max_scenes,
        max_pairs_per_scene=max(args.max_frames_per_scene - 1, 1),
        pair_stride=1,
        pair_sampling="fixed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=args.num_workers,
        stage=None,
        partial_finetune_blocks=None,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        depth_min=None,
        depth_max=None,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pose_head_checkpoint(model, checkpoint_path, allow_unsafe=False):
    if not getattr(model, "pose_head_enabled", False) or model.pose_head is None:
        raise RuntimeError(
            "A pose-head-only checkpoint requires pose_head.enabled=true in the config"
        )
    payload = safe_torch_load(
        checkpoint_path,
        map_location="cpu",
        allow_unsafe=allow_unsafe,
        purpose="pose-head-only checkpoint",
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Pose-head checkpoint payload must be a dictionary")
    state_dict = payload.get("pose_head", None)
    if state_dict is None and isinstance(payload.get("model"), dict):
        state_dict = {
            key[len("pose_head.") :]: value
            for key, value in payload["model"].items()
            if key.startswith("pose_head.")
        }
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            f"No pose-head state was found in checkpoint: {checkpoint_path}"
        )
    missing, unexpected = model.pose_head.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_kind": payload.get("checkpoint_kind", "pose_head_only"),
        "base_checkpoint": payload.get("base_checkpoint"),
        "step": payload.get("step"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _sample_keyed_seed(sample_id, base_seed):
    digest = hashlib.blake2b(str(sample_id).encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, byteorder="little") + int(base_seed)) % (2**63 - 1)


def apply_deterministic_lowlight(batch, args):
    """Create identical fixed-exposure RGB inputs across all model/event variants."""
    if args.lowlight_exposure is None:
        return batch, None
    images = batch.get("images_photo")
    if not torch.is_tensor(images):
        raise KeyError("Fixed low-light evaluation requires batch['images_photo']")

    sample_id = _batch_string(batch, "sample_id")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_sample_keyed_seed(sample_id, args.lowlight_seed))

    clean = images.detach().float().cpu().clamp(0.0, 1.0)
    linear = clean.pow(float(args.lowlight_gamma))
    dark_linear = linear * float(args.lowlight_exposure)
    if args.lowlight_shot_noise_std > 0.0:
        shot_std = torch.sqrt(dark_linear.clamp_min(0.0) + 1.0e-6)
        shot_std = shot_std * float(args.lowlight_shot_noise_std)
        dark_linear = dark_linear + torch.randn(
            dark_linear.shape, generator=generator, dtype=dark_linear.dtype
        ) * shot_std
    if args.lowlight_read_noise_std > 0.0:
        dark_linear = dark_linear + torch.randn(
            dark_linear.shape, generator=generator, dtype=dark_linear.dtype
        ) * float(args.lowlight_read_noise_std)
    dark = dark_linear.clamp(0.0, 1.0).pow(1.0 / float(args.lowlight_gamma))

    updated = dict(batch)
    updated["images_photo_dark"] = dark.to(dtype=images.dtype)
    updated["images_dark"] = updated["images_photo_dark"] * 2.0 - 1.0
    diagnostics = {
        "clean_mean": float(clean.mean()),
        "dark_mean": float(dark.mean()),
        "dark_to_clean_mean_ratio": float(dark.mean() / clean.mean().clamp_min(1.0e-8)),
        "dark_zero_fraction": float((dark <= 0.0).float().mean()),
    }
    return updated, diagnostics


def aggregate_lowlight_diagnostics(values):
    if not values:
        return None
    return {
        key: float(np.mean([item[key] for item in values]))
        for key in values[0]
    }


def _batch_scalar(batch, key):
    value = batch[key]
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _batch_string(batch, key):
    value = batch[key]
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def swap_pair_batch(batch):
    """Swap the two views while leaving scene-level calibration and GT untouched."""
    swapped = dict(batch)
    for key in (
        "images",
        "images_dark",
        "images_photo",
        "images_photo_dark",
        "event_voxel",
    ):
        value = swapped.get(key)
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == 2:
            swapped[key] = value.flip(1)
    return swapped


def _rotation_error_deg(r_est, r_gt):
    r_delta = r_est @ r_gt.T
    cosine = np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def project_to_so3(rotation):
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vt
    return projected


def convert_dl3dv_relative_pose_to_opencv(relative_pose):
    axis_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return axis_flip @ np.asarray(relative_pose, dtype=np.float64) @ axis_flip


def estimate_intrinsics_from_pointmap(points):
    """Estimate the square-pixel pinhole intrinsics used by Dust3R/MASt3R."""
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    if points.ndim == 3:
        points = points.unsqueeze(0)
    if points.ndim != 4 or points.shape[0] != 1 or points.shape[-1] != 3:
        raise ValueError(f"Expected one point map [1,H,W,3], got {tuple(points.shape)}")
    _, h, w, _ = points.shape
    principal_point = points.new_tensor([[w / 2.0, h / 2.0]])
    focal = estimate_focal_knowing_depth(
        points.float(), principal_point, focal_mode="weiszfeld"
    ).reshape(-1)[0]
    focal_value = float(focal.detach().cpu())
    if not np.isfinite(focal_value) or focal_value <= 0:
        raise ValueError(f"Invalid focal estimated from model point map: {focal_value}")
    intrinsics = np.array(
        [
            [focal_value, 0.0, w / 2.0],
            [0.0, focal_value, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsics, focal_value


def _reprojection_rmse(object_points, image_points, camera_matrix, rvec, tvec):
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, None
    )
    residual = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def estimate_relative_pose_pnp(
    points_in_camera_i,
    confidence,
    intrinsics_j,
    conf_quantile=0.70,
    max_points=20000,
    reprojection_error=4.0,
    iterations=200,
    min_inliers=64,
    seed=17,
    focal_candidates=None,
):
    """Estimate T_camera_j_from_camera_i from a view-j point map in camera i."""
    points = _as_numpy(points_in_camera_i)
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected point map [H,W,3], got {points.shape}")

    confidence = _as_numpy(confidence)
    while confidence.ndim > 2 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.ndim == 3 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.shape != points.shape[:2]:
        raise ValueError(f"Confidence shape {confidence.shape} != point grid {points.shape[:2]}")

    h, w = points.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    finite &= np.linalg.norm(points, axis=-1) > 1e-6
    finite_values = confidence[finite]
    if finite_values.size < max(min_inliers, 8):
        return None, {"success": False, "reason": "insufficient_finite_points"}

    threshold = float(np.quantile(finite_values, np.clip(conf_quantile, 0.0, 1.0)))
    valid = finite & (confidence >= threshold)
    indices = np.flatnonzero(valid.reshape(-1))
    if indices.size < max(min_inliers, 8):
        return None, {"success": False, "reason": "insufficient_confident_points"}

    rng = np.random.default_rng(seed)
    if indices.size > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    object_points = points.reshape(-1, 3)[indices].astype(np.float64)
    image_points = np.stack([xx.reshape(-1)[indices], yy.reshape(-1)[indices]], axis=-1)
    image_points = image_points.astype(np.float64)
    camera_matrix = (
        None
        if intrinsics_j is None
        else _as_numpy(intrinsics_j).reshape(3, 3).astype(np.float64)
    )

    def run_ransac(obj, img, intrinsics, iteration_count):
        cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
        return cv2.solvePnPRansac(
            obj,
            img,
            intrinsics,
            None,
            iterationsCount=int(iteration_count),
            reprojectionError=float(reprojection_error),
            confidence=0.999,
            flags=cv2.SOLVEPNP_SQPNP,
        )

    search_diagnostics = {}
    if focal_candidates is not None:
        candidates = np.asarray(focal_candidates, dtype=np.float64).reshape(-1)
        candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
        if candidates.size == 0:
            raise ValueError("focal_candidates must contain at least one positive value")
        search_count = min(indices.size, 5000)
        search_ids = rng.choice(indices.size, size=search_count, replace=False)
        search_object_points = object_points[search_ids]
        search_image_points = image_points[search_ids]
        best = None
        candidate_scores = []
        for candidate in candidates:
            candidate_matrix = np.array(
                [
                    [candidate, 0.0, w / 2.0],
                    [0.0, candidate, h / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            success_i, rvec_i, tvec_i, inliers_i = run_ransac(
                search_object_points,
                search_image_points,
                candidate_matrix,
                min(int(iterations), 100),
            )
            score = 0 if inliers_i is None else int(inliers_i.size)
            candidate_scores.append((float(candidate), score))
            if success_i and (best is None or score > best[0]):
                best = score, candidate_matrix, rvec_i, tvec_i
        if best is None:
            return None, {
                "success": False,
                "reason": "focal_search_failed",
                "selected_points": int(indices.size),
                "focal_candidates": int(candidates.size),
                "best_search_inliers": int(max((s for _, s in candidate_scores), default=0)),
            }
        camera_matrix = best[1]
        search_diagnostics = {
            "focal_search_candidates": int(candidates.size),
            "focal_search_points": int(search_count),
            "focal_search_best_inliers": int(best[0]),
            "focal_px": float(camera_matrix[0, 0]),
        }
    if camera_matrix is None:
        raise ValueError("intrinsics_j or focal_candidates is required")

    success, rvec, tvec, inliers = run_ransac(
        object_points,
        image_points,
        camera_matrix,
        iterations,
    )
    inlier_count = 0 if inliers is None else int(inliers.size)
    if not success or inlier_count < int(min_inliers):
        return None, {
            "success": False,
            "reason": "pnp_failed_or_too_few_inliers",
            "selected_points": int(indices.size),
            "inliers": inlier_count,
            **search_diagnostics,
        }

    inlier_ids = inliers.reshape(-1)
    inlier_object_points = object_points[inlier_ids]
    inlier_image_points = image_points[inlier_ids]
    reproj_rmse = _reprojection_rmse(
        inlier_object_points, inlier_image_points, camera_matrix, rvec, tvec
    )
    refined = False
    try:
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            inlier_object_points,
            inlier_image_points,
            camera_matrix,
            None,
            rvec.copy(),
            tvec.copy(),
        )
        refined_rmse = _reprojection_rmse(
            inlier_object_points,
            inlier_image_points,
            camera_matrix,
            refined_rvec,
            refined_tvec,
        )
        if np.isfinite(refined_rmse) and refined_rmse < reproj_rmse:
            rvec, tvec = refined_rvec, refined_tvec
            reproj_rmse = refined_rmse
            refined = True
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return transform, {
        "success": True,
        "selected_points": int(indices.size),
        "inliers": inlier_count,
        "inlier_ratio": float(inlier_count / max(indices.size, 1)),
        "reprojection_rmse_px": reproj_rmse,
        "geometric_residual": reproj_rmse,
        "geometric_residual_unit": "px",
        "refined": refined,
        "confidence_threshold": threshold,
        **search_diagnostics,
    }


def reciprocal_descriptor_matches(
    descriptor1,
    descriptor2,
    confidence1=None,
    confidence2=None,
    stride=8,
    max_matches=4096,
):
    """Find upstream MASt3R reciprocal matches from regular-grid seeds."""
    if descriptor1.ndim == 4:
        descriptor1 = descriptor1[0]
    if descriptor2.ndim == 4:
        descriptor2 = descriptor2[0]
    if descriptor1.ndim != 3 or descriptor2.ndim != 3:
        raise ValueError("Descriptors must be [H,W,D] or [1,H,W,D]")
    if descriptor1.shape[-1] != descriptor2.shape[-1]:
        raise ValueError("Descriptor dimensions must match")
    if confidence1 is None:
        confidence1 = torch.ones(descriptor1.shape[:2], device=descriptor1.device)
    if confidence2 is None:
        confidence2 = torch.ones(descriptor2.shape[:2], device=descriptor2.device)
    while confidence1.ndim > 2:
        confidence1 = confidence1[0]
    while confidence2.ndim > 2:
        confidence2 = confidence2[0]
    points1, points2, scores = extract_correspondences_nonsym(
        descriptor1,
        descriptor2,
        confidence1.detach().cpu(),
        confidence2.detach().cpu(),
        subsample=max(int(stride), 1),
        device=descriptor1.device,
        ptmap_key="pred_desc",
    )
    if points1.shape[0] > max_matches:
        keep = torch.topk(scores, k=int(max_matches), sorted=False).indices
        points1, points2, scores = points1[keep], points2[keep], scores[keep]
    return (
        points1.detach().float().cpu().numpy(),
        points2.detach().float().cpu().numpy(),
        scores.detach().float().cpu().numpy(),
    )


def _essential_sampson_rmse_px(essential, points1, points2, focal_px, mask):
    x1 = np.column_stack([points1, np.ones(len(points1))])
    x2 = np.column_stack([points2, np.ones(len(points2))])
    ex1 = (essential @ x1.T).T
    etx2 = (essential.T @ x2.T).T
    numerator = np.sum(x2 * ex1, axis=1) ** 2
    denominator = (
        ex1[:, 0] ** 2
        + ex1[:, 1] ** 2
        + etx2[:, 0] ** 2
        + etx2[:, 1] ** 2
        + 1e-12
    )
    errors = np.sqrt(numerator / denominator)
    errors = errors[np.asarray(mask).reshape(-1).astype(bool)]
    return None if errors.size == 0 else float(np.sqrt(np.mean(errors**2)) * focal_px)


def estimate_relative_pose_essential(
    points1,
    points2,
    intrinsics1,
    intrinsics2=None,
    reprojection_error=1.5,
    iterations=2000,
    min_inliers=32,
    seed=17,
):
    """Estimate T_camera_2_from_camera_1 from calibrated 2D correspondences."""
    points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    if len(points1) != len(points2) or len(points1) < max(int(min_inliers), 8):
        return None, {
            "success": False,
            "reason": "insufficient_descriptor_matches",
            "matches": int(len(points1)),
        }
    k1 = _as_numpy(intrinsics1).reshape(3, 3).astype(np.float64)
    k2 = k1 if intrinsics2 is None else _as_numpy(intrinsics2).reshape(3, 3).astype(np.float64)
    normalized1 = cv2.undistortPoints(points1[:, None], k1, None).reshape(-1, 2)
    normalized2 = cv2.undistortPoints(points2[:, None], k2, None).reshape(-1, 2)
    focal_px = float(np.mean([k1[0, 0], k1[1, 1], k2[0, 0], k2[1, 1]]))
    normalized_threshold = float(reprojection_error) / max(focal_px, 1e-6)
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    essential, ransac_mask = cv2.findEssentialMat(
        normalized1,
        normalized2,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=normalized_threshold,
        maxIters=int(iterations),
    )
    if essential is None or ransac_mask is None:
        return None, {
            "success": False,
            "reason": "essential_matrix_failed",
            "matches": int(len(points1)),
        }
    essential_candidates = np.asarray(essential).reshape(-1, 3, 3)
    best = None
    for candidate in essential_candidates:
        pose_inliers, rotation, translation, pose_mask = cv2.recoverPose(
            candidate,
            normalized1,
            normalized2,
            np.eye(3),
            mask=ransac_mask.copy(),
        )
        if best is None or int(pose_inliers) > best[0]:
            best = int(pose_inliers), rotation, translation, pose_mask, candidate
    inlier_count, rotation, translation, pose_mask, essential = best
    if inlier_count < int(min_inliers):
        return None, {
            "success": False,
            "reason": "essential_too_few_inliers",
            "matches": int(len(points1)),
            "inliers": int(inlier_count),
        }
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    sampson_rmse = _essential_sampson_rmse_px(
        essential, normalized1, normalized2, focal_px, pose_mask
    )
    return transform, {
        "success": True,
        "matches": int(len(points1)),
        "inliers": int(inlier_count),
        "inlier_ratio": float(inlier_count / max(len(points1), 1)),
        "reprojection_rmse_px": sampson_rmse,
        "geometric_residual": sampson_rmse,
        "geometric_residual_unit": "px_sampson",
        "focal_px": focal_px,
    }


def _umeyama_sim3(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / max(len(source), 1)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError("Degenerate source point map")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def estimate_relative_pose_pointmap_sim3(
    points_in_camera1,
    points_in_camera2,
    confidence1,
    confidence2,
    conf_quantile=0.70,
    max_points=50000,
    trim_quantile=0.80,
    min_inliers=512,
    normalize_translation=True,
    max_normalized_rmse=0.25,
    seed=17,
):
    """Register the same dense point map expressed in camera 1 and camera 2."""
    source = _as_numpy(points_in_camera1)
    target = _as_numpy(points_in_camera2)
    if source.ndim == 4 and source.shape[0] == 1:
        source = source[0]
    if target.ndim == 4 and target.shape[0] == 1:
        target = target[0]
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"Point-map shapes must match [H,W,3], got {source.shape}, {target.shape}")
    conf1 = _as_numpy(confidence1)
    conf2 = _as_numpy(confidence2)
    while conf1.ndim > 2:
        conf1 = conf1[0]
    while conf2.ndim > 2:
        conf2 = conf2[0]
    if conf1.shape != source.shape[:2] or conf2.shape != source.shape[:2]:
        raise ValueError("Confidence maps must match point-map spatial shape")
    confidence = np.sqrt(np.clip(conf1 * conf2, 0, None))
    finite = np.isfinite(source).all(axis=-1) & np.isfinite(target).all(axis=-1)
    finite &= np.isfinite(confidence)
    finite &= (np.linalg.norm(source, axis=-1) > 1e-6)
    finite &= (np.linalg.norm(target, axis=-1) > 1e-6)
    if finite.sum() < max(int(min_inliers), 8):
        return None, {"success": False, "reason": "insufficient_finite_point_pairs"}
    threshold = float(np.quantile(confidence[finite], np.clip(conf_quantile, 0, 1)))
    indices = np.flatnonzero((finite & (confidence >= threshold)).reshape(-1))
    if indices.size < max(int(min_inliers), 8):
        return None, {
            "success": False,
            "reason": "insufficient_confident_point_pairs",
            "selected_points": int(indices.size),
        }
    rng = np.random.default_rng(seed)
    if indices.size > int(max_points):
        indices = rng.choice(indices, size=int(max_points), replace=False)
    source_points = source.reshape(-1, 3)[indices].astype(np.float64)
    target_points = target.reshape(-1, 3)[indices].astype(np.float64)
    try:
        scale, rotation, translation = _umeyama_sim3(source_points, target_points)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, {"success": False, "reason": f"sim3_failed: {exc}"}
    prediction = (scale * (rotation @ source_points.T)).T + translation
    residual = np.linalg.norm(prediction - target_points, axis=1)
    trim_threshold = float(np.quantile(residual, np.clip(trim_quantile, 0.5, 1.0)))
    inlier_mask = residual <= trim_threshold
    if inlier_mask.sum() < max(int(min_inliers), 8):
        return None, {
            "success": False,
            "reason": "sim3_too_few_trimmed_inliers",
            "selected_points": int(indices.size),
            "inliers": int(inlier_mask.sum()),
        }
    scale, rotation, translation = _umeyama_sim3(
        source_points[inlier_mask], target_points[inlier_mask]
    )
    prediction = (scale * (rotation @ source_points[inlier_mask].T)).T + translation
    residual = np.linalg.norm(prediction - target_points[inlier_mask], axis=1)
    target_extent = float(
        np.median(
            np.linalg.norm(
                target_points[inlier_mask] - np.median(target_points[inlier_mask], axis=0),
                axis=1,
            )
        )
    )
    translation_norm = float(np.linalg.norm(translation))
    if normalize_translation:
        if translation_norm <= 1e-9:
            return None, {
                "success": False,
                "reason": "degenerate_sim3_translation",
                "selected_points": int(indices.size),
            }
        translation = translation / translation_norm
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    normalized_rmse = float(
        np.sqrt(np.mean(residual**2)) / max(target_extent, 1e-9)
    )
    diagnostics = {
        "selected_points": int(indices.size),
        "inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "geometric_residual": normalized_rmse,
        "geometric_residual_unit": "normalized_3d_rmse",
        "max_geometric_residual": float(max_normalized_rmse),
        "sim3_scale": float(scale),
        "translation_norm_before_normalization": translation_norm,
        "confidence_threshold": threshold,
    }
    if normalized_rmse > float(max_normalized_rmse):
        diagnostics.update(
            success=False,
            reason="pointmap_residual_above_threshold",
        )
        return None, diagnostics
    diagnostics["success"] = True
    return transform, diagnostics


def select_best_pointmap_registration(registrations):
    if not registrations:
        raise ValueError("At least one point-map registration is required")
    successful = [item for item in registrations if item[0] is not None]
    candidates = successful or registrations
    return min(
        candidates,
        key=lambda item: item[1].get("geometric_residual", float("inf")),
    )


def trajectory_metrics(pred_poses, gt_poses):
    if len(pred_poses) != len(gt_poses) or len(pred_poses) < 2:
        raise ValueError("Predicted and GT trajectories must have the same length >= 2")
    def clean_pose(pose):
        pose = np.asarray(pose, dtype=np.float64).copy()
        pose[:3, :3] = project_to_so3(pose[:3, :3])
        pose[3] = (0.0, 0.0, 0.0, 1.0)
        return pose

    pred = PosePath3D(poses_se3=[clean_pose(p) for p in pred_poses])
    gt = PosePath3D(poses_se3=[clean_pose(p) for p in gt_poses])

    pred_sim3_aligned = copy.deepcopy(pred)
    pred_sim3_aligned.align(gt, correct_scale=True)

    # Position-only Umeyama alignment is rotation-ambiguous for short or nearly
    # collinear trajectories. Keep the common first-camera gauge for rotation
    # metrics instead of injecting that unstable alignment rotation into poses.
    pred_origin_aligned = copy.deepcopy(pred)
    pred_origin_aligned.align_origin(gt)

    def run_metric(metric, estimate):
        metric.process_data((gt, estimate))
        return {key: float(value) for key, value in metric.get_all_statistics().items()}

    return {
        "ate_translation_m": run_metric(
            metrics.APE(metrics.PoseRelation.translation_part), pred_sim3_aligned
        ),
        "ape_rotation_deg": run_metric(
            metrics.APE(metrics.PoseRelation.rotation_angle_deg), pred_origin_aligned
        ),
        "rpe_translation_m": run_metric(
            metrics.RPE(
                metrics.PoseRelation.translation_part,
                delta=1,
                delta_unit=metrics.Unit.frames,
                rel_delta_tol=0.01,
                all_pairs=True,
            ),
            pred_sim3_aligned,
        ),
        "rpe_rotation_deg": run_metric(
            metrics.RPE(
                metrics.PoseRelation.rotation_angle_deg,
                delta=1,
                delta_unit=metrics.Unit.frames,
                rel_delta_tol=0.01,
                all_pairs=True,
            ),
            pred_origin_aligned,
        ),
    }


def aggregate_scene_metrics(scene_results):
    metric_paths = (
        ("ate_translation_m", "rmse"),
        ("ape_rotation_deg", "rmse"),
        ("rpe_translation_m", "rmse"),
        ("rpe_rotation_deg", "rmse"),
    )
    aggregate = {}
    valid_scenes = [item for item in scene_results if item.get("metrics")]
    for section, stat in metric_paths:
        values = [item["metrics"][section][stat] for item in valid_scenes]
        aggregate[f"{section}_{stat}_scene_mean"] = (
            None if not values else float(np.mean(values))
        )
        aggregate[f"{section}_{stat}_scene_median"] = (
            None if not values else float(np.median(values))
        )
    aggregate["num_scenes"] = len(valid_scenes)
    aggregate["num_frames"] = int(sum(item.get("num_frames", 0) for item in valid_scenes))
    requested_edges = sum(item.get("num_requested_edges", 0) for item in scene_results)
    successful_edges = sum(item.get("num_successful_edges", 0) for item in scene_results)
    aggregate["num_requested_edges"] = int(requested_edges)
    aggregate["num_successful_edges"] = int(successful_edges)
    aggregate["pose_recovery_success_rate"] = float(
        successful_edges / max(requested_edges, 1)
    )
    for key in (
        "pose_inlier_ratio_mean",
        "geometric_residual_mean",
        "registration_residual_mean_all_edges",
        "pair_rotation_error_deg_mean",
        "estimated_focal_px_mean",
    ):
        values = [item[key] for item in scene_results if item.get(key) is not None]
        aggregate[key] = None if not values else float(np.mean(values))
    return aggregate


def resolve_include_scenes(values):
    if not values:
        return set()
    return {
        name.strip()
        for value in values
        for name in str(value).split(",")
        if name.strip()
    }


def evaluate(args):
    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg["enabled"] = False
    cfg["lowlight_aug"] = lowlight_cfg
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg["exclude_scenes"] = []
    cfg["dataset"] = dataset_cfg
    train_args = make_train_args(args)
    include_scenes = resolve_include_scenes(args.include_scenes)
    if include_scenes:
        # Scene discovery is lexical, so an exact scene filter must not be
        # constrained by a prior max-scenes truncation.
        train_args.max_scenes = None
    dataset = build_dataset(train_args, cfg)
    if include_scenes:
        dataset.samples = [
            sample for sample in dataset.samples if sample["scene"] in include_scenes
        ]
        found_scenes = {sample["scene"] for sample in dataset.samples}
        missing_scenes = sorted(include_scenes - found_scenes)
        if missing_scenes:
            raise RuntimeError(
                "Requested trajectory scenes are missing or contain no valid pairs: "
                + ", ".join(missing_scenes)
            )
    if not dataset.has_relative_pose:
        raise RuntimeError(
            f"ATE/RPE requires GT poses, but no poses_selected.npz/transforms.json "
            f"was found under {args.data_root}"
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )
    device = resolve_device(args.device)
    model_cfg = cfg if args.variant == "event_dark3r" else cfg_without_lora(cfg)
    model, load_info = build_model(args, model_cfg, train_args, device)
    if args.pose_head_checkpoint is not None:
        if args.variant != "event_dark3r":
            raise RuntimeError(
                "--pose-head-checkpoint is supported only for --variant event_dark3r"
            )
        base_missing = list(load_info.get("missing_keys", []))
        base_unexpected = list(load_info.get("unexpected_keys", []))
        non_pose_missing = [
            key for key in base_missing if not key.startswith("pose_head.")
        ]
        if non_pose_missing or base_unexpected:
            raise RuntimeError(
                "Base Night3R checkpoint is incomplete for pose-head composition: "
                f"non_pose_missing={non_pose_missing} unexpected={base_unexpected}"
            )
        pose_head_load_info = load_pose_head_checkpoint(
            model,
            args.pose_head_checkpoint,
            allow_unsafe=args.allow_unsafe_checkpoint_load,
        )
        load_info = dict(load_info)
        load_info["base_expected_pose_head_missing_keys"] = base_missing
        load_info["missing_keys"] = []
        load_info["pose_head_checkpoint"] = pose_head_load_info
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)

    scene_edges = defaultdict(list)
    lowlight_diagnostics = []
    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(loader):
            scene = _batch_string(batch_cpu, "scene")
            dataset_source = _batch_string(batch_cpu, "dataset_source").lower()
            idx1 = _batch_scalar(batch_cpu, "idx1")
            idx2 = _batch_scalar(batch_cpu, "idx2")
            if idx2 >= args.max_frames_per_scene:
                continue
            batch_cpu, input_diagnostics = apply_deterministic_lowlight(batch_cpu, args)
            if input_diagnostics is not None:
                lowlight_diagnostics.append(input_diagnostics)
            batch = move_batch_to_device(batch_cpu, device)
            if args.variant == "event_dark3r":
                batch = apply_event_mode(batch, args.event_mode, batch_idx=batch_idx)
                if args.pose_recovery == "direct":
                    batch["pose_only_inference"] = True
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                if args.variant == "event_dark3r":
                    pred1, pred2 = model(batch)
                    if args.pose_recovery == "pointmap_sim3":
                        pred1_reverse, pred2_reverse = model(swap_pair_batch(batch))
                else:
                    from model.event_dark3r_student import make_view_pair_from_batch

                    view1, view2 = make_view_pair_from_batch(batch, prefer_dark=True)
                    pred1, pred2 = model(view1, view2)
                    if args.pose_recovery == "pointmap_sim3":
                        reverse1, reverse2 = make_view_pair_from_batch(
                            swap_pair_batch(batch), prefer_dark=True
                        )
                        pred1_reverse, pred2_reverse = model(reverse1, reverse2)

            if args.pose_recovery == "direct":
                relative_pose = pred1.get("relative_pose")
                if relative_pose is None:
                    raise RuntimeError(
                        "Direct pose recovery requires pred1['relative_pose']; use a "
                        "checkpoint and config with pose_head.enabled=true"
                    )
                pred_rel = _as_numpy(relative_pose)[0].astype(np.float64)
                pred_rel[:3, :3] = project_to_so3(pred_rel[:3, :3])
                diagnostics = {
                    "success": True,
                    "inlier_ratio": 1.0,
                    "geometric_residual": None,
                    "intrinsics_source": "not_used",
                    "pose_source": "direct_relative_pose_head",
                }
                if not np.isfinite(pred_rel).all():
                    pred_rel = None
                    diagnostics["success"] = False
                    diagnostics["failure"] = "non-finite direct pose prediction"
            elif args.pose_recovery == "pointmap_sim3":
                candidates = (
                    (
                        "view2",
                        pred2.get("pts3d_in_other_view"),
                        pred1_reverse.get("pts3d"),
                        pred2.get("conf"),
                        pred1_reverse.get("conf"),
                    ),
                    (
                        "view1",
                        pred1.get("pts3d"),
                        pred2_reverse.get("pts3d_in_other_view"),
                        pred1.get("conf"),
                        pred2_reverse.get("conf"),
                    ),
                )
                if any(value is None for candidate in candidates for value in candidate[1:]):
                    raise RuntimeError("Point-map Sim(3) recovery requires points and confidence")
                registrations = []
                for direction, source_points, target_points, source_conf, target_conf in candidates:
                    transform_i, diagnostics_i = estimate_relative_pose_pointmap_sim3(
                        source_points,
                        target_points,
                        source_conf,
                        target_conf,
                        conf_quantile=args.conf_quantile,
                        max_points=args.max_pnp_points,
                        min_inliers=max(args.min_pnp_inliers, 512),
                        normalize_translation=True,
                        max_normalized_rmse=args.max_pointmap_residual,
                        seed=args.seed + batch_idx,
                    )
                    diagnostics_i["registration_direction"] = direction
                    registrations.append((transform_i, diagnostics_i))
                pred_rel, diagnostics = select_best_pointmap_registration(registrations)
                diagnostics["candidate_residuals"] = {
                    item[1]["registration_direction"]: item[1].get("geometric_residual")
                    for item in registrations
                }
                diagnostics["intrinsics_source"] = "not_used"
            elif args.pose_recovery == "essential":
                descriptor1 = pred1.get("desc")
                descriptor2 = pred2.get("desc")
                if descriptor1 is None or descriptor2 is None:
                    raise RuntimeError("Essential pose recovery requires MASt3R descriptors")
                matches1, matches2, match_scores = reciprocal_descriptor_matches(
                    descriptor1,
                    descriptor2,
                    pred1.get("desc_conf"),
                    pred2.get("desc_conf"),
                    stride=args.match_stride,
                    max_matches=args.max_matches,
                )
                intrinsics = _as_numpy(batch["intrinsics"]).reshape(3, 3)
                pred_rel, diagnostics = estimate_relative_pose_essential(
                    matches1,
                    matches2,
                    intrinsics,
                    reprojection_error=args.essential_reproj_error,
                    iterations=args.essential_iterations,
                    min_inliers=args.min_essential_inliers,
                    seed=args.seed + batch_idx,
                )
                diagnostics["match_score_mean"] = (
                    None if match_scores.size == 0 else float(np.mean(match_scores))
                )
                diagnostics["intrinsics_source"] = "gt_calibration"
            else:
                points = pred2.get("pts3d_in_other_view")
                confidence = pred2.get("conf")
                if points is None or confidence is None:
                    raise RuntimeError("PnP requires pts3d_in_other_view and conf")
                focal_candidates = None
                if args.pnp_intrinsics == "search":
                    point_shape = tuple(points.shape)
                    if len(point_shape) != 4:
                        raise ValueError(f"Expected batched point map, got {point_shape}")
                    image_extent = max(point_shape[1], point_shape[2])
                    focal_candidates = np.geomspace(
                        image_extent / 2.0, image_extent * 3.0, 21
                    )
                    intrinsics = None
                    focal = None
                elif args.pnp_intrinsics == "predicted":
                    points1 = pred1.get("pts3d")
                    if points1 is None:
                        raise RuntimeError("Model output must contain pred1['pts3d']")
                    intrinsics, focal = estimate_intrinsics_from_pointmap(points1)
                else:
                    intrinsics = _as_numpy(batch["intrinsics"]).reshape(3, 3)
                    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) * 0.5)
                pred_rel, diagnostics = estimate_relative_pose_pnp(
                    points,
                    confidence,
                    intrinsics,
                    conf_quantile=args.conf_quantile,
                    max_points=args.max_pnp_points,
                    reprojection_error=args.pnp_reproj_error,
                    iterations=args.pnp_iterations,
                    min_inliers=args.min_pnp_inliers,
                    seed=args.seed + batch_idx,
                    focal_candidates=focal_candidates,
                )
                if focal is not None:
                    diagnostics.setdefault("focal_px", focal)
                diagnostics["intrinsics_source"] = args.pnp_intrinsics
            gt_rel = _as_numpy(batch["relative_pose_gt"])[0]
            if (
                dataset_source == "dl3dv"
                and getattr(dataset, "pose_output_convention", "native") != "opencv"
            ):
                gt_rel = convert_dl3dv_relative_pose_to_opencv(gt_rel)
            if pred_rel is not None:
                diagnostics["relative_rotation_error_deg"] = _rotation_error_deg(
                    pred_rel[:3, :3], gt_rel[:3, :3]
                )
                diagnostics["relative_translation_direction_cosine"] = float(
                    np.dot(pred_rel[:3, 3], gt_rel[:3, 3])
                    / (
                        np.linalg.norm(pred_rel[:3, 3])
                        * np.linalg.norm(gt_rel[:3, 3])
                        + 1e-12
                    )
                )
            scene_edges[scene].append(
                {
                    "idx1": idx1,
                    "idx2": idx2,
                    "pred_relative": pred_rel,
                    "gt_relative": gt_rel,
                    "pose_recovery": diagnostics,
                }
            )

    scene_results = []
    for scene, edges in sorted(scene_edges.items()):
        edges = sorted(edges, key=lambda item: (item["idx1"], item["idx2"]))
        valid_runs = []
        current_run = []
        for edge in edges:
            is_contiguous = (
                edge["pred_relative"] is not None
                and edge["idx2"] == edge["idx1"] + 1
                and (
                    not current_run
                    or edge["idx1"] == current_run[-1]["idx2"]
                )
            )
            if is_contiguous:
                current_run.append(edge)
            else:
                if current_run:
                    valid_runs.append(current_run)
                current_run = (
                    [edge]
                    if edge["pred_relative"] is not None
                    and edge["idx2"] == edge["idx1"] + 1
                    else []
                )
        if current_run:
            valid_runs.append(current_run)
        longest_run = max(valid_runs, key=len, default=[])

        pred_poses = [np.eye(4, dtype=np.float64)]
        gt_poses = [np.eye(4, dtype=np.float64)]
        for edge in longest_run:
            pred_poses.append(pred_poses[-1] @ np.linalg.inv(edge["pred_relative"]))
            gt_poses.append(gt_poses[-1] @ np.linalg.inv(edge["gt_relative"]))
        successful_rows = [
            edge["pose_recovery"] for edge in edges if edge["pred_relative"] is not None
        ]
        recovery_rows = [edge["pose_recovery"] for edge in longest_run]
        residuals = [
            row["geometric_residual"]
            for row in successful_rows
            if row.get("geometric_residual") is not None
        ]
        attempted_residuals = [
            edge["pose_recovery"]["geometric_residual"]
            for edge in edges
            if edge["pose_recovery"].get("geometric_residual") is not None
        ]
        focals = [
            row["focal_px"] for row in successful_rows if row.get("focal_px") is not None
        ]
        metrics_out = trajectory_metrics(pred_poses, gt_poses) if len(pred_poses) >= 3 else None
        scene_result = {
                "scene": scene,
                "num_requested_edges": len(edges),
                "num_successful_edges": len(successful_rows),
                "num_edges_in_evaluated_trajectory": len(recovery_rows),
                "trajectory_start_frame": (
                    None if not longest_run else longest_run[0]["idx1"]
                ),
                "num_frames": len(pred_poses),
                "pose_recovery_success_rate": float(
                    len(successful_rows) / max(len(edges), 1)
                ),
                "pose_recovery_failures": [
                    {
                        "idx1": edge["idx1"],
                        "idx2": edge["idx2"],
                        **edge["pose_recovery"],
                    }
                    for edge in edges
                    if edge["pred_relative"] is None
                ],
                "pose_inlier_ratio_mean": (
                    None
                    if not successful_rows
                    else float(np.mean([row["inlier_ratio"] for row in successful_rows]))
                ),
                "geometric_residual_mean": (
                    None if not residuals else float(np.mean(residuals))
                ),
                "registration_residual_mean_all_edges": (
                    None
                    if not attempted_residuals
                    else float(np.mean(attempted_residuals))
                ),
                "pair_rotation_error_deg_mean": (
                    None
                    if not successful_rows
                    else float(
                        np.mean(
                            [row["relative_rotation_error_deg"] for row in successful_rows]
                        )
                    )
                ),
                "estimated_focal_px_mean": (
                    None if not focals else float(np.mean(focals))
                ),
                "metrics": metrics_out,
        }
        if args.export_trajectories and longest_run:
            scene_result["trajectory"] = {
                "frame_indices": [
                    int(longest_run[0]["idx1"]),
                    *[int(edge["idx2"]) for edge in longest_run],
                ],
                "pred_camera_to_world": [pose.tolist() for pose in pred_poses],
                "gt_camera_to_world": [pose.tolist() for pose in gt_poses],
            }
        scene_results.append(scene_result)

    result = {
        "protocol": {
            "trajectory_source": (
                "direct relative-pose regression from frozen cross-view decoder tokens"
                if args.pose_recovery == "direct"
                else (
                    "bidirectional dense point maps registered by trimmed Sim(3)"
                    if args.pose_recovery == "pointmap_sim3"
                    else (
                        "reciprocal MASt3R descriptor matches plus calibrated essential matrix"
                        if args.pose_recovery == "essential"
                        else "confidence-filtered dense PnP on pts3d_in_other_view"
                    )
                )
            ),
            "pose_recovery": args.pose_recovery,
            "pose_convention": "relative T_camera_j_from_camera_i; trajectories stored camera-to-world",
            "gt_axis_conversion": (
                "dataset pose_output_convention=opencv converts DL3DV by diag(1,-1,-1)"
                if getattr(dataset, "pose_output_convention", "native") == "opencv"
                else "DL3DV OpenGL camera axes converted to OpenCV by diag(1,-1,-1)"
            ),
            "translation_alignment": "per-scene Sim(3) alignment via EVO",
            "rotation_alignment": (
                "first-pose origin alignment; position-only Umeyama rotation is not "
                "used because it is underconstrained for near-collinear trajectories"
            ),
            "pair_stride": 1,
            "max_frames_per_scene": args.max_frames_per_scene,
            "include_scenes": sorted(include_scenes) or None,
            "trajectories_exported": bool(args.export_trajectories),
            "conf_quantile": args.conf_quantile,
            "max_pointmap_residual": (
                args.max_pointmap_residual
                if args.pose_recovery == "pointmap_sim3"
                else None
            ),
            "max_pnp_points": args.max_pnp_points,
            "pnp_reproj_error_px": args.pnp_reproj_error,
            "pnp_iterations": args.pnp_iterations,
            "min_pnp_inliers": args.min_pnp_inliers,
            "pnp_intrinsics": args.pnp_intrinsics,
            "focal_search_range": (
                None
                if args.pnp_intrinsics != "search"
                else "21 geometric candidates from max(H,W)/2 to 3*max(H,W)"
            ),
            "pnp_solver": "SQPnP RANSAC; LM refinement retained only when it lowers inlier RMSE",
            "match_stride": args.match_stride,
            "max_matches": args.max_matches,
            "essential_reproj_error_px": args.essential_reproj_error,
            "essential_iterations": args.essential_iterations,
            "min_essential_inliers": args.min_essential_inliers,
            "monocular_scale": (
                "directly regressed relative translation, followed by one per-scene Sim(3) alignment"
                if args.pose_recovery == "direct"
                else (
                    "unit translation per adjacent edge, followed by one per-scene Sim(3) alignment"
                    if args.pose_recovery in {"pointmap_sim3", "essential"}
                    else "predicted point-map scale, followed by one per-scene Sim(3) alignment"
                )
            ),
            "failed_edge_policy": "evaluate the longest contiguous successful edge run per scene",
            "rgb_input": {
                "mode": (
                    "stored_rgb"
                    if args.lowlight_exposure is None
                    else "deterministic_fixed_exposure_linear_domain"
                ),
                "exposure_scale": args.lowlight_exposure,
                "gamma_linearization": args.lowlight_gamma,
                "shot_noise_std": args.lowlight_shot_noise_std,
                "read_noise_std": args.lowlight_read_noise_std,
                "sample_keyed_noise_seed": (
                    None if args.lowlight_exposure is None else args.lowlight_seed
                ),
                "diagnostics": aggregate_lowlight_diagnostics(lowlight_diagnostics),
            },
        },
        "variant": args.variant,
        "event_mode": args.event_mode if args.variant == "event_dark3r" else None,
        "checkpoint": load_info,
        "config": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
        "aggregate": aggregate_scene_metrics(scene_results),
        "scenes": scene_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
