#!/usr/bin/env python3
"""Evaluate VGGT-family camera predictions on the Night3R trajectory protocol."""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import build_dataset, load_config  # noqa: E402
from utils.eval_event_dark3r_variant import apply_event_mode, resolve_device  # noqa: E402
from utils.eval_night3r_trajectory import (  # noqa: E402
    _as_numpy,
    _batch_scalar,
    _batch_string,
    _rotation_error_deg,
    aggregate_lowlight_diagnostics,
    aggregate_scene_metrics,
    apply_deterministic_lowlight,
    convert_dl3dv_relative_pose_to_opencv,
    project_to_so3,
    trajectory_metrics,
)
from utils.eval_vggt_baseline import (  # noqa: E402
    event_voxel_to_red_blue,
    preprocess_vggt,
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate VGGT camera poses on Night3R data")
    parser.add_argument("--variant", choices=("vggt", "eventvggt"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-vggt-checkpoint", default="/data/fcr/ckpt/model.pt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vggt-root", default="/data/fcr/vggt")
    parser.add_argument("--eventvggt-root", default="/data/fcr/EventVGGT")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--event-size", type=int, nargs=2, default=None)
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--max-frames-per-scene", type=int, default=32)
    parser.add_argument(
        "--export-trajectories",
        action="store_true",
        help="Store composed predicted and GT camera-to-world trajectories.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--event-mode", choices=("real", "zero", "shuffle"), default="real")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lowlight-exposure", type=float, default=None)
    parser.add_argument("--lowlight-read-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-gamma", type=float, default=2.2)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    args = parser.parse_args()
    if args.lowlight_exposure is not None and not 0.0 < args.lowlight_exposure <= 1.0:
        parser.error("--lowlight-exposure must be in (0, 1]")
    return args


def _load_state(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("model", payload) if isinstance(payload, dict) else payload


def load_vggt_camera_model(args, device):
    sys.path.insert(0, str(Path(args.vggt_root).resolve()))
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_camera=True,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    state = _load_state(args.checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    critical_missing = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("aggregator.", "camera_head."))
    ]
    if critical_missing:
        raise RuntimeError(f"VGGT camera model is missing {len(critical_missing)} required keys")
    return model.eval().to(device), {
        "implementation": str(Path(args.vggt_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "camera_head_source": "official VGGT checkpoint",
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
    }


def load_eventvggt_camera_model(args, device):
    event_root = str(Path(args.eventvggt_root).resolve())
    sys.path.insert(0, event_root)
    from vggt.models.vggt_lora_dino import (
        create_vggt_with_lora_dino,
        create_vggt_with_split_qkv,
    )

    base = create_vggt_with_split_qkv(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
    )
    base_state = _load_state(args.base_vggt_checkpoint)
    converted_state = {}
    for name, value in base_state.items():
        if name.startswith("aggregator.") and name.endswith(".qkv.weight"):
            prefix = name[: -len("qkv.weight")]
            q_value, k_value, v_value = value.chunk(3, dim=0)
            converted_state[f"{prefix}q_proj.weight"] = q_value
            converted_state[f"{prefix}k_proj.weight"] = k_value
            converted_state[f"{prefix}v_proj.weight"] = v_value
        elif name.startswith("aggregator.") and name.endswith(".qkv.bias"):
            prefix = name[: -len("qkv.bias")]
            q_value, k_value, v_value = value.chunk(3, dim=0)
            converted_state[f"{prefix}q_proj.bias"] = q_value
            converted_state[f"{prefix}k_proj.bias"] = k_value
            converted_state[f"{prefix}v_proj.bias"] = v_value
        else:
            converted_state[name] = value
    base_incompatible = base.load_state_dict(converted_state, strict=False)
    critical_base_missing = [
        key
        for key in base_incompatible.missing_keys
        if key.startswith(("aggregator.", "camera_head."))
    ]
    if critical_base_missing:
        raise RuntimeError(
            "EventVGGT base initialization is missing required aggregator/camera "
            f"weights: {critical_base_missing[:8]}"
        )
    model = create_vggt_with_lora_dino(
        base,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    event_state = _load_state(args.checkpoint)
    incompatible = model.load_state_dict(event_state, strict=False)
    return model.eval().to(device), {
        "implementation": event_root,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_vggt_checkpoint": str(Path(args.base_vggt_checkpoint).resolve()),
        "camera_head_source": "official VGGT camera head; EventVGGT checkpoint has no camera-head parameters",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
    }


def relative_pose_from_extrinsics(extrinsics):
    """Convert pairwise OpenCV world-to-camera extrinsics to T_camera2_from_camera1."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape != (2, 3, 4):
        raise ValueError(f"Expected pair extrinsics [2,3,4], got {extrinsics.shape}")
    homogeneous = np.broadcast_to(np.eye(4, dtype=np.float64), (2, 4, 4)).copy()
    homogeneous[:, :3] = extrinsics
    relative = homogeneous[1] @ np.linalg.inv(homogeneous[0])
    relative[:3, :3] = project_to_so3(relative[:3, :3])
    relative[3] = (0.0, 0.0, 0.0, 1.0)
    return relative


def build_eval_dataset(args):
    cfg = dict(load_config(args.config))
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg["enabled"] = False
    cfg["lowlight_aug"] = lowlight_cfg
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg["exclude_scenes"] = []
    cfg["dataset"] = dataset_cfg
    dataset_args = SimpleNamespace(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        pair_stride=1,
        max_scenes=args.max_scenes,
        max_pairs_per_scene=max(args.max_frames_per_scene - 1, 1),
        pair_sampling="fixed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
    )
    dataset = build_dataset(dataset_args, cfg)
    if not dataset.has_relative_pose:
        raise RuntimeError(f"Trajectory GT is unavailable under {args.data_root}")
    return dataset


def scene_results_from_edges(scene_edges, export_trajectories=False):
    scene_results = []
    for scene, edges in sorted(scene_edges.items()):
        edges = sorted(edges, key=lambda item: (item["idx1"], item["idx2"]))
        valid_runs = []
        current_run = []
        for edge in edges:
            contiguous = (
                edge["pred_relative"] is not None
                and edge["idx2"] == edge["idx1"] + 1
                and (not current_run or edge["idx1"] == current_run[-1]["idx2"])
            )
            if contiguous:
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
        successful = [edge for edge in edges if edge["pred_relative"] is not None]
        pair_errors = [
            edge["pose_recovery"]["relative_rotation_error_deg"] for edge in successful
        ]
        result = {
                "scene": scene,
                "num_requested_edges": len(edges),
                "num_successful_edges": len(successful),
                "num_edges_in_evaluated_trajectory": len(longest_run),
                "trajectory_start_frame": None if not longest_run else longest_run[0]["idx1"],
                "num_frames": len(pred_poses),
                "pose_recovery_success_rate": len(successful) / max(len(edges), 1),
                "pose_recovery_failures": [
                    {
                        "idx1": edge["idx1"],
                        "idx2": edge["idx2"],
                        **edge["pose_recovery"],
                    }
                    for edge in edges
                    if edge["pred_relative"] is None
                ],
                "pose_inlier_ratio_mean": 1.0 if successful else None,
                "geometric_residual_mean": None,
                "registration_residual_mean_all_edges": None,
                "pair_rotation_error_deg_mean": (
                    None if not pair_errors else float(np.mean(pair_errors))
                ),
                "estimated_focal_px_mean": None,
                "metrics": (
                    trajectory_metrics(pred_poses, gt_poses)
                    if len(pred_poses) >= 3
                    else None
                ),
            }
        if export_trajectories and longest_run:
            result["trajectory"] = {
                "frame_indices": [longest_run[0]["idx1"]]
                + [edge["idx2"] for edge in longest_run],
                "pred_camera_to_world": [pose.tolist() for pose in pred_poses],
                "gt_camera_to_world": [pose.tolist() for pose in gt_poses],
            }
        scene_results.append(result)
    return scene_results


def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset = build_eval_dataset(args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.variant == "vggt":
        model, load_info = load_vggt_camera_model(args, device)
    else:
        model, load_info = load_eventvggt_camera_model(args, device)
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    event_size = args.event_size
    if event_size is None:
        event_size = (252, 336) if "mvsec" in args.data_root.lower() else (252, 504)

    scene_edges = defaultdict(list)
    lowlight_diagnostics = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            idx1 = _batch_scalar(batch, "idx1")
            idx2 = _batch_scalar(batch, "idx2")
            if idx2 >= args.max_frames_per_scene:
                continue
            scene = _batch_string(batch, "scene")
            dataset_source = _batch_string(batch, "dataset_source").lower()
            batch, input_diagnostics = apply_deterministic_lowlight(batch, args)
            if input_diagnostics is not None:
                lowlight_diagnostics.append(input_diagnostics)

            if args.variant == "vggt":
                model_input = preprocess_vggt(batch["images_photo_dark"].to(device))
            else:
                controlled = apply_event_mode(batch, args.event_mode, batch_idx=batch_idx)
                model_input = event_voxel_to_red_blue(
                    controlled["event_voxel"].to(device), event_size
                )

            diagnostics = {
                "success": True,
                "inlier_ratio": 1.0,
                "geometric_residual": None,
                "intrinsics_source": "VGGT camera head",
                "pose_source": f"{args.variant}_pairwise_camera_head",
            }
            with torch.amp.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=bool(args.amp and device.type == "cuda"),
            ):
                prediction = model(images=model_input)
            pose_encoding = prediction.get("pose_enc")
            if pose_encoding is None:
                raise RuntimeError(f"{args.variant} did not return pose_enc")
            extrinsics, _ = pose_encoding_to_extri_intri(
                pose_encoding.float(), build_intrinsics=False
            )
            pred_rel = relative_pose_from_extrinsics(
                extrinsics[0].detach().cpu().numpy()
            )
            if not np.isfinite(pred_rel).all():
                pred_rel = None
                diagnostics.update(success=False, failure="non-finite camera prediction")

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
                        + 1.0e-12
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

    scene_results = scene_results_from_edges(
        scene_edges, export_trajectories=args.export_trajectories
    )
    result = {
        "protocol": {
            "trajectory_source": "pairwise VGGT camera-head prediction",
            "pose_recovery": "direct_camera_head",
            "pose_convention": "relative T_camera_j_from_camera_i; trajectories stored camera-to-world",
            "translation_alignment": "per-scene Sim(3) alignment via EVO",
            "rotation_alignment": "first-pose origin alignment",
            "pair_stride": 1,
            "max_frames_per_scene": args.max_frames_per_scene,
            "rgb_input": {
                "mode": (
                    "not_used_event_only"
                    if args.variant == "eventvggt"
                    else (
                        "stored_rgb"
                        if args.lowlight_exposure is None
                        else "deterministic_fixed_exposure_linear_domain"
                    )
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
            "event_input": (
                None
                if args.variant == "vggt"
                else {
                    "mode": args.event_mode,
                    "adapter": "split-polarity voxel to EventVGGT red/blue occupancy image",
                    "size": list(event_size),
                }
            ),
        },
        "variant": args.variant,
        "checkpoint": load_info,
        "config": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "amp": bool(args.amp and device.type == "cuda"),
        "amp_dtype": args.amp_dtype if args.amp and device.type == "cuda" else None,
        "aggregate": aggregate_scene_metrics(scene_results),
        "scenes": scene_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate(parse_args())
