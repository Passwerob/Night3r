#!/usr/bin/env python3
"""
Evaluate depth maps from Dark3R reconstruction against ground truth.

Scale from poses (Methods 1/2/3), rescale test depths with Umeyama scale,
then compute AbsRel and δ<1.25 on the test depth set.

- Scale is computed using ALL common frames between the two transforms.json files,
  sorted by filename stem (basename without extension).
- Metrics are computed ONLY on the depth maps found in test_depth_dir (by stem).
- Results saved to parent of test_depth_dir as eval_depth.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d


def filter_gt_by_target(gt, target, subsample=1):
    """Filter GT frames to match target frames by filename stem."""
    target_filenames = set(
        [os.path.basename(x["file_path"]).split(".")[0] for x in target["frames"]]
    )
    gt_filtered = [
        x
        for x in gt["frames"]
        if os.path.basename(x["file_path"]).split(".")[0] in target_filenames
    ]
    gt["frames"] = gt_filtered[::subsample]
    return gt


def read_json(json_path):
    """Read JSON file."""
    with open(json_path) as f:
        return json.load(f)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate depth maps from Dark3R reconstruction"
    )
    ap.add_argument(
        "--gt_transforms",
        "-gt",
        default=None,
        help="Path to ground truth transforms.json (default: gt_depth_dir/../transforms.json)",
    )
    ap.add_argument(
        "--test_transforms",
        "-tt",
        default=None,
        help="Path to test transforms.json (default: test_depth_dir/../transforms.json)",
    )
    ap.add_argument(
        "--gt_depth_dir",
        "-g",
        required=True,
        help="Directory containing ground truth depth .npy files",
    )
    ap.add_argument(
        "--test_depth_dir",
        "-t",
        required=True,
        help="Directory containing test/predicted depth .npy files",
    )
    ap.add_argument(
        "--max_pairs",
        "-mp",
        type=int,
        default=200000,
        help="Max pairwise samples for Method 2 (default: 200000)",
    )
    return ap.parse_args()


def list_depth_stems(depth_dir: Path):
    """List all .npy file stems in a directory."""
    return sorted([p.stem for p in Path(depth_dir).glob("*.npy")])


def reshape4x4(M):
    """Reshape a flat list or array into a 4x4 matrix."""
    if isinstance(M, list) and len(M) == 16:
        M = [M[i * 4 : (i + 1) * 4] for i in range(4)]
    M = np.asarray(M, dtype=float)
    assert M.shape == (4, 4), f"Expected 4x4, got {M.shape}"
    return M


def get_matrix_from_frame(frame):
    """Extract 4x4 transform matrix from a frame dictionary."""
    for k in ("transform_matrix", "transform", "pose"):
        if k in frame:
            return reshape4x4(frame[k])
    raise KeyError(
        "No transform matrix found in frame (looked for 'transform_matrix', 'transform', 'pose')."
    )


def center_from_matrix(M, mode):
    """Extract camera center from a transform matrix."""
    R = M[:3, :3]
    t = M[:3, 3]
    if mode == "c2w":
        return t
    elif mode == "w2c":
        return -(R.T @ t)
    else:
        raise ValueError("mode must be 'c2w' or 'w2c'")


def centers_from_frames(frames, mode):
    """Extract camera centers from a list of frames."""
    Cs = []
    for fr in frames:
        M = get_matrix_from_frame(fr)
        Cs.append(center_from_matrix(M, mode))
    return np.stack(Cs, axis=0)  # (N,3)


def umeyama_o3d_scale_and_rms(CA, CB):
    """
    Open3D closed-form Sim(3) with scaling. Computes TEST->GT mapping:
        CB ≈ s * R * CA + t
    Returns (s, T(4x4), rms).
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(CA))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(CB))
    corr = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(len(CA)), np.arange(len(CB))]).astype(np.int32)
    )
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    T = est.compute_transformation(src, dst, corr)  # 4x4; upper-left is sR
    SR = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(SR)))  # det(SR) ~ s^3
    if s < 0:
        s = -s
        SR = -SR
        T[:3, :3] = SR
    CA_aligned = (SR @ CA.T).T + T[:3, 3]
    rms = float(np.sqrt(np.mean(np.sum((CB - CA_aligned) ** 2, axis=1))))
    return s, T, rms


def choose_best_mode(frames_test, frames_gt):
    """
    Try (test_mode, gt_mode) in {c2w,w2c}^2 and pick smallest Umeyama RMS.
    Returns dict with CA, CB, s, T, rms, mTest, mGT.
    """
    best = None
    for mT in ("c2w", "w2c"):
        CT = centers_from_frames(frames_test, mT)
        for mG in ("c2w", "w2c"):
            CG = centers_from_frames(frames_gt, mG)
            s, T, rms = umeyama_o3d_scale_and_rms(CT, CG)  # TEST -> GT
            if (best is None) or (rms < best["rms"]):
                best = dict(mTest=mT, mGT=mG, CA=CT, CB=CG, s=s, T=T, rms=rms)
    return best


def method1_path_ratio(CA, CB):
    """Total consecutive path-length ratio: sum||ΔCB|| / sum||ΔCA|| (+ step median)."""
    if len(CA) < 2 or len(CB) < 2:
        return float("nan"), float("nan"), 0.0, 0.0
    da = np.linalg.norm(CA[1:] - CA[:-1], axis=1)
    db = np.linalg.norm(CB[1:] - CB[:-1], axis=1)
    total_a = float(np.sum(da))
    total_b = float(np.sum(db))
    ratio = (total_b / total_a) if total_a > 1e-12 else float("nan")
    good = da > 1e-12
    step_med = float(np.median(db[good] / da[good])) if np.any(good) else float("nan")
    return ratio, step_med, total_a, total_b


def method2_pairwise_median(CA, CB, max_pairs=200000):
    """Median of pairwise distance ratios; subsample if large."""
    N = len(CA)
    if N < 2:
        return float("nan"), 0
    total_pairs = N * (N - 1) // 2
    if total_pairs <= max_pairs:
        i, j = np.triu_indices(N, k=1)
    else:
        rng = np.random.default_rng(0)
        # ensure j > i
        i = rng.integers(0, N - 1, size=max_pairs)  # upper bound exclusive -> 0..N-2
        j = i + 1 + rng.integers(0, N - 1 - i)
    da = np.linalg.norm(CA[i] - CA[j], axis=1)
    db = np.linalg.norm(CB[i] - CB[j], axis=1)
    good = da > 1e-12
    ratios = db[good] / da[good]
    return float(np.median(ratios)), int(np.sum(good))


def load_depth(path: Path):
    """Load a depth map from a .npy file."""
    arr = np.load(path)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def compute_absrel_and_delta(pred, gt, eps=1e-8):
    """
    pred, gt: same shape; invalid if <=0 or nonfinite.
    Returns dict with abs_rel, delta_125, valid_px.
    """
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > eps) & (pred > eps)
    if not np.any(mask):
        return dict(abs_rel=float("nan"), delta_125=float("nan"), valid_px=0)
    p = pred[mask]
    g = gt[mask]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    delta_125 = float(1 - np.mean(ratio > 1.25))
    return dict(abs_rel=abs_rel, delta_125=delta_125, valid_px=int(mask.sum()))


def safe_mean(arr):
    """Compute mean ignoring non-finite values."""
    arr = [x for x in arr if np.isfinite(x)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def safe_median(arr):
    """Compute median ignoring non-finite values."""
    arr = [x for x in arr if np.isfinite(x)]
    return float(np.median(arr)) if len(arr) else float("nan")


def main():
    args = parse_args()

    if args.gt_transforms is None:
        args.gt_transforms = Path(args.gt_depth_dir).parent / "transforms.json"
    if args.test_transforms is None:
        args.test_transforms = Path(args.test_depth_dir).parent / "transforms.json"

    # --- Load transforms
    gt_tf = read_json(args.gt_transforms)
    test_tf = read_json(args.test_transforms)

    def stem_of_frame(fr):
        return os.path.basename(fr["file_path"]).split(".")[0]

    gt_frames_map = {stem_of_frame(fr): fr for fr in gt_tf.get("frames", [])}
    test_frames_map = {stem_of_frame(fr): fr for fr in test_tf.get("frames", [])}

    # --- SCALE SET: use ALL common transform frames, sorted by stem
    scale_stems = sorted(set(gt_frames_map.keys()) & set(test_frames_map.keys()))
    if not scale_stems:
        print("No overlapping frames between GT and TEST transforms.", file=sys.stderr)
        sys.exit(2)
    scale_frames_test = [test_frames_map[s] for s in scale_stems]
    scale_frames_gt = [gt_frames_map[s] for s in scale_stems]

    # Compute scales (Methods 1/2/3) on the SCALE SET
    choice = choose_best_mode(scale_frames_test, scale_frames_gt)  # TEST -> GT
    CA, CB = choice["CA"], choice["CB"]
    s_umeyama, T_umeyama, rms = choice["s"], choice["T"], choice["rms"]

    s_path, s_stepmed, total_a, total_b = method1_path_ratio(CA, CB)
    s_pair, used_pairs = method2_pairwise_median(CA, CB, max_pairs=args.max_pairs)

    print(f"[Scale] Matched {len(scale_stems)} frames (sorted by basename).")
    print(f"[Scale] Chosen interpretation: TEST={choice['mTest']}  GT={choice['mGT']}")
    print(f"[Scale] Method 3 (Umeyama/Open3D): scale={s_umeyama:.9f} | RMS={rms:.6e}")
    print(
        f"[Scale] Method 2 (Pairwise med.) : scale={s_pair:.9f}  | pairs used={used_pairs}"
    )
    print(
        f"[Scale] Method 1 (Path-length)   : scale={s_path:.9f}  (step-median={s_stepmed:.9f})  totals A={total_a:.6f} B={total_b:.6f}"
    )

    # --- EVAL SET: only stems present in test_depth_dir (must exist in both transforms)
    test_depth_dir = Path(args.test_depth_dir)
    gt_depth_dir = Path(args.gt_depth_dir)
    eval_stems_all = list_depth_stems(test_depth_dir)
    eval_stems = [
        s for s in eval_stems_all if (s in gt_frames_map) and (s in test_frames_map)
    ]
    if not eval_stems:
        print(
            "No evaluable stems in test_depth_dir that also exist in both transforms.",
            file=sys.stderr,
        )
        sys.exit(3)

    per_image = {}
    absrels, deltas, valid_counts = [], [], []
    skipped = []

    for stem in eval_stems:
        test_path = test_depth_dir / f"{stem}.npy"
        gt_path = gt_depth_dir / f"{stem}.npy"
        if (not test_path.exists()) or (not gt_path.exists()):
            skipped.append(stem)
            continue

        d_test = load_depth(test_path) * s_umeyama  # rescale TEST -> GT scale
        d_gt = load_depth(gt_path)

        if d_test.shape != d_gt.shape:
            # Resize d_test to match d_gt using bilinear interpolation
            d_test_resized = cv2.resize(
                d_test, (d_gt.shape[1], d_gt.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            print(
                f"WARNING: shape mismatch for {stem}: test {d_test.shape} vs gt {d_gt.shape}. Resizing test depth.",
            )
            d_test = d_test_resized

        m = compute_absrel_and_delta(d_test, d_gt)
        per_image[stem] = m
        if np.isfinite(m["abs_rel"]):
            absrels.append(m["abs_rel"])
            deltas.append(m["delta_125"])
            valid_counts.append(m["valid_px"])

    def r(x, d):
        return round(float(x), d) if np.isfinite(x) else float("nan")

    summary = {
        "num_images": len(per_image),
        "abs_rel_mean": r(safe_mean(absrels), 4),
        "abs_rel_median": r(safe_median(absrels), 4),
        "delta_1.25_mean": r(safe_mean(deltas) * 100, 2),
        "delta_1.25_median": r(safe_median(deltas) * 100, 2),
        "total_valid_pixels": int(sum(valid_counts)) if valid_counts else 0,
    }

    scales = {
        "method_1_path_length": s_path,
        "method_1_step_median": s_stepmed,
        "method_2_pairwise_median": s_pair,
        "method_3_umeyama": s_umeyama,
        "umeyama_rms": rms,
        "pose_interpretation": {"test": choice["mTest"], "gt": choice["mGT"]},
        "scale_frame_count": len(scale_stems),
    }

    out_path = test_depth_dir.resolve().parent / "eval_depth.json"
    payload = {
        "args": {
            "gt_transforms": os.path.abspath(args.gt_transforms),
            "test_transforms": os.path.abspath(args.test_transforms),
            "gt_depth_dir": os.path.abspath(args.gt_depth_dir),
            "test_depth_dir": os.path.abspath(args.test_depth_dir),
            "max_pairs": args.max_pairs,
        },
        "scale_stems_count": len(scale_stems),
        "eval_stems_count": len(per_image),
        "skipped_or_missing": skipped,
        "scales": scales,
        "metrics_summary": summary,
        "metrics_per_image": per_image,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
