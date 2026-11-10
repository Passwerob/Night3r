#!/usr/bin/env python3
"""
Transforms a Nerfstudio camera_path.json from one scene's coordinate system
to another's, using their respective transforms.json and
dataparser_transforms.json files to calculate the full alignment.

This script uses the Umeyama alignment (Sim(3)) from Open3D on the
camera centers to find the alignment between the two "world" coordinate systems.
"""

import os
import sys
import json
import math
import argparse
import copy
import numpy as np
from pathlib import Path
import runpy

try:
    import open3d as o3d
except ImportError as e:
    print(
        "ERROR: open3d not installed. Install with: pip install open3d", file=sys.stderr
    )
    sys.exit(1)


# --- Helper Functions from transform_camera_path.py ---


def load_dataparser_transforms(path: Path):
    """
    Load dataparser transforms from either:
      - JSON: { "transform": [[...],[...],[...]], "scale": ... }
      - PY:   a script defining variables `transform` and `scale`
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r") as f:
            data = json.load(f)
        transform = np.array(data["transform"], dtype=float)
        scale = float(data["scale"])
        return transform, scale

    if path.suffix.lower() == ".py":
        ns = runpy.run_path(str(path))
        if "transform" not in ns or "scale" not in ns:
            raise ValueError(
                f"{path} must define `transform` and `scale` variables when using .py"
            )
        transform = np.array(ns["transform"], dtype=float)
        scale = float(ns["scale"])
        return transform, scale

    raise ValueError(f"Unknown dataparser file type: {path.suffix}")


def build_T4x4_from_3x4(transform_3x4: np.ndarray) -> np.ndarray:
    """
    Given a 3x4 or 4x4 array, return a 4x4 transform.
    """
    transform_3x4 = np.array(transform_3x4, dtype=float)
    if transform_3x4.shape == (4, 4):
        return transform_3x4

    if transform_3x4.shape == (3, 4):
        T = np.eye(4, dtype=float)
        T[:3, :4] = transform_3x4
        return T

    raise ValueError(f"Expected transform to be 3x4 or 4x4, got {transform_3x4.shape}")


def compute_fov_deg(fl: float, x: float) -> float:
    """
    Vertical FOV (in degrees) from focal length and image height.
    Nerfstudio camera paths use vertical FOV.
    """
    fov_rad = 2.0 * math.atan(0.5 * x / fl)
    return math.degrees(fov_rad)


# --- Helper Functions from scale_compare.py / evaluate_poses.py ---


def read_json(json_path):
    f = open(json_path)
    positions = json.load(f)
    f.close()
    return positions


def filter_gt_by_target(gt, target, subsample=1):
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


def reshape4x4(M):
    if isinstance(M, list) and len(M) == 16:
        M = [M[i * 4 : (i + 1) * 4] for i in range(4)]
    M = np.asarray(M, dtype=float)
    assert M.shape == (4, 4), f"Expected 4x4, got {M.shape}"
    return M


def get_matrix_from_frame(frame):
    for k in ("transform_matrix", "transform", "pose"):
        if k in frame:
            return reshape4x4(frame[k])
    raise KeyError(
        "No transform matrix found in frame (looked for 'transform_matrix', 'transform', 'pose')."
    )


def center_from_matrix(M, mode):
    R = M[:3, :3]
    t = M[:3, 3]
    if mode == "c2w":
        return t
    elif mode == "w2c":
        return -(R.T @ t)
    else:
        raise ValueError("mode must be 'c2w' or 'w2c'")


def centers_from_frames(frames, mode):
    Cs = []
    for fr in frames:
        M = get_matrix_from_frame(fr)
        Cs.append(center_from_matrix(M, mode))
    return np.stack(Cs, axis=0)  # (N,3)


def match_frames(a_dict, b_dict, subsample=1, sort_by_name=True):
    # Make deep copies to avoid mutating original dicts
    A = copy.deepcopy(a_dict)
    B = copy.deepcopy(b_dict)
    # Two-way filter -> intersection
    A = filter_gt_by_target(A, B, subsample=subsample)
    B = filter_gt_by_target(B, A, subsample=1)  # keep same set
    # Enforce identical sets of names and counts
    namesA = [os.path.basename(x["file_path"]).split(".")[0] for x in A["frames"]]
    namesB = [os.path.basename(x["file_path"]).split(".")[0] for x in B["frames"]]
    common = sorted(set(namesA).intersection(namesB))
    if not common:
        raise RuntimeError("No overlapping frames after filtering.")

    def sort_frames(frames):
        if not sort_by_name:
            return frames
        # Sort by basename stem to ensure consistent order across A/B
        return sorted(
            frames, key=lambda x: os.path.basename(x["file_path"]).split(".")[0]
        )

    A_sorted = [
        fr
        for fr in sort_frames(A["frames"])
        if os.path.basename(fr["file_path"]).split(".")[0] in common
    ]
    B_sorted = [
        fr
        for fr in sort_frames(B["frames"])
        if os.path.basename(fr["file_path"]).split(".")[0] in common
    ]

    if len(A_sorted) != len(B_sorted):
        raise RuntimeError(
            f"Matched lists differ in length: {len(A_sorted)} vs {len(B_sorted)}"
        )

    print(f"Matched {len(A_sorted)} frames between T1 and T2.")
    return A_sorted, B_sorted


def umeyama_o3d_scale_and_rms(CA, CB):
    """
    Open3D closed-form Sim(3) with scaling. Computes CA->CB mapping:
        CB ≈ s * R * CA + t
    Returns (s, T(4x4), rms).
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(CA))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(CB))
    # 1-1 correspondences by index
    corr = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(len(CA)), np.arange(len(CB))]).astype(np.int32)
    )
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    T = est.compute_transformation(src, dst, corr)  # 4x4, top-left is s*R
    SR = T[:3, :3]
    # Uniform scale s; det(SR) = s^3 (since det(R)=+1 ideally)
    s = float(np.cbrt(np.linalg.det(SR)))

    # Handle reflection
    if s < 0:
        s = -s
        SR = -SR
        T[:3, :3] = SR

    # RMS residual
    CA_aligned = (SR @ CA.T).T + T[:3, 3]
    rms = float(np.sqrt(np.mean(np.sum((CB - CA_aligned) ** 2, axis=1))))
    return s, T, rms


def choose_best_mode(framesA, framesB):
    # Try all c2w/w2c combos; pick the one giving smallest RMS after Umeyama
    best = None
    for mA in ("c2w", "w2c"):
        CA = centers_from_frames(framesA, mA)  # T2 centers
        for mB in ("c2w", "w2c"):
            CB = centers_from_frames(framesB, mB)  # T1 centers
            s, T, rms = umeyama_o3d_scale_and_rms(
                CA, CB
            )  # Find T2_centers -> T1_centers
            if (best is None) or (rms < best["rms"]):
                best = dict(mA=mA, mB=mB, CA=CA, CB=CB, s=s, T=T, rms=rms)
    return best


# --- End Helper Functions ---


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Nerfstudio camera_path.json from one scene's coordinate system to another."
    )
    parser.add_argument(
        "--c2", required=True, help="Path to input camera_path.json (Scene 2, C2)"
    )
    parser.add_argument(
        "--t2", default=None, help="Path to transforms.json for Scene 2 (T2)"
    )
    parser.add_argument(
        "--d2",
        required=True,
        help="Path to dataparser_transforms.json for Scene 2 (D2)",
    )
    parser.add_argument(
        "--t1",
        required=True,
        help="Path to transforms.json for Scene 1 (T1, Reference)",
    )
    parser.add_argument(
        "--d1",
        required=True,
        help="Path to dataparser_transforms.json for Scene 1 (D1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for the new C1 camera_path.json",
    )
    args = parser.parse_args()

    # --- 1. Load All Inputs ---
    if args.t2 is None:
        args.t2 = Path(args.c2).parent.parent / "transforms.json"

    print("Loading all input files...")
    C2_json = read_json(args.c2)
    T2_raw = read_json(args.t2)
    T1_raw = read_json(args.t1)

    T_raw_2, scale_2 = load_dataparser_transforms(Path(args.d2))
    T_raw_1, scale_1 = load_dataparser_transforms(Path(args.d1))

    # Convert dataparser transforms to 4x4
    T_raw_2 = build_T4x4_from_3x4(T_raw_2)
    T_raw_1 = build_T4x4_from_3x4(T_raw_1)

    # Pre-calculate inverse for D2
    inv_T_raw_2 = np.linalg.inv(T_raw_2)

    # --- 2. Get Alignment Matrix L ---
    # Match T2 (A) to T1 (B)
    framesT2, framesT1 = match_frames(T2_raw, T1_raw, sort_by_name=True)
    if not framesT2:
        print("Error: No matching frames found between T1 and T2. Cannot align.")
        sys.exit(1)

    # Find best alignment from T2_centers -> T1_centers
    # CA = T2 centers, CB = T1 centers
    best_align = choose_best_mode(framesT2, framesT1)
    L = best_align["T"]  # This is the 4x4 matrix (L) that maps W2 -> W1

    print(f"Alignment found (T2_mode={best_align['mA']}, T1_mode={best_align['mB']})")
    print(
        f"Alignment L (scale={best_align['s']:.4f}, RMS={best_align['rms']:.4e}):\n{L}"
    )

    # --- 3. Process All Poses ---
    new_camera_path_entries = []

    if "camera_path" not in C2_json:
        raise ValueError(
            "Input C2 JSON is not a valid camera path. Missing 'camera_path' key."
        )

    print(f"Transforming {len(C2_json['camera_path'])} poses...")

    # This list will store the intermediate P_W1 poses
    aligned_w1_frames = []

    for i, entry in enumerate(C2_json["camera_path"]):
        # P_NS2 (Pose in Nerfstudio Space 2)
        P_NS2 = reshape4x4(entry["camera_to_world"])

        # Step 1: "Un-apply" D2 (P_NS2 -> P_W2)
        # P_W2 = inv(D2_op)(P_NS2)
        P_NS2_unscaled = P_NS2.copy()
        P_NS2_unscaled[:3, 3] /= scale_2
        P_W2 = inv_T_raw_2 @ P_NS2_unscaled

        # Step 2: Align L (P_W2 -> P_W1)
        # P_W1 = L @ P_W2
        P_W1 = L @ P_W2

        # --- Store this intermediate pose for the new transforms.json ---
        # We copy all other data from the C2 frame (like file_path, etc.)
        new_w1_frame = {
            k: v
            for k, v in entry.items()
            if k not in ["camera_to_world", "matrix", "fov", "aspect"]
        }
        new_w1_frame["transform_matrix"] = P_W1.tolist()
        new_w1_frame["file_path"] = (
            f"DUMMY_PATH/images/{str(i).zfill(6)}.npy"  # Dummy path
        )
        aligned_w1_frames.append(new_w1_frame)
        # --- End ---

        # Step 3: "Apply" D1 (P_W1 -> P_NS1)
        # P_NS1 = D1_op(P_W1)
        P_NS1 = T_raw_1 @ P_W1
        P_NS1[:3, 3] *= scale_1

        # Add the new pose to our list
        new_entry = entry.copy()  # Keep fov, aspect from C2 for now
        new_entry["camera_to_world"] = P_NS1.reshape(-1).tolist()
        new_camera_path_entries.append(new_entry)

    print("All poses transformed.")

    # --- 3.5 Save Intermediate Aligned transforms.json (P_W1) ---
    output_path = Path(args.output)

    # Use T1_raw as a template to get all header/intrinsic fields
    C1_transforms_json = {k: v for k, v in T1_raw.items() if k != "frames"}
    C1_transforms_json["frames"] = aligned_w1_frames

    # Save it in the same directory with a new name
    transforms_out_path = output_path.parent / (
        output_path.stem + "_aligned_transforms.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with transforms_out_path.open("w") as f:
        json.dump(C1_transforms_json, f, indent=4)
    print(f"\nSuccessfully saved aligned transforms (C1-Raw) to: {transforms_out_path}")

    # --- 4. Build Final C1 JSON (Camera Path) ---
    print("\nBuilding new camera path JSON (C1)...")

    # We will use C2 as a template but replace the critical parts
    C1_json = copy.deepcopy(C2_json)

    # Update intrinsics from T1
    w1 = float(T1_raw["w"])
    h1 = float(T1_raw["h"])
    fl_y1 = float(T1_raw["fl_y"])  # Nerfstudio uses vertical FOV
    aspect1 = w1 / h1
    fov_deg1 = compute_fov_deg(fl_y1, h1)

    print(
        f"Setting new intrinsics (from T1): W={int(w1)}, H={int(h1)}, vFOV={fov_deg1:.2f}"
    )

    C1_json["render_width"] = int(w1)
    C1_json["render_height"] = int(h1)
    C1_json["default_fov"] = fov_deg1

    # Update camera path
    C1_json["camera_path"] = new_camera_path_entries

    # Update keyframes (if they exist)
    if "keyframes" in C1_json:
        print("Transforming keyframes as well...")
        new_keyframes = []
        for frame in C1_json["keyframes"]:
            # P_NS2 (Pose in Nerfstudio Space 2)
            P_NS2 = reshape4x4(frame["matrix"])

            # Step 1: "Un-apply" D2 (P_NS2 -> P_W2)
            P_NS2_unscaled = P_NS2.copy()
            P_NS2_unscaled[:3, 3] /= scale_2
            P_W2 = inv_T_raw_2 @ P_NS2_unscaled

            # Step 2: Align L (P_W2 -> P_W1)
            P_W1 = L @ P_W2

            # Step 3: "Apply" D1 (P_W1 -> P_NS1)
            P_NS1 = T_raw_1 @ P_W1
            P_NS1[:3, 3] *= scale_1

            new_frame = frame.copy()
            new_frame["matrix"] = P_NS1.reshape(-1).tolist()
            new_frame["fov"] = fov_deg1
            new_frame["aspect"] = aspect1
            new_keyframes.append(new_frame)

        C1_json["keyframes"] = new_keyframes

    # --- 5. Save Final Camera Path ---
    with output_path.open("w") as f:
        json.dump(C1_json, f, indent=2)

    print(f"\nSuccessfully saved new camera path (C1-Nerfstudio) to: {output_path}")


if __name__ == "__main__":
    main()
