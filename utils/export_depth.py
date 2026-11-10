#!/usr/bin/env python3
"""
Export sparse depth maps and confidence maps from a Dark3R reconstruction scene.

This script loads a scene.pkl file from Dark3R inference and exports depth maps
and confidence maps as .npy files, naming them according to the original image names.
"""

import argparse
import os
import pickle

import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and reshape depthmaps and confidences from a Dark3R scene, "
        "naming them by original imagename"
    )
    parser.add_argument(
        "--pkl_path",
        "-p",
        type=str,
        required=True,
        help="Path to the .pkl containing the Dark3R scene (typically scene.pkl)",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=None,
        help="Base directory under which to create 'depth' and 'conf' folders "
        "(default: same directory as pkl_path)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=8,
        help="Subsample factor for dense depth computation (default: 8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.pkl_path))

    # 1) Load the Dark3R scene object from pickle
    print(f"Loading scene from {args.pkl_path}...")
    with open(args.pkl_path, "rb") as f:
        scene = pickle.load(f)

    # 2) Load the matching transforms.json from the parent of the pickle file
    parent_dir = os.path.dirname(os.path.abspath(args.pkl_path))
    transforms_path = os.path.join(parent_dir, "transforms.json")

    if not os.path.exists(transforms_path):
        raise FileNotFoundError(
            f"transforms.json not found at {transforms_path}. "
            "Please ensure transforms.json is in the same directory as scene.pkl"
        )

    with open(transforms_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", data)

    # Extract the original image filenames
    image_names = []
    for f in frames:
        imagename = f.get("imagename", f.get("file_path", ""))
        image_names.append(os.path.splitext(os.path.basename(imagename))[0])

    # 3) Get the raw depth and confidence data
    #    scene.get_dense_pts3d() returns (pts3d, depthmaps, confs)
    print(f"Computing dense depth maps with subsample={args.subsample}...")
    _, depthmaps, confs = scene.get_dense_pts3d(subsample=args.subsample)
    num_views = len(depthmaps)
    assert num_views == len(confs) == len(image_names), (
        f"Mismatch: depthmaps={num_views}, confs={len(confs)}, "
        f"transforms.json frames={len(image_names)}"
    )

    # 4) Determine image H,W from the first confidence map
    if hasattr(confs[0], "cpu"):
        H, W = confs[0].cpu().numpy().shape
    else:
        H, W = confs[0].shape

    # 5) Prepare output subdirectories
    depth_dir = os.path.join(args.out_dir, "depth")
    conf_dir = os.path.join(args.out_dir, "conf")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    # 6) Loop over each view, reshape, and save
    print(f"Exporting {num_views} depth and confidence maps...")
    for i, (d_flat, c_map, name) in enumerate(zip(depthmaps, confs, image_names)):
        # a) depth: flat → (H, W)
        if hasattr(d_flat, "cpu"):
            d_np = d_flat.cpu().numpy().astype(np.float32).reshape(H, W)
        else:
            d_np = np.array(d_flat).astype(np.float32).reshape(H, W)

        # b) confidence: already (H, W)
        if hasattr(c_map, "cpu"):
            c_np = c_map.cpu().numpy().astype(np.float32)
        else:
            c_np = np.array(c_map).astype(np.float32)

        # c) build filepaths using the original imagename
        depth_path = os.path.join(depth_dir, f"{name}.npy")
        conf_path = os.path.join(conf_dir, f"{name}.npy")

        # d) save .npy arrays
        np.save(depth_path, d_np)
        np.save(conf_path, c_np)

        if (i + 1) % 10 == 0 or (i + 1) == num_views:
            print(f"[{i+1}/{num_views}] Saved depth → {depth_path}, conf → {conf_path}")

    print(f"\nExport complete! Depth maps saved to: {depth_dir}")
    print(f"Confidence maps saved to: {conf_dir}")


if __name__ == "__main__":
    main()
