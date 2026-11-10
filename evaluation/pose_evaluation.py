from evo.core.trajectory import PosePath3D
import numpy as np
import json
import os
from evaluation.evo_utils import eval_metrics


def project_to_so3(R):
    """Project a matrix to the closest valid SO(3) using SVD"""
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


def fix_transform_matrix(T):
    """Fix a 4x4 transform matrix to have a valid SO(3) rotation block"""
    T = np.array(T)
    T[:3, :3] = project_to_so3(T[:3, :3])
    return T


def read_json(json_path):
    """Read JSON file."""
    with open(json_path) as f:
        return json.load(f)


def filter_gt_by_target(gt, target, subsample=1):
    """Filter ground truth by target filenames."""
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


def transforms_to_trajectory(transforms):
    """Convert transforms JSON to trajectory."""
    frames = sorted(transforms["frames"], key=lambda x: x["file_path"])
    matrices = [fix_transform_matrix(x["transform_matrix"]) for x in frames]
    frames = np.stack(matrices, 0)
    return PosePath3D(poses_se3=frames)


def evaluate_poses(
    transforms, transforms_gt, path="./", rot_thresh=0, trans_thresh=0, subsample=1
):
    """Evaluate pose estimations against ground truth."""
    transforms = read_json(transforms)
    transforms_gt = read_json(transforms_gt)

    print(f"len(transforms_gt['frames']): {len(transforms_gt['frames'])}")
    print(f"len(transforms['frames']): {len(transforms['frames'])}")
    filtered_transforms_gt = filter_gt_by_target(transforms_gt, transforms, subsample)
    filtered_transforms = filter_gt_by_target(transforms, transforms_gt)
    print(
        f"len(filtered_transforms_gt['frames']): {len(filtered_transforms_gt['frames'])}"
    )
    print(f"len(filtered_transforms['frames']): {len(filtered_transforms['frames'])}")

    traj = transforms_to_trajectory(filtered_transforms)
    traj_gt = transforms_to_trajectory(filtered_transforms_gt)

    eval_metrics(
        traj,
        traj_gt,
        filename=path,
        rotation_threshold=rot_thresh,
        translation_threshold=trans_thresh,
        transforms_pred=filtered_transforms,
    )
