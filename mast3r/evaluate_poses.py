from evo.core.trajectory import PosePath3D, PoseTrajectory3D
import numpy as np
import json
from evo_utils import eval_metrics
import os
import argparse


def tranforms_to_trajectory(transforms):
    frames = sorted(transforms["frames"], key=lambda x: x["file_path"])  # SORTS HERE

    matrices = [fix_transform_matrix(x["transform_matrix"]) for x in frames]
    frames = np.stack(matrices, 0)
    # return PoseTrajectory3D(
    #         positions_xyz=frames[:, :3, 3],
    #         orientations_quat_wxyz=traj[:, 3:],
    #         timestamps=tstamps,
    #     )
    return PosePath3D(poses_se3=frames)


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


def evaluate_poses(
    transforms, transforms_gt, path="./", rot_thresh=0, trans_thresh=0, subsample=1
):

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
    traj, traj_gt = tranforms_to_trajectory(
        filtered_transforms
    ), tranforms_to_trajectory(filtered_transforms_gt)
    eval_metrics(
        traj,
        traj_gt,
        filename=path,
        rotation_threshold=rot_thresh,
        translation_threshold=trans_thresh,
        transforms_pred=filtered_transforms,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pose estimations against ground truth."
    )
    parser.add_argument(
        "transforms_gt", type=str, help="Path to the ground truth transforms JSON file."
    )
    parser.add_argument(
        "transforms", type=str, help="Path to the estimated transforms JSON file."
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for evaluating poses ",
    )
    parser.add_argument(
        "--rot_thresh",
        type=float,
        default=0.05,
        help="Rotation threshold in degrees for RRA (default: 0.05°)",
    )
    parser.add_argument(
        "--trans_thresh",
        type=float,
        default=0.01,
        help="Translation threshold (units) for RTA (default: 0.01 units)",
    )

    args = parser.parse_args()

    output = os.path.join(os.path.dirname(args.transforms), f"eval_pose.txt")

    evaluate_poses(
        args.transforms,
        args.transforms_gt,
        output,
        args.rot_thresh,
        args.trans_thresh,
        args.subsample,
    )
