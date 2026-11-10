from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_night3r_trajectory import (
    estimate_relative_pose_essential,
    estimate_relative_pose_pointmap_sim3,
    estimate_relative_pose_pnp,
    select_best_pointmap_registration,
    trajectory_metrics,
)


def test_dense_pnp_recovers_known_relative_pose():
    rng = np.random.default_rng(5)
    h, w = 48, 64
    intrinsics = np.array([[80.0, 0.0, 31.5], [0.0, 80.0, 23.5], [0.0, 0.0, 1.0]])
    rvec_gt = np.array([0.02, -0.03, 0.01], dtype=np.float64)
    rotation_gt, _ = cv2.Rodrigues(rvec_gt)
    translation_gt = np.array([0.12, -0.03, 0.08], dtype=np.float64)

    yy, xx = np.mgrid[:h, :w]
    depth = rng.uniform(2.0, 8.0, size=(h, w))
    points_j = np.stack(
        [
            (xx - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ],
        axis=-1,
    )
    points_i = (rotation_gt.T @ (points_j.reshape(-1, 3) - translation_gt).T).T
    points_i = points_i.reshape(h, w, 3)
    confidence = np.ones((h, w), dtype=np.float64)

    estimate, diagnostics = estimate_relative_pose_pnp(
        points_i,
        confidence,
        intrinsics,
        conf_quantile=0.0,
        max_points=h * w,
        reprojection_error=0.5,
        min_inliers=128,
        seed=5,
    )
    assert diagnostics["success"]
    assert diagnostics["inlier_ratio"] > 0.99
    assert np.allclose(estimate[:3, :3], rotation_gt, atol=1e-5)
    assert np.allclose(estimate[:3, 3], translation_gt, atol=1e-5)


def test_identical_trajectories_have_zero_ate_and_rpe():
    poses = []
    for index in range(5):
        pose = np.eye(4)
        pose[0, 3] = 0.2 * index
        pose[1, 3] = 0.03 * index**2
        pose[2, 3] = 0.01 * index**3
        poses.append(pose)
    result = trajectory_metrics(poses, poses)
    assert result["ate_translation_m"]["rmse"] < 1e-10
    assert result["rpe_translation_m"]["rmse"] < 1e-10
    assert result["rpe_rotation_deg"]["rmse"] < 1e-10


def test_rotation_metrics_do_not_inherit_position_alignment_roll():
    roll, _ = cv2.Rodrigues(np.array([np.deg2rad(150.0), 0.0, 0.0]))
    pred_poses = []
    gt_poses = []
    for index in range(8):
        rotation, _ = cv2.Rodrigues(np.array([0.0, 0.01 * index, 0.0]))
        gt_position = np.array([0.2 * index, 0.01 * index**2, 0.0])
        pred_position = roll @ gt_position
        gt_pose = np.eye(4)
        pred_pose = np.eye(4)
        gt_pose[:3, :3] = rotation
        pred_pose[:3, :3] = rotation
        gt_pose[:3, 3] = gt_position
        pred_pose[:3, 3] = pred_position
        gt_poses.append(gt_pose)
        pred_poses.append(pred_pose)

    result = trajectory_metrics(pred_poses, gt_poses)
    assert result["ate_translation_m"]["rmse"] < 1e-10
    assert result["ape_rotation_deg"]["rmse"] < 1e-10
    assert result["rpe_rotation_deg"]["rmse"] < 1e-10


def test_essential_pose_recovers_rotation_and_translation_direction():
    rng = np.random.default_rng(11)
    intrinsics = np.array([[420.0, 0.0, 255.5], [0.0, 420.0, 255.5], [0.0, 0.0, 1.0]])
    points3d = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, 400),
            rng.uniform(-1.0, 1.0, 400),
            rng.uniform(4.0, 10.0, 400),
        ]
    )
    rvec_gt = np.array([0.015, -0.025, 0.01])
    rotation_gt, _ = cv2.Rodrigues(rvec_gt)
    translation_gt = np.array([0.3, -0.05, 0.08])
    points3d_2 = (rotation_gt @ points3d.T).T + translation_gt

    def project(points):
        pixels = (intrinsics @ points.T).T
        return pixels[:, :2] / pixels[:, 2:3]

    estimate, diagnostics = estimate_relative_pose_essential(
        project(points3d),
        project(points3d_2),
        intrinsics,
        reprojection_error=0.5,
        min_inliers=64,
        seed=11,
    )
    assert diagnostics["success"]
    rotation_delta = estimate[:3, :3] @ rotation_gt.T
    rotation_error = np.degrees(
        np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
    )
    direction_cosine = np.dot(estimate[:3, 3], translation_gt) / (
        np.linalg.norm(estimate[:3, 3]) * np.linalg.norm(translation_gt)
    )
    assert rotation_error < 0.1
    assert direction_cosine > 0.99


def test_pointmap_sim3_recovers_pose_with_scale_change_and_outliers():
    rng = np.random.default_rng(21)
    h, w = 48, 64
    source = rng.normal(size=(h, w, 3))
    source[..., 2] += 5.0
    rotation_gt, _ = cv2.Rodrigues(np.array([0.04, -0.02, 0.03]))
    translation_gt = np.array([0.4, -0.1, 0.2])
    scale_gt = 1.35
    target = (scale_gt * (rotation_gt @ source.reshape(-1, 3).T)).T + translation_gt
    target = target.reshape(h, w, 3)
    outliers = rng.choice(h * w, size=300, replace=False)
    target.reshape(-1, 3)[outliers] += rng.normal(scale=8.0, size=(len(outliers), 3))
    confidence = np.ones((h, w), dtype=np.float64)

    estimate, diagnostics = estimate_relative_pose_pointmap_sim3(
        source,
        target,
        confidence,
        confidence,
        conf_quantile=0.0,
        max_points=h * w,
        trim_quantile=0.8,
        min_inliers=512,
        seed=21,
    )
    assert diagnostics["success"]
    assert abs(diagnostics["sim3_scale"] - scale_gt) < 1e-4
    assert diagnostics["geometric_residual"] < 1e-4
    assert np.allclose(estimate[:3, :3], rotation_gt, atol=1e-4)
    direction_cosine = np.dot(estimate[:3, 3], translation_gt) / (
        np.linalg.norm(estimate[:3, 3]) * np.linalg.norm(translation_gt)
    )
    assert direction_cosine > 0.999


def test_pointmap_sim3_rejects_nonrigid_point_maps():
    rng = np.random.default_rng(37)
    h, w = 48, 64
    source = rng.normal(size=(h, w, 3))
    source[..., 2] += 5.0
    target = rng.normal(size=(h, w, 3))
    target[..., 2] += 5.0
    confidence = np.ones((h, w), dtype=np.float64)

    estimate, diagnostics = estimate_relative_pose_pointmap_sim3(
        source,
        target,
        confidence,
        confidence,
        conf_quantile=0.0,
        max_points=h * w,
        trim_quantile=0.8,
        min_inliers=512,
        max_normalized_rmse=0.25,
        seed=37,
    )
    assert estimate is None
    assert not diagnostics["success"]
    assert diagnostics["reason"] == "pointmap_residual_above_threshold"
    assert diagnostics["geometric_residual"] > diagnostics["max_geometric_residual"]


def test_failed_pointmap_candidates_keep_the_lower_residual_diagnostic():
    registrations = [
        (None, {"success": False, "geometric_residual": 1.3}),
        (None, {"success": False, "geometric_residual": 0.9}),
    ]
    estimate, diagnostics = select_best_pointmap_registration(registrations)
    assert estimate is None
    assert diagnostics["geometric_residual"] == 0.9
