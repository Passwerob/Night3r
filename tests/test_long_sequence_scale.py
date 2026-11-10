import numpy as np
import torch

from utils.export_night3r_long_sequence import (
    estimate_initial_shared_frame_scales,
    estimate_shared_frame_depth_scale,
)


def test_shared_frame_scale_uses_pointmap_xyz_in_its_native_camera_frame():
    rng = np.random.default_rng(73)
    height, width = 32, 32
    previous = rng.normal(scale=0.4, size=(height, width, 3))
    previous[..., 2] += 12.0

    angle = np.deg2rad(4.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    translation = np.array([-0.12, 0.02, 0.08], dtype=np.float64)
    relative_pose = np.eye(4, dtype=np.float64)
    relative_pose[:3, :3] = rotation
    relative_pose[:3, 3] = translation

    previous_scale = 0.025
    current_scale = 0.008
    current = (
        (previous * previous_scale) @ rotation.T + translation
    ) / current_scale
    confidence = torch.ones((height, width), dtype=torch.float32)
    intrinsics = np.eye(3, dtype=np.float64)

    initial_previous, initial_current, support = estimate_initial_shared_frame_scales(
        torch.from_numpy(current),
        confidence,
        torch.from_numpy(previous),
        confidence,
        relative_pose,
        intrinsics,
    )
    assert support >= 512
    assert np.isclose(initial_previous, previous_scale, rtol=1.0e-5)
    assert np.isclose(initial_current, current_scale, rtol=1.0e-5)

    recovered_current, support = estimate_shared_frame_depth_scale(
        torch.from_numpy(current),
        confidence,
        torch.from_numpy(previous),
        confidence,
        relative_pose,
        previous_scale,
        intrinsics,
    )
    assert support >= 512
    assert np.isclose(recovered_current, current_scale, rtol=1.0e-5)
