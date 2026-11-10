import numpy as np

from utils.eval_vggt_trajectory import relative_pose_from_extrinsics


def test_relative_pose_from_pairwise_world_to_camera_extrinsics():
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    first = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    second = np.concatenate((rotation, np.array([[1.0], [2.0], [3.0]])), axis=1)

    relative = relative_pose_from_extrinsics(np.stack((first, second)))

    np.testing.assert_allclose(relative[:3, :3], rotation, atol=1.0e-7)
    np.testing.assert_allclose(relative[:3, 3], (1.0, 2.0, 3.0), atol=1.0e-7)
    np.testing.assert_allclose(relative[3], (0.0, 0.0, 0.0, 1.0))
