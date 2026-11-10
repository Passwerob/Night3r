import json

import torch

from utils.eval_depthanyevent_baseline import (
    apply_gt_scale_shift,
    checkpoint_settings,
    decode_prediction,
    scene_names,
    select_cached_depth,
    voxel_to_tencode,
)


def test_apply_gt_scale_shift_recovers_affine_target():
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = 2.5 * prediction + 7.0
    valid = torch.ones_like(target, dtype=torch.bool)

    aligned, scale, shift = apply_gt_scale_shift(
        prediction, target, valid, clip_distance=100.0
    )

    assert torch.allclose(scale, torch.tensor([2.5]))
    assert torch.allclose(shift, torch.tensor([7.0]))
    assert torch.allclose(aligned, target)


def test_apply_gt_scale_shift_ignores_invalid_gt_pixels():
    prediction = torch.tensor([[[[1.0, 2.0], [3.0, 100.0]]]])
    target = torch.tensor([[[[3.0, 5.0], [7.0, float("nan")]]]])
    valid = torch.tensor([[[[True, True], [True, False]]]])

    aligned, scale, shift = apply_gt_scale_shift(
        prediction, target, valid, clip_distance=1000.0
    )

    assert torch.allclose(scale, torch.tensor([2.0]))
    assert torch.allclose(shift, torch.tensor([1.0]))
    assert torch.allclose(aligned[0, 0, 0, :], target[0, 0, 0, :])


def test_voxel_to_tencode_uses_polarity_and_latest_active_bin():
    voxel = torch.zeros(1, 8, 2, 2)
    voxel[0, 0, 0, 0] = 2.0
    voxel[0, 6, 0, 0] = 3.0
    voxel[0, 1, 1, 1] = -4.0

    encoded = voxel_to_tencode(voxel)

    assert encoded.shape == (1, 3, 2, 2)
    assert torch.equal(encoded[0, :, 0, 0], torch.tensor([0.0, 1.0 / 3.0, 1.0]))
    assert torch.equal(encoded[0, :, 1, 1], torch.tensor([1.0, 2.0 / 3.0, 0.0]))
    assert torch.equal(encoded[0, :, 0, 1], torch.zeros(3))


def test_voxel_to_tencode_breaks_equal_activity_toward_positive():
    voxel = torch.zeros(1, 4, 1, 1)
    voxel[0, 1, 0, 0] = -2.0
    voxel[0, 3, 0, 0] = 2.0

    encoded = voxel_to_tencode(voxel)

    assert torch.equal(encoded[0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_decode_logdepth_normalizes_each_sample_independently():
    prediction = torch.tensor([[[[1.0, 0.0]]], [[[2.0, 1.0]]]])

    decoded = decode_prediction(prediction, True, clip_distance=10.0, reg_factor=1.0)

    expected = torch.tensor([[[[10.0, 10.0 / torch.e]]], [[[10.0, 10.0 / torch.e]]]])
    assert torch.allclose(decoded, expected)


def test_scene_names_falls_back_to_sample_id_prefix():
    batch = {"sample_id": ["screen-000001:000002-000003"]}
    assert scene_names(batch, 1) == ["screen-000001"]


def test_second_view_uses_its_own_depth_and_mask():
    depth1 = torch.tensor([[[1.0, 2.0]]])
    depth2 = torch.tensor([[[3.0, float("nan")]]])
    batch = {
        "depth_gt": depth1,
        "depth_gt2": depth2,
        "depth_valid": torch.ones_like(depth1, dtype=torch.bool),
    }

    selected, valid = select_cached_depth(batch, 1)

    assert selected[0, 0, 0, 0] == 3.0
    assert torch.isnan(selected[0, 0, 0, 1])
    assert torch.equal(valid, torch.tensor([[[[True, False]]]]))


def test_checkpoint_settings_reads_external_official_config(tmp_path):
    config = {
        "trainer": {"use_scaleshift": True, "clip_distance": 80.0},
        "model": {"model_type": "RecDAv2"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    settings = checkpoint_settings({"state_dict": {}}, model_config_path=path)

    assert settings["model_type"] == "RecDAv2"
    assert settings["checkpoint_use_scaleshift"] is True
