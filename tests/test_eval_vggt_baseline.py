import torch

from utils.eval_vggt_baseline import event_voxel_to_red_blue, prediction_depth


def test_event_voxel_to_red_blue_preserves_split_polarity():
    event = torch.zeros(1, 2, 8, 2, 2)
    event[0, 0, 1, 0, 0] = -3.0
    event[0, 0, 6, 1, 1] = 4.0

    image = event_voxel_to_red_blue(event, (2, 2))

    assert image.shape == (1, 2, 3, 2, 2)
    assert torch.equal(image[0, 0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(image[0, 0, :, 1, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(image[0, 0, :, 0, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_prediction_depth_selects_first_pair_frame():
    depth = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4, 1)
    selected = prediction_depth({"depth": depth})

    assert selected.shape == (1, 1, 3, 4)
    assert torch.equal(selected[:, 0], depth[:, 0, :, :, 0])
