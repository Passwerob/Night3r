import torch

from train_event_dark3r import slice_outputs


def test_slice_outputs_preserves_missing_teacher_predictions():
    assert slice_outputs((None, None), 0, 1, 2) == (None, None)


def test_slice_outputs_still_slices_batched_prediction_tensors():
    outputs = (
        {"depth": torch.tensor([[1.0], [2.0]]), "event_enabled": True},
        {"depth": torch.tensor([[3.0], [4.0]]), "event_enabled": True},
    )

    sliced = slice_outputs(outputs, 1, 2, 2)

    assert torch.equal(sliced[0]["depth"], torch.tensor([[2.0]]))
    assert torch.equal(sliced[1]["depth"], torch.tensor([[4.0]]))
    assert sliced[0]["event_enabled"] is True
