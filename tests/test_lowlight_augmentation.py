from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.lowlight_augmentation import ContinuousLowLightAug


def test_shot_noise_std_defaults_to_legacy_read_noise_std():
    aug = ContinuousLowLightAug(read_noise_std=0.02, shot_noise=True)
    assert aug.shot_noise_std == 0.02


def test_shot_only_and_read_only_are_independently_configurable():
    image = torch.full((3, 16, 16), 0.5)
    shot_only = ContinuousLowLightAug(
        min_exposure=0.5,
        max_exposure=0.5,
        read_noise_std=0.0,
        shot_noise=True,
        shot_noise_std=0.01,
    )
    read_only = ContinuousLowLightAug(
        min_exposure=0.5,
        max_exposure=0.5,
        read_noise_std=0.01,
        shot_noise=False,
    )
    torch.manual_seed(3)
    out_shot = shot_only(image)
    torch.manual_seed(3)
    out_read = read_only(image)
    assert torch.isfinite(out_shot).all()
    assert torch.isfinite(out_read).all()
    assert not torch.allclose(out_shot, out_read)
