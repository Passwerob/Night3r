import math

import numpy as np
import torch


class ContinuousLowLightAug:
    """Physics-inspired continuous low-light degradation.

    The augmentation expects image values in [0, 1]. It does not use any learned
    enhancement model.
    """

    def __init__(
        self,
        min_exposure=0.03,
        max_exposure=1.0,
        read_noise_std=0.01,
        shot_noise=True,
        shot_noise_std=None,
        gamma_space_input=True,
        return_metadata=False,
    ):
        if min_exposure <= 0 or max_exposure <= 0:
            raise ValueError("exposure bounds must be positive")
        if min_exposure > max_exposure:
            raise ValueError("min_exposure must be <= max_exposure")
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.read_noise_std = read_noise_std
        self.shot_noise = shot_noise
        self.shot_noise_std = (
            float(read_noise_std) if shot_noise_std is None else float(shot_noise_std)
        )
        self.gamma_space_input = gamma_space_input
        self.return_metadata = return_metadata
        self.last_metadata = {}

    def _sample_exposure(self, device=None):
        lo = math.log(self.min_exposure)
        hi = math.log(self.max_exposure)
        u = torch.rand((), device=device)
        return torch.exp(torch.empty((), device=device).fill_(lo) + u * (hi - lo))

    def __call__(self, image):
        is_numpy = isinstance(image, np.ndarray)
        if is_numpy:
            image_t = torch.from_numpy(image).float()
        elif torch.is_tensor(image):
            image_t = image.float()
        else:
            image_t = torch.as_tensor(image).float()

        exposure = self._sample_exposure(device=image_t.device).to(image_t.dtype)
        img = image_t.clamp(0, 1)

        if self.gamma_space_input:
            img_linear = img.pow(2.2)
        else:
            img_linear = img

        img_dark = img_linear * exposure
        if self.shot_noise:
            shot_std = torch.sqrt(img_dark.clamp_min(0) + 1e-6) * self.shot_noise_std
            img_dark = img_dark + torch.randn_like(img_dark) * shot_std
        if self.read_noise_std > 0:
            img_dark = img_dark + torch.randn_like(img_dark) * self.read_noise_std

        if self.gamma_space_input:
            out = img_dark.clamp(0, 1).pow(1.0 / 2.2)
        else:
            out = img_dark.clamp(0, 1)

        metadata = {"exposure_scale": float(exposure.detach().cpu())}
        self.last_metadata = metadata

        if is_numpy:
            out_np = out.detach().cpu().numpy()
            if np.issubdtype(image.dtype, np.floating):
                out_np = out_np.astype(image.dtype, copy=False)
            out = out_np
        elif torch.is_tensor(image):
            out = out.to(dtype=image.dtype)

        if self.return_metadata:
            return out, metadata
        return out
