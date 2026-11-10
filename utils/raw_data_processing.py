#!/usr/bin/env python3
import numpy as np

import glob

from dataset.dataset_utils import convert_and_normalize, return_tensor


def load_npy_images(image_pattern, max_value=16383):
    """
    Load a set of images from a given pattern and return them as a list of torch tensors.

    Parameters:
        image_pattern (str or list): A glob pattern to search for images OR a list of image paths.
        max_value (int): The maximum possible pixel value.

    Returns:
        list[torch.Tensor]: A list of torch tensors with shape (C, H, W) and values in [-1, 1].
    """
    if type(image_pattern) == str:
        image_paths = sorted(glob.glob(image_pattern))
    elif type(image_pattern) == list:
        image_paths = image_pattern
    images = []
    for idx, path in enumerate(image_paths):
        image = np.load(path)
        tensor = convert_and_normalize(image, max_value)
        images.append(return_tensor(tensor[None], idx))

    # print(f' (Found {len(images)} images)')
    return images
