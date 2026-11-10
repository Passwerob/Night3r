import numpy as np
import torch

import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def visualize_noisy_images(
    clean,
    simulated,
    actual,
    viz_path,
    filename,
    clean_vmin=None,
    clean_vmax=None,
    noisy_vmin=None,
    noisy_vmax=None,
    scaling_factor=-1,
    scaling_constant=-1,
):
    """
    Create a 3x3 grid visualization showing the separate R, G, B channels.

    - Row 0: Clean image channels using the provided bounds (clean_vmin, clean_vmax).
    - Row 1: Simulated noisy image channels using the provided noisy bounds (noisy_vmin, noisy_vmax).
    - Row 2: Actual noisy image channels using the provided noisy bounds.

    Two global colorbars are added at the bottom of the figure:
      - One colorbar for the clean image scale.
      - One colorbar for the simulated/actual noisy image scale.

    The images are shown with the 'viridis' colormap.

    Parameters:
        clean (np.ndarray): Clean image array of shape (H, W, 3).
        simulated (np.ndarray): Simulated noisy image array of shape (H, W, 3).
        actual (np.ndarray): Actual noisy image array of shape (H, W, 3).
        viz_path (str): Directory where the visualization image will be saved.
        filename (str): Base filename for saving the visualization.
        clean_vmin (float): Lower bound for the clean image visualization.
        clean_vmax (float): Upper bound for the clean image visualization.
        noisy_vmin (float): Lower bound for the simulated/actual noisy image visualization.
        noisy_vmax (float): Upper bound for the simulated/actual noisy image visualization.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    image_titles = ["Clean", "Simulated Noisy", "Actual Noisy"]
    channel_titles = ["Red", "Green", "Blue"]

    # Row 0: Clean image channels using provided clean bounds.
    for ch in range(3):
        ax = axes[0, ch]
        m = int(np.mean(clean[..., ch]))
        mn = int(np.min(clean[..., ch]))
        mx = int(np.max(clean[..., ch]))
        stat_text = f"Mean: {m}, Min: {mn}, Max: {mx}"
        im = ax.imshow(clean[..., ch], cmap="plasma", vmin=clean_vmin, vmax=clean_vmax)
        ax.set_title(
            f"{image_titles[0]} - {channel_titles[ch]}\n{stat_text}", fontsize=12
        )

        ax.axis("off")

    # Row 1: Simulated noisy image channels using provided noisy bounds.
    for ch in range(3):
        ax = axes[1, ch]
        m = int(np.mean(simulated[..., ch]))
        mn = int(np.min(simulated[..., ch]))
        mx = int(np.max(simulated[..., ch]))
        stat_text = f"Mean: {m}, Min: {mn}, Max: {mx}"
        im = ax.imshow(
            simulated[..., ch], cmap="viridis", vmin=noisy_vmin, vmax=noisy_vmax
        )
        ax.set_title(
            f"{image_titles[1]} - {channel_titles[ch]}\n{stat_text}", fontsize=12
        )

        ax.axis("off")

    if actual is not None:
        # Row 2: Actual noisy image channels using provided noisy bounds.
        for ch in range(3):
            ax = axes[2, ch]
            m = int(np.mean(actual[..., ch]))
            mn = int(np.min(actual[..., ch]))
            mx = int(np.max(actual[..., ch]))
            stat_text = f"Mean: {m}, Min: {mn}, Max: {mx}"
            im = ax.imshow(
                actual[..., ch], cmap="viridis", vmin=noisy_vmin, vmax=noisy_vmax
            )
            ax.set_title(
                f"{image_titles[2]} - {channel_titles[ch]}\n{stat_text}", fontsize=12
            )

            ax.axis("off")

    # Create ScalarMappables for the two scales.
    sm_clean = cm.ScalarMappable(
        norm=colors.Normalize(vmin=clean_vmin, vmax=clean_vmax), cmap="plasma"
    )
    sm_clean.set_array([])
    sm_noisy = cm.ScalarMappable(
        norm=colors.Normalize(vmin=noisy_vmin, vmax=noisy_vmax), cmap="viridis"
    )
    sm_noisy.set_array([])

    # Adjust layout and add two global horizontal colorbars at the bottom.
    fig.suptitle(
        f"Scaling Factor: {scaling_factor:.2f} | Scaling Constant: {scaling_constant:.2f}",
        fontsize=12,
    )
    plt.tight_layout()
    cbar_ax_clean = fig.add_axes([0.125, 0.045, 0.35, 0.02])
    cbar_ax_noisy = fig.add_axes([0.525, 0.045, 0.35, 0.02])
    fig.colorbar(
        sm_clean, cax=cbar_ax_clean, orientation="horizontal", label="Clean Scale"
    )
    fig.colorbar(
        sm_noisy, cax=cbar_ax_noisy, orientation="horizontal", label="Noisy Scale"
    )

    os.makedirs(viz_path, exist_ok=True)
    save_path = os.path.join(viz_path, f"{filename}_vis.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    # print(f"Saved visualization to {save_path}")


def convert_and_normalize(image, max_value):
    """
    Convert a numpy image to a torch float tensor, normalize it to [-1, 1],
    and rearrange the dimensions to (C, H, W).

    Parameters:
        image (np.array): Input image array.
            - For 8-bit images: values in [0, 255]
            - For 16-bit images: values in [0, 16383]
        max_value (int): The maximum possible pixel value.

    Returns:
        torch.Tensor: A float tensor with shape (C, H, W) and values in [-1, 1].
    """
    image = image.astype(np.float32)
    normalized = image / (max_value / 2.0) - 1.0  # Normalize to [-1, 1]
    tensor = torch.from_numpy(normalized).permute(
        2, 0, 1
    )  # Convert to tensor and rearrange to (C, H, W)
    return tensor


def normalize_burst_tensor(tensor, max_value):
    normalized = tensor / (max_value / 2.0) - 1.0  # Normalize to [-1, 1]

    n, c, h, w = normalized.shape
    normalized = normalized.reshape(n * c, h, w)

    return normalized


def normalize_tensor(tensor, max_value):
    normalized = tensor / (max_value / 2.0) - 1.0  # Normalize to [-1, 1]

    return normalized


def return_tensor(tensor, current_idx):
    true_shape = tensor.size()[2:] if len(tensor.size()) == 4 else tensor.size()[1:]
    res = dict(
        img=tensor,
        true_shape=np.int32([true_shape]),
        idx=current_idx,
        instance=str(current_idx),
    )
    return res
