import torch
import os
import json
import re

ISO_DIR_RE = re.compile(r"^iso\d{6}_s\d_\d{5}$")


def _first_iso_dir_in_parent(parent_dir: str) -> str | None:
    """Return the lexicographically first directory in parent_dir that matches ISO_DIR_RE, or None."""
    try:
        candidates = [
            d
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d)) and ISO_DIR_RE.match(d)
        ]
        if not candidates:
            return None
        candidates.sort()  # lexicographic "first"
        return candidates[0]
    except FileNotFoundError:
        return None


def _rewrite_to_clean_srgb_path(orig_path: str) -> str:
    """
    Rewrite path according to rules:
      1) Replace iso*** segment with the first iso*** in its parent folder.
      2) Replace images_npy -> images_jpg
      3) Replace .npy -> .jpg
    If the iso segment/parent can't be resolved, other replacements still apply.
    """
    parts = orig_path.split("/")
    iso_idx = None
    for i, p in enumerate(parts):
        if ISO_DIR_RE.match(p or ""):
            iso_idx = i
            break

    new_parts = parts[:]
    if iso_idx is not None and iso_idx > 0:
        parent_dir = "/".join(parts[:iso_idx]) or "/"
        replacement_iso = _first_iso_dir_in_parent(parent_dir)
        if replacement_iso:
            new_parts[iso_idx] = replacement_iso

    rewritten = "/".join(new_parts)
    rewritten = rewritten.replace("images_npy", "images_jpg")
    if rewritten.endswith(".npy"):
        rewritten = rewritten[:-4] + ".jpg"
    return rewritten


def _make_transforms_dict(w=512, h=352, elapsed_s=None):
    # Base metadata copied from your original
    td = {
        "camera_angle_x": 1.368,
        "camera_angle_y": 1.034,
        "fl_x": 313.08 * (w / 512),
        "fl_y": 311.47 * (h / 352),
        "cx": 258.01 * (w / 512),
        "cy": 176.32 * (h / 352),
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "w": w,
        "h": h,
    }
    if elapsed_s:
        hours, remainder = divmod(int(elapsed_s), 3600)
        minutes, seconds = divmod(remainder, 60)
        td["reconstruction_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"
    return td


def save_json_file(scene_state, output_path, elapsed_s=None):
    """
    Saves two files:
      1) The original transforms at output_path (unmodified file paths).
      2) A second copy with SRGB path updates at <same_dir>/<stem>_srgb.json
    """
    filenames = scene_state.sparse_ga.img_paths
    c2w = scene_state.sparse_ga.get_im_poses()

    opencv_to_opengl = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],  # Flip Y  # Flip Z
        dtype=torch.float32,
        device=c2w.device,
    )

    c2w = c2w @ opencv_to_opengl  # Post-multiplication (row-major)

    # ---- Build ORIGINAL (unmodified) transforms ----
    transforms_json_orig = _make_transforms_dict(w=512, h=352, elapsed_s=elapsed_s)
    frames_orig = []
    for name, transform in zip(filenames, c2w):
        frame = {
            "file_path": name,
            "imagename": os.path.basename(name),
            "depth_file_path": f"depth/{os.path.basename(name)}",  # keep depth tied to original basename
            "transform_matrix": (transform).tolist(),
        }
        frames_orig.append(frame)
    transforms_json_orig["frames"] = frames_orig

    # Save original
    with open(output_path, "w") as f:
        json.dump(transforms_json_orig, f, indent=2)
    print(f"Saved original json file to {output_path}")

    # # ---- Build SRGB-updated transforms ----
    # transforms_json_srgb = _make_transforms_dict(w=512, h=352, elapsed_s=elapsed_s)
    # frames_srgb = []
    # for name, transform in zip(filenames, c2w):
    #     new_file_path = _rewrite_to_clean_srgb_path(name)
    #     frame = {
    #         "file_path": new_file_path,
    #         "imagename": os.path.basename(new_file_path),
    #         "depth_file_path": f"depth/{os.path.basename(name)}",  # still use original basename for depth
    #         "transform_matrix": (transform).tolist()
    #     }
    #     frames_srgb.append(frame)
    # transforms_json_srgb["frames"] = frames_srgb

    # # Compute sibling srgb path
    # out_dir = os.path.dirname(output_path)
    # stem = os.path.splitext(os.path.basename(output_path))[0]
    # srgb_path = os.path.join(out_dir, f"{stem}_srgb.json")

    # with open(srgb_path, 'w') as f:
    #     json.dump(transforms_json_srgb, f, indent=2)
    # print(f"Saved SRGB-updated json file to {srgb_path}")

    # # Save another copy called {stem}_srgb_ds004.json where:
    # # - transform_matrix values are scaled by 4 (extrinsics translation part only)
    # # - "downsampled_016" in file paths is replaced with "downsampled_004"
    # transforms_json_ds004 = _make_transforms_dict(w=2048, h=1408, elapsed_s=elapsed_s)
    # frames_ds004 = []
    # for name, transform in zip(filenames, c2w):
    #     new_file_path = _rewrite_to_clean_srgb_path(name).replace("downsampled_016", "downsampled_004")
    #     # Scale translation part (last column of transform matrix) by 4
    #     transform_ds004 = transform.clone()
    #     frame = {
    #         "file_path": new_file_path,
    #         "imagename": os.path.basename(new_file_path),
    #         "depth_file_path": f"depth/{os.path.basename(name)}",
    #         "transform_matrix": transform_ds004.tolist()
    #     }
    #     frames_ds004.append(frame)
    # transforms_json_ds004["frames"] = frames_ds004

    # ds004_path = os.path.join(out_dir, f"{stem}_srgb_ds004.json")
    # with open(ds004_path, 'w') as f:
    #     json.dump(transforms_json_ds004, f, indent=2)
    # print(f"Saved SRGB-downsampled ds004 json file to {ds004_path}")
