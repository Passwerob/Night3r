#!/usr/bin/env python3
"""Export a final-checkpoint MASt3R/Night3R pair comparison."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    build_event_student,
    load_backbone,
    load_config,
    move_batch_to_device,
    safe_torch_load,
)
from utils.compare_event_dark3r_checkpoint import depth_from_pred  # noqa: E402
from utils.export_night3r_pair_pointcloud import (  # noqa: E402
    camera_views,
    render_points,
    write_ply,
)
from utils.export_rgb_lowlight_event_assets import event_red_blue, load_event  # noqa: E402


BACKGROUND = (8, 12, 18)
TEXT = (244, 247, 250)
MUTED = (168, 178, 190)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/fcr/data/train_data/MVSEC_outdoor_night2_test"),
    )
    parser.add_argument("--scene", default="screen-000207-mvsec")
    parser.add_argument("--idx1", type=int, default=35)
    parser.add_argument("--idx2", type=int, default=36)
    parser.add_argument("--dataset-name", default="MVSEC")
    parser.add_argument("--source-sequence", default=None)
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Apply the standard low-light transform at a fixed exposure.",
    )
    parser.add_argument(
        "--geometry-scale-mode",
        choices=("median", "raw"),
        default="median",
        help="Scale policy used only for the pair-geometry render.",
    )
    parser.add_argument(
        "--split-audit-file",
        type=Path,
        default=None,
        help="Optional JSON audit proving that the source sequence is held out.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/train_night3r_mvsec_only_adapt_1k_from_best.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=REPO_ROOT
        / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT
        / "outputs/paper_showcase_mvsec207_final1k_mast3r_vs_night3r_20260721",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def load_pair(args, cfg):
    lowlight_enabled = args.exposure is not None
    lowlight_kwargs = None
    if lowlight_enabled:
        lowlight_kwargs = {
            "min_exposure": args.exposure,
            "max_exposure": args.exposure,
            "read_noise_std": 0.01,
            "shot_noise": True,
            "shot_noise_std": 0.01,
            "gamma_space_input": True,
        }
    dataset = DSECEventDataset(
        root_path=str(args.data_root),
        image_size=(args.image_size, args.image_size),
        pair_stride=1,
        expected_event_channels=int(cfg.get("event", {}).get("voxel_channels", 8)),
        pair_sampling="fixed",
        lowlight_enabled=lowlight_enabled,
        lowlight_kwargs=lowlight_kwargs,
    )
    for sample_index, spec in enumerate(dataset.samples):
        if (
            spec["scene"] == args.scene
            and int(spec["idx1"]) == args.idx1
            and int(spec["idx2"]) == args.idx2
        ):
            sample = dataset[sample_index]
            batch = {
                key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                for key, value in sample.items()
            }
            return sample, batch
    raise RuntimeError(
        f"Pair {args.scene}:{args.idx1:06d}-{args.idx2:06d} is unavailable"
    )


def model_args(args):
    return SimpleNamespace(
        use_tiny_model=False,
        model_name_or_path=str(args.backbone),
        backbone_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=True,
    )


def infer_mast3r(args, cfg, batch, device):
    model = load_backbone(model_args(args), cfg, model_name_or_path=str(args.backbone))
    model = model.to(device).eval()
    batch_gpu = move_batch_to_device(batch, device)
    view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=True)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        pred1_forward, _ = model(view1, view2)
        pred1_reverse, _ = model(view2, view1)
    depths = [
        depth_from_pred(pred1_forward).detach().cpu().float(),
        depth_from_pred(pred1_reverse).detach().cpu().float(),
    ]
    del pred1_forward, pred1_reverse, model, batch_gpu, view1, view2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depths


def reverse_pair_batch(batch):
    reversed_batch = dict(batch)
    for key in (
        "images",
        "images_dark",
        "images_photo",
        "images_photo_dark",
        "event_voxel",
    ):
        value = reversed_batch.get(key)
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == 2:
            reversed_batch[key] = value.flip(1)
    return reversed_batch


def infer_night3r(args, cfg, batch, device):
    backbone = load_backbone(
        model_args(args), cfg, model_name_or_path=str(args.backbone)
    )
    model = build_event_student(backbone, config=cfg)
    payload = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=True,
        purpose="final MVSEC showcase",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={list(missing)[:8]}, "
            f"unexpected={list(unexpected)[:8]}"
        )
    model = model.to(device).eval()
    batch_gpu = move_batch_to_device(batch, device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        pred1_forward, _ = model(batch_gpu)
        pred1_reverse, _ = model(reverse_pair_batch(batch_gpu))
    depths = [
        depth_from_pred(pred1_forward).detach().cpu().float(),
        depth_from_pred(pred1_reverse).detach().cpu().float(),
    ]
    load_info = {
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    del pred1_forward, pred1_reverse, model, backbone, batch_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depths, load_info


def resize_depth(depth, target_hw):
    if depth.shape[-2:] != target_hw:
        depth = F.interpolate(depth.float(), size=target_hw, mode="bilinear", align_corners=False)
    return depth[0, 0]


def evaluate_depth(depth, gt, valid):
    valid = valid & torch.isfinite(depth) & torch.isfinite(gt) & (depth > 0) & (gt > 0)
    pred_values = depth[valid].clamp_min(1.0e-8)
    gt_values = gt[valid].clamp_min(1.0e-8)
    scale = gt_values.median() / pred_values.median().clamp_min(1.0e-8)
    aligned = depth * scale
    aligned_values = aligned[valid].clamp_min(1.0e-8)
    raw_ratio = torch.maximum(pred_values / gt_values, gt_values / pred_values)
    aligned_ratio = torch.maximum(aligned_values / gt_values, gt_values / aligned_values)
    metrics = {
        "valid_pixels": int(valid.sum()),
        "raw_absrel": float(((pred_values - gt_values).abs() / gt_values).mean()),
        "raw_delta1": float((raw_ratio < 1.25).float().mean()),
        "median_scale": float(scale),
        "median_aligned_absrel": float(
            ((aligned_values - gt_values).abs() / gt_values).mean()
        ),
        "median_aligned_delta1": float((aligned_ratio < 1.25).float().mean()),
    }
    return metrics, aligned


def depth_range(gt, valid):
    values = gt[valid & torch.isfinite(gt) & (gt > 0)].numpy()
    lo, hi = np.percentile(values, (2.0, 98.0))
    return float(lo), float(hi)


def depth_image(depth, valid, lo, hi):
    array = depth.numpy()
    mask = valid.numpy() & np.isfinite(array) & (array > 0)
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[mask] = np.clip((array[mask] - lo) / max(hi - lo, 1.0e-8), 0, 1)
    color = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~mask] = np.asarray(BACKGROUND, dtype=np.uint8)
    return Image.fromarray(color)


def error_image(depth, gt, valid, error_hi):
    error = (depth - gt).abs().numpy()
    mask = valid.numpy() & np.isfinite(error)
    normalized = np.zeros_like(error, dtype=np.float32)
    normalized[mask] = np.clip(error[mask] / max(error_hi, 1.0e-8), 0, 1)
    color = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~mask] = np.asarray(BACKGROUND, dtype=np.uint8)
    return Image.fromarray(color)


def contain(image, size, background=BACKGROUND):
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def compose_panels(
    specs,
    output,
    panel_size,
    background=BACKGROUND,
    text_color=TEXT,
    muted_color=MUTED,
):
    margin = 28
    gap = 14
    header = 90
    footer = 26
    width = 2 * margin + len(specs) * panel_size + (len(specs) - 1) * gap
    height = header + panel_size + footer
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    subtitle_font = load_font(17)
    for index, spec in enumerate(specs):
        left = margin + index * (panel_size + gap)
        draw.text(
            (left + panel_size / 2, 29), spec["title"], fill=text_color,
            font=title_font, anchor="mm",
        )
        draw.text(
            (left + panel_size / 2, 63), spec["subtitle"],
            fill=spec.get("subtitle_color", muted_color), font=subtitle_font, anchor="mm",
        )
        canvas.paste(
            contain(spec["image"], panel_size, background=background), (left, header)
        )
    canvas.save(output)


def compose_pair_grid(
    rows,
    output,
    panel_size,
    background=BACKGROUND,
    text_color=TEXT,
    muted_color=MUTED,
):
    margin = 28
    gap_x = 14
    gap_y = 18
    header = 82
    row_height = header + panel_size
    columns = max(len(row) for row in rows)
    width = 2 * margin + columns * panel_size + (columns - 1) * gap_x
    height = 2 * margin + len(rows) * row_height + (len(rows) - 1) * gap_y
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23, bold=True)
    subtitle_font = load_font(16)
    for row_index, specs in enumerate(rows):
        row_top = margin + row_index * (row_height + gap_y)
        for column, spec in enumerate(specs):
            left = margin + column * (panel_size + gap_x)
            draw.text(
                (left + panel_size / 2, row_top + 24),
                spec["title"],
                fill=text_color,
                font=title_font,
                anchor="mm",
            )
            draw.text(
                (left + panel_size / 2, row_top + 56),
                spec["subtitle"],
                fill=spec.get("subtitle_color", muted_color),
                font=subtitle_font,
                anchor="mm",
            )
            canvas.paste(
                contain(spec["image"], panel_size, background=background),
                (left, row_top + header),
            )
    canvas.save(output)


def image_statistics(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = np.abs(gray - smooth)
    dark = gray < 0.20
    return {
        "mean_luminance": float(gray.mean()),
        "dark_pixel_fraction_below_0.20": float(dark.mean()),
        "dark_region_high_frequency_residual_mean": float(residual[dark].mean()),
    }


def tensor_image(image):
    array = image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8))


def backproject(depth, image, intrinsics, valid, max_points, seed):
    h, w = depth.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mask = valid & torch.isfinite(depth) & (depth > 0)
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
    if indices.numel() > max_points:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(indices.numel(), generator=generator)[:max_points]
        indices = indices[order]
    z = depth.reshape(-1)[indices]
    x = xx.reshape(-1)[indices].float()
    y = yy.reshape(-1)[indices].float()
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points = torch.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), dim=1)
    colors = image.permute(1, 2, 0).reshape(-1, 3)[indices]
    return points.numpy().astype(np.float32), np.round(colors.numpy() * 255).astype(np.uint8)


def transform_points(points, transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def build_predicted_pair_geometry(raw_depths, sample, scale, max_points, seed):
    intrinsics = sample["intrinsics"].float()
    points_by_view = []
    colors_by_view = []
    for view_index, depth_key in enumerate(("depth_gt1", "depth_gt2")):
        gt_depth = sample[depth_key].float()
        valid = torch.isfinite(gt_depth) & (gt_depth > 0)
        depth = resize_depth(raw_depths[view_index], gt_depth.shape) * float(scale)
        points, colors = backproject(
            depth,
            sample["images_photo"][view_index].float().clamp(0, 1),
            intrinsics,
            valid,
            max_points // 2,
            seed + view_index,
        )
        points_by_view.append(points)
        colors_by_view.append(colors)

    if "relative_pose" in sample:
        camera1_from_camera2 = np.linalg.inv(sample["relative_pose"].numpy())
        points_by_view[1] = transform_points(points_by_view[1], camera1_from_camera2)
    points = np.concatenate(points_by_view, axis=0)
    colors = np.concatenate(colors_by_view, axis=0)
    return points.astype(np.float32), colors


def build_gt_pair_geometry(sample, max_points, seed):
    intrinsics = sample["intrinsics"].float()
    points_by_view = []
    colors_by_view = []
    for view_index, depth_key in enumerate(("depth_gt1", "depth_gt2")):
        depth = sample[depth_key].float()
        valid = torch.isfinite(depth) & (depth > 0)
        points, colors = backproject(
            depth,
            sample["images_photo"][view_index].float().clamp(0, 1),
            intrinsics,
            valid,
            max_points // 2,
            seed + view_index,
        )
        points_by_view.append(points)
        colors_by_view.append(colors)

    if "relative_pose" in sample:
        camera1_from_camera2 = np.linalg.inv(sample["relative_pose"].numpy())
        points_by_view[1] = transform_points(points_by_view[1], camera1_from_camera2)
    points = np.concatenate(points_by_view, axis=0)
    colors = np.concatenate(colors_by_view, axis=0)
    return points.astype(np.float32), colors


def shared_geometry_volume(gt_points):
    lower, upper = np.quantile(gt_points, (0.005, 0.995), axis=0)
    margin = np.maximum((upper - lower) * 0.12, 1.0e-3)
    return lower - margin, upper + margin


def crop_geometry(points, colors, lower, upper):
    keep = np.isfinite(points).all(axis=1)
    keep &= np.all(points >= lower[None], axis=1)
    keep &= np.all(points <= upper[None], axis=1)
    return points[keep], colors[keep]


def common_depth_colors(points, lo, hi):
    value = np.clip((points[:, 2] - lo) / max(hi - lo, 1.0e-8), 0, 1)
    lut = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS
    )
    lut = cv2.cvtColor(lut, cv2.COLOR_BGR2RGB)[:, 0]
    return lut[np.round(value * 255).astype(np.uint8)]


def replace_render_background(image, replacement):
    array = np.asarray(image.convert("RGB")).copy()
    mask = np.all(array == np.asarray(BACKGROUND, dtype=np.uint8), axis=2)
    array[mask] = np.asarray(replacement, dtype=np.uint8)
    return Image.fromarray(array)


def render_clouds(clouds, gt_points, lo, hi, output, panel_size):
    center, _, camera = camera_views(gt_points)
    specs = []
    for title, subtitle, points in clouds:
        colors = common_depth_colors(points, lo, hi)
        rendered = render_points(
            points, colors, camera, center, panel_size, panel_size, radius=1
        )
        specs.append(
            {"title": title, "subtitle": subtitle, "image": Image.fromarray(rendered)}
        )
    compose_panels(specs, output, panel_size)


def render_pair_geometry_comparison(
    args,
    mast_points,
    night_points,
    gt_points,
    mast_absrel,
    night_absrel,
    metric_label,
):
    geometry_dir = args.outdir / "geometry_single_images"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    depth_lo, depth_hi = np.quantile(gt_points[:, 2], (0.02, 0.98))
    center, front_camera, _ = camera_views(gt_points)
    lower, upper = np.quantile(gt_points, (0.02, 0.98), axis=0)
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-3)
    elevated_camera = center + np.asarray(
        [0.0, -0.42 * extent, -1.12 * extent], dtype=np.float32
    )
    render_size = max(args.panel_size, 520)
    methods = (
        ("MASt3R", mast_points, f"{metric_label} {mast_absrel:.3f}"),
        ("Night3R", night_points, f"{metric_label} {night_absrel:.3f}"),
        ("Ground truth", gt_points, "Reference pair geometry"),
    )
    views = (("front", front_camera), ("elevated", elevated_camera))
    rows = []
    for view_name, camera in views:
        row = []
        for method_name, points, metric_text in methods:
            colors = common_depth_colors(points, depth_lo, depth_hi)
            rendered = Image.fromarray(
                render_points(
                    points,
                    colors,
                    camera,
                    center,
                    render_size,
                    render_size,
                    radius=1,
                )
            )
            filename = f"{method_name.lower().replace(' ', '_')}_{view_name}.png"
            rendered.save(geometry_dir / filename)
            row.append(
                {
                    "title": f"{method_name} | {view_name.capitalize()}",
                    "subtitle": metric_text,
                    "subtitle_color": (91, 224, 176)
                    if method_name == "Night3R"
                    else MUTED,
                    "image": rendered,
                }
            )
        rows.append(row)
    output = args.outdir / "comparison_pair_geometry_same_view.png"
    compose_pair_grid(rows, output, render_size)
    elevated_output = args.outdir / "comparison_pair_geometry_elevated_same_view.png"
    compose_panels(rows[1], elevated_output, render_size)

    white_background = (255, 255, 255)
    white_text = (20, 24, 29)
    white_muted = (75, 83, 94)
    white_night = (0, 122, 82)
    white_dir = args.outdir / "geometry_single_images_white"
    white_dir.mkdir(parents=True, exist_ok=True)
    white_rows = []
    for row in rows:
        white_row = []
        for spec in row:
            white_image = replace_render_background(spec["image"], white_background)
            white_image.save(
                white_dir
                / f"{spec['title'].lower().replace(' | ', '_').replace(' ', '_')}.png"
            )
            white_row.append(
                {
                    **spec,
                    "image": white_image,
                    "subtitle_color": white_night
                    if spec["title"].startswith("Night3R")
                    else white_muted,
                }
            )
        white_rows.append(white_row)
    white_output = args.outdir / "comparison_pair_geometry_same_view_white.png"
    compose_pair_grid(
        white_rows,
        white_output,
        render_size,
        background=white_background,
        text_color=white_text,
        muted_color=white_muted,
    )
    white_elevated_output = (
        args.outdir / "comparison_pair_geometry_elevated_same_view_white.png"
    )
    compose_panels(
        white_rows[1],
        white_elevated_output,
        render_size,
        background=white_background,
        text_color=white_text,
        muted_color=white_muted,
    )
    return {
        "comparison": output.name,
        "elevated_comparison": elevated_output.name,
        "white_comparison": white_output.name,
        "white_elevated_comparison": white_elevated_output.name,
        "single_images": geometry_dir.name,
        "white_single_images": white_dir.name,
        "shared_depth_range": [float(depth_lo), float(depth_hi)],
        "camera_center": center.tolist(),
        "front_camera": front_camera.tolist(),
        "elevated_camera": elevated_camera.tolist(),
    }


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    split_audit = None
    if args.split_audit_file is not None:
        split_audit = json.loads(args.split_audit_file.read_text(encoding="utf-8"))
        if split_audit.get("source_sequence_overlap") is not False:
            raise RuntimeError(
                f"Split audit does not prove source disjointness: {args.split_audit_file}"
            )
        if args.source_sequence and split_audit.get("source_sequence") != args.source_sequence:
            raise RuntimeError(
                "Split-audit source sequence does not match --source-sequence: "
                f"{split_audit.get('source_sequence')} != {args.source_sequence}"
            )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_config(args.config)
    sample, batch = load_pair(args, cfg)
    device = torch.device(args.device)

    mast3r_raw_pair = infer_mast3r(args, cfg, batch, device)
    night3r_raw_pair, load_info = infer_night3r(args, cfg, batch, device)
    gt_pair = [sample["depth_gt1"].float(), sample["depth_gt2"].float()]
    if args.exposure is None:
        rgb_pair = [
            Image.open(sample["image1_path"]).convert("RGB"),
            Image.open(sample["image2_path"]).convert("RGB"),
        ]
    else:
        rgb_pair = [tensor_image(image) for image in sample["images_photo_dark"]]
    event_pair = [
        load_event(sample["event1_path"]),
        load_event(sample["event2_path"]),
    ]
    event_images = [event_red_blue(event, background="black")[0] for event in event_pair]
    singles_root = args.outdir / "single_images"
    view_results = []

    for view_index in range(2):
        view_number = view_index + 1
        view_dir = singles_root / f"view{view_number:02d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        gt = gt_pair[view_index]
        valid = torch.isfinite(gt) & (gt > 0)
        mast3r_raw = resize_depth(mast3r_raw_pair[view_index], gt.shape)
        night3r_raw = resize_depth(night3r_raw_pair[view_index], gt.shape)
        mast3r_metrics, mast3r_aligned = evaluate_depth(mast3r_raw, gt, valid)
        night3r_metrics, night3r_aligned = evaluate_depth(night3r_raw, gt, valid)
        lo, hi = depth_range(gt, valid)
        mast3r_vis = depth_image(mast3r_aligned, valid, lo, hi)
        night3r_vis = depth_image(night3r_aligned, valid, lo, hi)
        gt_vis = depth_image(gt, valid, lo, hi)
        mast_error_values = (mast3r_aligned - gt).abs()[valid].numpy()
        night_error_values = (night3r_aligned - gt).abs()[valid].numpy()
        error_hi = float(
            np.percentile(
                np.concatenate((mast_error_values, night_error_values)), 95
            )
        )
        mast_error_vis = error_image(mast3r_aligned, gt, valid, error_hi)
        night_error_vis = error_image(night3r_aligned, gt, valid, error_hi)

        rgb_pair[view_index].save(view_dir / "01_RGB_native.png")
        event_images[view_index].save(view_dir / "02_Event_recent4bins_red_blue.png")
        mast3r_vis.save(view_dir / "03_MASt3R_depth_median_aligned.png")
        night3r_vis.save(view_dir / "04_Night3R_depth_median_aligned.png")
        gt_vis.save(view_dir / "05_GT_depth.png")
        mast_error_vis.save(view_dir / "06_MASt3R_error.png")
        night_error_vis.save(view_dir / "07_Night3R_error.png")
        np.save(view_dir / "MASt3R_depth_raw.npy", mast3r_raw.numpy())
        np.save(view_dir / "MASt3R_depth_median_aligned.npy", mast3r_aligned.numpy())
        np.save(view_dir / "Night3R_depth_raw.npy", night3r_raw.numpy())
        np.save(view_dir / "Night3R_depth_median_aligned.npy", night3r_aligned.numpy())
        np.save(view_dir / "GT_depth.npy", gt.numpy())
        np.save(view_dir / "depth_valid.npy", valid.numpy())
        view_results.append(
            {
                "view": view_number,
                "rgb": rgb_pair[view_index],
                "event": event_images[view_index],
                "gt": gt,
                "valid": valid,
                "mast3r_raw": mast3r_raw,
                "night3r_raw": night3r_raw,
                "mast3r_aligned": mast3r_aligned,
                "night3r_aligned": night3r_aligned,
                "mast3r_metrics": mast3r_metrics,
                "night3r_metrics": night3r_metrics,
                "mast3r_vis": mast3r_vis,
                "night3r_vis": night3r_vis,
                "gt_vis": gt_vis,
                "mast_error_vis": mast_error_vis,
                "night_error_vis": night_error_vis,
                "depth_range": [lo, hi],
                "error_range": [0.0, error_hi],
            }
        )

    first = view_results[0]
    rgb1, rgb2 = rgb_pair
    event_image = event_images[0]
    gt = first["gt"]
    valid = first["valid"]
    mast3r_raw = first["mast3r_raw"]
    night3r_raw = first["night3r_raw"]
    mast3r_aligned = first["mast3r_aligned"]
    night3r_aligned = first["night3r_aligned"]
    mast3r_metrics = first["mast3r_metrics"]
    night3r_metrics = first["night3r_metrics"]
    mast3r_vis = first["mast3r_vis"]
    night3r_vis = first["night3r_vis"]
    gt_vis = first["gt_vis"]
    lo, hi = first["depth_range"]

    # Keep the original top-level names as convenient view-1 aliases.
    rgb1.save(args.outdir / "01_mvsec_rgb_view1_native.png")
    rgb2.save(args.outdir / "02_mvsec_rgb_view2_native.png")
    event_image.save(args.outdir / "03_event_recent4bins_red_blue.png")
    mast3r_vis.save(args.outdir / "04_mast3r_depth_median_aligned.png")
    night3r_vis.save(args.outdir / "05_night3r_depth_median_aligned.png")
    gt_vis.save(args.outdir / "06_gt_depth.png")
    np.save(args.outdir / "mast3r_depth_raw.npy", mast3r_raw.numpy())
    np.save(args.outdir / "mast3r_depth_median_aligned.npy", mast3r_aligned.numpy())
    np.save(args.outdir / "night3r_depth_raw.npy", night3r_raw.numpy())
    np.save(args.outdir / "night3r_depth_median_aligned.npy", night3r_aligned.numpy())
    np.save(args.outdir / "gt_depth.npy", gt.numpy())
    np.save(args.outdir / "depth_valid.npy", valid.numpy())

    pair_depth_rows = []
    pair_error_rows = []
    for result in view_results:
        view_label = f"View {result['view']}"
        pair_depth_rows.append(
            [
                {
                    "title": f"{view_label} RGB",
                    "subtitle": (
                        f"{args.dataset_name}, exposure {args.exposure:.2f}"
                        if args.exposure is not None
                        else f"original {args.dataset_name}"
                    ),
                    "image": result["rgb"],
                },
                {"title": f"{view_label} events", "subtitle": "4 temporal bins", "image": result["event"]},
                {
                    "title": f"{view_label} MASt3R",
                    "subtitle": f"Aligned AbsRel {result['mast3r_metrics']['median_aligned_absrel']:.3f}",
                    "image": result["mast3r_vis"],
                },
                {
                    "title": f"{view_label} Night3R",
                    "subtitle": f"Aligned AbsRel {result['night3r_metrics']['median_aligned_absrel']:.3f}",
                    "subtitle_color": (91, 224, 176),
                    "image": result["night3r_vis"],
                },
                {"title": f"{view_label} GT", "subtitle": "metric depth", "image": result["gt_vis"]},
            ]
        )
        pair_error_rows.append(
            [
                {"title": f"{view_label} GT", "subtitle": "metric depth", "image": result["gt_vis"]},
                {
                    "title": f"{view_label} MASt3R error",
                    "subtitle": f"Aligned AbsRel {result['mast3r_metrics']['median_aligned_absrel']:.3f}",
                    "image": result["mast_error_vis"],
                },
                {
                    "title": f"{view_label} Night3R error",
                    "subtitle": f"Aligned AbsRel {result['night3r_metrics']['median_aligned_absrel']:.3f}",
                    "image": result["night_error_vis"],
                },
            ]
        )
    compose_pair_grid(
        pair_depth_rows,
        args.outdir / "comparison_pair_depths.png",
        args.panel_size,
    )
    compose_pair_grid(
        pair_error_rows,
        args.outdir / "comparison_pair_errors.png",
        args.panel_size,
    )

    compose_panels(
        pair_depth_rows[0], args.outdir / "comparison_depth_teaser.png", args.panel_size
    )
    compose_panels(
        pair_error_rows[0], args.outdir / "comparison_error_maps.png", args.panel_size
    )

    intrinsics = sample["intrinsics"].float()
    image_tensor = sample["images_photo"][0].float().clamp(0, 1)
    gt_points, gt_rgb = backproject(gt, image_tensor, intrinsics, valid, args.max_points, args.seed)
    mast_points, mast_rgb = backproject(
        mast3r_aligned, image_tensor, intrinsics, valid, args.max_points, args.seed + 1
    )
    night_points, night_rgb = backproject(
        night3r_aligned, image_tensor, intrinsics, valid, args.max_points, args.seed + 2
    )
    write_ply(args.outdir / "mast3r_cloud_median_aligned.ply", mast_points, mast_rgb)
    write_ply(args.outdir / "night3r_cloud_median_aligned.ply", night_points, night_rgb)
    write_ply(args.outdir / "gt_cloud.ply", gt_points, gt_rgb)
    render_clouds(
        [
            (
                "MASt3R",
                f"Aligned AbsRel {mast3r_metrics['median_aligned_absrel']:.3f}",
                mast_points,
            ),
            (
                "Night3R",
                f"Aligned AbsRel {night3r_metrics['median_aligned_absrel']:.3f}",
                night_points,
            ),
            ("Ground truth", "same camera view", gt_points),
        ],
        gt_points,
        lo,
        hi,
        args.outdir / "comparison_pointcloud_same_camera.png",
        args.panel_size,
    )

    reduction = 100.0 * (
        mast3r_metrics["median_aligned_absrel"]
        - night3r_metrics["median_aligned_absrel"]
    ) / max(mast3r_metrics["median_aligned_absrel"], 1.0e-8)
    view_evaluations = []
    for result in view_results:
        view_reduction = 100.0 * (
            result["mast3r_metrics"]["median_aligned_absrel"]
            - result["night3r_metrics"]["median_aligned_absrel"]
        ) / max(result["mast3r_metrics"]["median_aligned_absrel"], 1.0e-8)
        view_evaluations.append(
            {
                "view": result["view"],
                "mast3r": result["mast3r_metrics"],
                "night3r": result["night3r_metrics"],
                "aligned_absrel_reduction_percent": view_reduction,
                "single_image_directory": f"single_images/view{result['view']:02d}",
            }
        )
    pair_mean_mast3r_absrel = float(
        np.mean(
            [item["mast3r"]["median_aligned_absrel"] for item in view_evaluations]
        )
    )
    pair_mean_night3r_absrel = float(
        np.mean(
            [item["night3r"]["median_aligned_absrel"] for item in view_evaluations]
        )
    )
    pair_mean_mast3r_raw_absrel = float(
        np.mean([item["mast3r"]["raw_absrel"] for item in view_evaluations])
    )
    pair_mean_night3r_raw_absrel = float(
        np.mean([item["night3r"]["raw_absrel"] for item in view_evaluations])
    )

    if args.geometry_scale_mode == "raw":
        mast_geometry_scale = 1.0
        night_geometry_scale = 1.0
        geometry_mast_absrel = pair_mean_mast3r_raw_absrel
        geometry_night_absrel = pair_mean_night3r_raw_absrel
        geometry_metric_label = "Raw AbsRel"
        geometry_scale_description = "no scale alignment"
    else:
        mast_geometry_scale = mast3r_metrics["median_scale"]
        night_geometry_scale = night3r_metrics["median_scale"]
        geometry_mast_absrel = pair_mean_mast3r_absrel
        geometry_night_absrel = pair_mean_night3r_absrel
        geometry_metric_label = "Aligned AbsRel"
        geometry_scale_description = (
            "one view-1 median scale per method, shared by both depth maps"
        )

    mast_pair_points, mast_pair_rgb = build_predicted_pair_geometry(
        mast3r_raw_pair,
        sample,
        mast_geometry_scale,
        args.max_points,
        args.seed + 10,
    )
    night_pair_points, night_pair_rgb = build_predicted_pair_geometry(
        night3r_raw_pair,
        sample,
        night_geometry_scale,
        args.max_points,
        args.seed + 20,
    )
    gt_pair_points, gt_pair_rgb = build_gt_pair_geometry(
        sample, args.max_points, args.seed + 30
    )
    volume_lower, volume_upper = shared_geometry_volume(gt_pair_points)
    mast_render_points, mast_render_rgb = crop_geometry(
        mast_pair_points, mast_pair_rgb, volume_lower, volume_upper
    )
    night_render_points, night_render_rgb = crop_geometry(
        night_pair_points, night_pair_rgb, volume_lower, volume_upper
    )
    gt_render_points, gt_render_rgb = crop_geometry(
        gt_pair_points, gt_pair_rgb, volume_lower, volume_upper
    )
    write_ply(
        args.outdir / "mast3r_pair_geometry_view1_frame.ply",
        mast_pair_points,
        mast_pair_rgb,
    )
    write_ply(
        args.outdir / "night3r_pair_geometry_view1_frame.ply",
        night_pair_points,
        night_pair_rgb,
    )
    write_ply(args.outdir / "gt_pair_geometry_view1_frame.ply", gt_pair_points, gt_pair_rgb)
    geometry_outputs = render_pair_geometry_comparison(
        args,
        mast_render_points,
        night_render_points,
        gt_render_points,
        geometry_mast_absrel,
        geometry_night_absrel,
        geometry_metric_label,
    )
    geometry_outputs.update(
        {
            "coordinate_frame": "view 1",
            "geometry_source": (
                "ordered camera-local depth predictions fused with the shared GT relative pose"
            ),
            "scale_alignment": geometry_scale_description,
            "metric_type": geometry_metric_label,
            "support_mask": "same finite positive GT depth support for both methods",
            "render_volume_lower": volume_lower.tolist(),
            "render_volume_upper": volume_upper.tolist(),
            "full_point_counts": {
                "mast3r": int(len(mast_pair_points)),
                "night3r": int(len(night_pair_points)),
                "ground_truth": int(len(gt_pair_points)),
            },
            "rendered_point_counts": {
                "mast3r": int(len(mast_render_points)),
                "night3r": int(len(night_render_points)),
                "ground_truth": int(len(gt_render_points)),
            },
        }
    )
    manifest = {
        "sample_id": sample["sample_id"],
        "dataset_name": args.dataset_name,
        "source_sequence": args.source_sequence,
        "selection_reason": (
            "Pre-registered temporal position on a source-disjoint test sequence; "
            "the pair was selected without inspecting model metrics."
        ),
        "input_protocol": (
            f"{args.dataset_name} RGB with fixed exposure {args.exposure:.2f}, "
            "read noise 0.01, and shot noise"
            if args.exposure is not None
            else f"original {args.dataset_name} RGB without additional darkening"
        ),
        "split_audit": {
            "status": "pass" if split_audit else "not_provided",
            "source_sequence": args.source_sequence,
            "source_sequence_overlap": False if split_audit else None,
            "audit_file": (
                str(args.split_audit_file.resolve()) if args.split_audit_file else None
            ),
            "allowed_use": "strict scene-held-out qualitative evaluation",
        },
        "source_image_size_wh": list(rgb1.size),
        "event_representation": "4 temporal bins x 2 polarities; red/blue display",
        "depth_colormap": "Viridis; near=purple, far=yellow; per-scene GT range",
        "image_statistics": image_statistics(rgb1),
        "depth_evaluation": {
            "alignment": "independent camera-local median scale for each ordered view",
            "valid_mask": "finite positive MVSEC depth GT",
            "mast3r": mast3r_metrics,
            "night3r": night3r_metrics,
            "aligned_absrel_reduction_percent": reduction,
            "views": view_evaluations,
            "pair_view_mean": {
                "mast3r_raw_absrel": pair_mean_mast3r_raw_absrel,
                "night3r_raw_absrel": pair_mean_night3r_raw_absrel,
                "mast3r_median_aligned_absrel": pair_mean_mast3r_absrel,
                "night3r_median_aligned_absrel": pair_mean_night3r_absrel,
            },
        },
        "pair_geometry": geometry_outputs,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_load": load_info,
        "backbone": str(args.backbone.resolve()),
        "outputs": {
            "depth_teaser": "comparison_depth_teaser.png",
            "error_maps": "comparison_error_maps.png",
            "pair_depths": "comparison_pair_depths.png",
            "pair_errors": "comparison_pair_errors.png",
            "single_images": "single_images/view01 and single_images/view02",
            "pointcloud_same_camera": "comparison_pointcloud_same_camera.png",
            "pair_geometry_same_view": "comparison_pair_geometry_same_view.png",
            "pair_geometry_elevated_same_view": (
                "comparison_pair_geometry_elevated_same_view.png"
            ),
            "pair_geometry_same_view_white": (
                "comparison_pair_geometry_same_view_white.png"
            ),
            "pair_geometry_elevated_same_view_white": (
                "comparison_pair_geometry_elevated_same_view_white.png"
            ),
            "pair_geometry_single_images": "geometry_single_images",
            "pair_geometry_single_images_white": "geometry_single_images_white",
        },
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.outdir / "README.md").write_text(
        f"# Final {args.dataset_name} showcase\n\n"
        "This folder compares MASt3R and the final MVSEC-1k Night3R checkpoint on\n"
        f"`{sample['sample_id']}`. Input protocol: {manifest['input_protocol']}.\n\n"
        "All standalone pair assets are under `single_images/view01` and\n"
        "`single_images/view02`. Each view contains RGB, event, MASt3R depth,\n"
        "Night3R depth, GT depth, MASt3R error, and Night3R error images. View 2\n"
        "is inferred from the reversed pair so its depth remains camera-local.\n\n"
        "`comparison_pair_geometry_same_view.png` fuses the two ordered,\n"
        "camera-local depth predictions into the view-1 coordinate frame using\n"
        "the shared GT relative pose. MASt3R,\n"
        "Night3R, and GT share the same front/elevated cameras, GT-defined render\n"
        f"volume, and depth color range. Geometry scale policy: {geometry_scale_description}. "
        "Standalone renders\n"
        "are in `geometry_single_images/`, and the full pair clouds are saved as\n"
        "PLY files.\n\n"
        f"Split audit: {manifest['split_audit']}.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
