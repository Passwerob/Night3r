#!/usr/bin/env python3
"""Render a synchronized orbit view for an exported Night3R/GT cloud pair."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#0c1118"
TEXT = "#f0f4f8"
SUBTEXT = "#aab7c4"
LIGHT_BACKGROUND = "#f5f7fa"
LIGHT_TEXT = "#1e2936"
LIGHT_SUBTEXT = "#5f6f80"


def parse_args():
    parser = argparse.ArgumentParser(
        "Render a synchronized Night3R and sensor-depth point-cloud orbit."
    )
    parser.add_argument("--night3r", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--night-points-key",
        default="points",
        help="NPZ key for the Night3R XYZ array when using a geometry export.",
    )
    parser.add_argument(
        "--night-depth-key",
        default=None,
        help="2D predicted-depth key used to make a single-view RGB point cloud.",
    )
    parser.add_argument(
        "--night-valid-key",
        default="valid",
        help="Optional 2D validity-mask key accompanying --night-depth-key.",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=2,
        help="Pixel stride used when backprojecting a predicted depth map.",
    )
    parser.add_argument(
        "--night-rgb-image",
        type=Path,
        default=None,
        help="Reference RGB image used to texture an XYZ-only Night3R cloud.",
    )
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=None,
        help="JSON file containing an 'intrinsics' 3x3 matrix for RGB reprojection.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-points", type=int, default=24000)
    parser.add_argument(
        "--display-gamma",
        type=float,
        default=1.0,
        help="Display-only RGB gamma exponent; values below 1 brighten dark textures.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--start-azim", type=float, default=-66.0)
    parser.add_argument("--orbit-degrees", type=float, default=164.0)
    parser.add_argument(
        "--sample-label",
        default="Point-cloud reconstruction",
        help="Small top-line label rendered above the synchronized orbit.",
    )
    parser.add_argument(
        "--ground-truth-title",
        default="Sensor-depth point cloud (GT)",
    )
    parser.add_argument(
        "--single-panel",
        action="store_true",
        help="Render only the Night3R reconstruction for a presentation-focused demo.",
    )
    parser.add_argument(
        "--light-background",
        action="store_true",
        help="Use a light presentation background for RGB-colored point clouds.",
    )
    parser.add_argument(
        "--renderer",
        choices=("matplotlib", "raster"),
        default="matplotlib",
        help="Use depth-sorted rasterization for an opaque RGB point-cloud render.",
    )
    parser.add_argument(
        "--camera-space",
        action="store_true",
        help="Start rasterized RGB depth clouds from their source camera and apply a small yaw.",
    )
    return parser.parse_args()


def load_cloud(path, max_points, rng, display_gamma, points_key="points", rgb_image=None, intrinsics=None):
    arrays = np.load(path)
    if points_key not in arrays:
        raise KeyError(f"{path} has no {points_key!r} array")
    points = np.asarray(arrays[points_key], dtype=np.float32)
    if "colors" in arrays:
        colors = np.asarray(arrays["colors"], dtype=np.float32) / 255.0
    else:
        if rgb_image is None or intrinsics is None:
            raise ValueError(
                f"{path} has no colors; provide --night-rgb-image and --intrinsics-json"
            )
        image = np.asarray(Image.open(rgb_image).convert("RGB"), dtype=np.float32) / 255.0
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        depth = points[:, 2]
        u = np.rint(fx * points[:, 0] / np.maximum(depth, 1.0e-6) + cx).astype(np.int32)
        v = np.rint(fy * points[:, 1] / np.maximum(depth, 1.0e-6) + cy).astype(np.int32)
        projected = (
            np.isfinite(depth)
            & (depth > 1.0e-6)
            & (u >= 0)
            & (u < image.shape[1])
            & (v >= 0)
            & (v < image.shape[0])
        )
        colors = np.zeros((len(points), 3), dtype=np.float32)
        colors[projected] = image[v[projected], u[projected]]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    if "projected" in locals():
        valid &= projected
    points = points[valid]
    colors = np.clip(colors[valid], 0.0, 1.0) ** display_gamma
    if len(points) == 0:
        raise RuntimeError(f"No valid points in {path}")
    if len(points) > max_points:
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def load_intrinsics(path):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "intrinsics" not in payload:
        raise KeyError(f"{path} has no 'intrinsics' entry")
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsics matrix in {path}")
    return intrinsics


def load_rgb_depth_cloud(path, depth_key, valid_key, rgb_image, intrinsics, stride, display_gamma):
    if rgb_image is None or intrinsics is None:
        raise ValueError("A depth cloud requires --night-rgb-image and --intrinsics-json")
    arrays = np.load(path)
    if depth_key not in arrays:
        raise KeyError(f"{path} has no {depth_key!r} array")
    depth = np.asarray(arrays[depth_key], dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"{depth_key!r} in {path} must be a 2D depth map")
    image = np.asarray(Image.open(rgb_image).convert("RGB"), dtype=np.float32) / 255.0
    if image.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB image shape {image.shape[:2]} does not match depth shape {depth.shape}"
        )
    valid = np.isfinite(depth) & (depth > 1.0e-6)
    if valid_key in arrays:
        valid &= np.asarray(arrays[valid_key], dtype=bool)
    if not np.any(valid):
        raise RuntimeError(f"No valid depth values in {path}:{depth_key}")
    max_depth = float(np.quantile(depth[valid], 0.995))
    valid &= depth <= max_depth
    yy, xx = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    sampled_valid = valid[::stride, ::stride]
    sampled_depth = depth[::stride, ::stride]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points = np.stack(
        (
            (xx.astype(np.float32) - cx) * sampled_depth / fx,
            (yy.astype(np.float32) - cy) * sampled_depth / fy,
            sampled_depth,
        ),
        axis=-1,
    )[sampled_valid]
    colors = image[::stride, ::stride][sampled_valid]
    return points.astype(np.float32), np.clip(colors, 0.0, 1.0) ** display_gamma


def shared_limits(*clouds):
    points = np.concatenate(clouds, axis=0)
    lower = np.percentile(points, 1.0, axis=0)
    upper = np.percentile(points, 99.0, axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower) * 0.55), 1.0e-3)
    return center, radius


def style_axis(axis, center, radius, title, theme):
    axis.set_facecolor(theme["background"])
    axis.set_proj_type("persp")
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[2] - radius, center[2] + radius)
    axis.set_zlim(center[1] - radius, center[1] + radius)
    axis.set_box_aspect((1.0, 1.0, 0.82), zoom=1.48)
    axis.set_axis_off()
    axis.set_title(title, color=theme["text"], fontsize=25, pad=16)


def load_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def rotate_about_vertical(points, center, angle_radians):
    rotated = points - center[None]
    cos_angle, sin_angle = math.cos(angle_radians), math.sin(angle_radians)
    x = rotated[:, 0].copy()
    z = rotated[:, 2].copy()
    rotated[:, 0] = cos_angle * x - sin_angle * z
    rotated[:, 2] = sin_angle * x + cos_angle * z
    return rotated + center[None]


def rasterize_rgb_cloud(points, colors, width, height, background, point_radius=1):
    canvas = np.broadcast_to(background, (height, width, 3)).copy()
    lower = np.percentile(points, 1.0, axis=0)
    upper = np.percentile(points, 99.0, axis=0)
    center = (lower + upper) * 0.5
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-6)
    camera = center + np.asarray([0.40, 0.24, -1.18], dtype=np.float32) * extent
    forward = center - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)

    relative = points - camera[None]
    x, y, z = relative @ right, relative @ up, relative @ forward
    focal = 0.5 * width / np.tan(np.deg2rad(48.0) * 0.5)
    u = focal * x / np.maximum(z, 1.0e-6) + 0.5 * width
    v = 0.5 * height - focal * y / np.maximum(z, 1.0e-6)
    valid = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -point_radius) & (u < width + point_radius)
    valid &= (v >= -point_radius) & (v < height + point_radius)
    if not np.any(valid):
        return canvas

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    camera_depth = z[valid]
    visible_colors = colors[valid]
    luminance = visible_colors @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    white = max(float(np.percentile(luminance, 98.0)), 0.05)
    visible_colors = np.clip(visible_colors / white, 0.0, 1.0) ** 0.72
    visible_colors = (visible_colors * 255.0 + 0.5).astype(np.uint8)
    order = np.argsort(camera_depth)[::-1]
    u, v, visible_colors = u[order], v[order], visible_colors[order]
    for delta_y in range(-point_radius, point_radius + 1):
        for delta_x in range(-point_radius, point_radius + 1):
            pixel_x = u + delta_x
            pixel_y = v + delta_y
            inside = (
                (pixel_x >= 0)
                & (pixel_x < width)
                & (pixel_y >= 0)
                & (pixel_y < height)
            )
            canvas[pixel_y[inside], pixel_x[inside]] = visible_colors[inside]
    return canvas


def rasterize_camera_space_cloud(
    points, colors, intrinsics, source_size, panel_size, background, yaw_radians, point_radius=1
):
    canvas = np.broadcast_to(background, (panel_size, panel_size, 3)).copy()
    cos_angle, sin_angle = math.cos(yaw_radians), math.sin(yaw_radians)
    x = cos_angle * points[:, 0] - sin_angle * points[:, 2]
    z = sin_angle * points[:, 0] + cos_angle * points[:, 2]
    y = points[:, 1]
    scale_x = panel_size / float(source_size[0])
    scale_y = panel_size / float(source_size[1])
    u = float(intrinsics[0, 0]) * scale_x * x / np.maximum(z, 1.0e-6) + float(intrinsics[0, 2]) * scale_x
    v = float(intrinsics[1, 1]) * scale_y * y / np.maximum(z, 1.0e-6) + float(intrinsics[1, 2]) * scale_y
    valid = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -point_radius) & (u < panel_size + point_radius)
    valid &= (v >= -point_radius) & (v < panel_size + point_radius)
    if not np.any(valid):
        return canvas
    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    visible_colors = np.clip(colors[valid], 0.0, 1.0) ** 0.82
    visible_colors = (visible_colors * 255.0 + 0.5).astype(np.uint8)
    order = np.argsort(z[valid])[::-1]
    u, v, visible_colors = u[order], v[order], visible_colors[order]
    for delta_y in range(-point_radius, point_radius + 1):
        for delta_x in range(-point_radius, point_radius + 1):
            pixel_x = u + delta_x
            pixel_y = v + delta_y
            inside = (
                (pixel_x >= 0)
                & (pixel_x < panel_size)
                & (pixel_y >= 0)
                & (pixel_y < panel_size)
            )
            canvas[pixel_y[inside], pixel_x[inside]] = visible_colors[inside]
    return canvas


def render_raster_frame(
    path,
    cloud,
    center,
    orbit_angle,
    width,
    height,
    sample_label,
    theme,
    camera_space,
    intrinsics,
    source_size,
):
    points, colors, title = cloud
    header_height = 150
    background = np.asarray(
        [int(theme["background"][offset : offset + 2], 16) for offset in (1, 3, 5)],
        dtype=np.uint8,
    )
    if camera_space:
        if intrinsics is None or source_size is None:
            raise ValueError("--camera-space requires --intrinsics-json and --night-rgb-image")
        panel_size = min(height - header_height - 20, width - 80)
        raster = rasterize_camera_space_cloud(
            points,
            colors,
            intrinsics,
            source_size,
            panel_size,
            background,
            orbit_angle,
            point_radius=2,
        )
        raster_position = ((width - panel_size) // 2, header_height + 10)
    else:
        rotated_points = rotate_about_vertical(points, center, orbit_angle)
        raster = rasterize_rgb_cloud(
            rotated_points,
            colors,
            width,
            height - header_height,
            background,
            point_radius=1,
        )
        raster_position = (0, header_height)
    image = Image.new("RGB", (width, height), tuple(background.tolist()))
    image.paste(Image.fromarray(raster), raster_position)
    draw = ImageDraw.Draw(image)
    draw.text(
        (width // 2, 44),
        f"{sample_label}   |   RGB-textured orbit",
        fill=theme["subtext"],
        font=load_font(25),
        anchor="mm",
    )
    draw.text(
        (width // 2, 96),
        title,
        fill=theme["text"],
        font=load_font(38),
        anchor="mm",
    )
    image.save(path)


def render_frame(
    path,
    clouds,
    center,
    radius,
    azimuth,
    elevation,
    width,
    height,
    sample_label,
    theme,
    renderer,
    start_azimuth,
    camera_space,
    intrinsics,
    source_size,
):
    if renderer == "raster":
        if len(clouds) != 1:
            raise ValueError("The raster renderer currently requires --single-panel")
        render_raster_frame(
            path,
            clouds[0],
            center,
            math.radians(azimuth - start_azimuth),
            width,
            height,
            sample_label,
            theme,
            camera_space,
            intrinsics,
            source_size,
        )
        return
    figure = plt.figure(
        figsize=(width / 100.0, height / 100.0), dpi=100, facecolor=theme["background"]
    )
    figure.subplots_adjust(left=0.012, right=0.988, bottom=0.055, top=0.89, wspace=0.025)
    figure.text(
        0.5,
        0.965,
        f"{sample_label}   |   synchronized camera orbit",
        color=theme["subtext"],
        fontsize=17,
        ha="center",
        va="top",
    )
    for index, (points, colors, title) in enumerate(clouds, start=1):
        axis = figure.add_subplot(1, len(clouds), index, projection="3d")
        style_axis(axis, center, radius, title, theme)
        axis.scatter(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            s=2.2,
            c=colors,
            marker=".",
            alpha=0.92,
            linewidths=0,
            rasterized=True,
        )
        axis.view_init(elev=elevation, azim=azimuth)
    figure.savefig(path, facecolor=theme["background"])
    plt.close(figure)


def main():
    args = parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.max_points < 1:
        raise ValueError("--max-points must be positive")
    if args.depth_stride < 1:
        raise ValueError("--depth-stride must be positive")
    if args.display_gamma <= 0:
        raise ValueError("--display-gamma must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    intrinsics = load_intrinsics(args.intrinsics_json)
    source_size = Image.open(args.night_rgb_image).size if args.night_rgb_image else None
    if args.night_depth_key:
        night_points, night_colors = load_rgb_depth_cloud(
            args.night3r,
            args.night_depth_key,
            args.night_valid_key,
            args.night_rgb_image,
            intrinsics,
            args.depth_stride,
            args.display_gamma,
        )
    else:
        night_points, night_colors = load_cloud(
            args.night3r,
            args.max_points,
            rng,
            args.display_gamma,
            points_key=args.night_points_key,
            rgb_image=args.night_rgb_image,
            intrinsics=intrinsics,
        )
    clouds = [
        (night_points, night_colors, "Night3R reconstruction"),
    ]
    if not args.single_panel:
        gt_points, gt_colors = load_cloud(
            args.ground_truth, args.max_points, rng, args.display_gamma
        )
        clouds.append((gt_points, gt_colors, args.ground_truth_title))
    center, radius = shared_limits(*(points for points, _, _ in clouds))
    theme = {
        "background": LIGHT_BACKGROUND if args.light_background else BACKGROUND,
        "text": LIGHT_TEXT if args.light_background else TEXT,
        "subtext": LIGHT_SUBTEXT if args.light_background else SUBTEXT,
    }
    for frame_index in range(args.frames):
        phase = frame_index / (args.frames - 1)
        azimuth = args.start_azim + args.orbit_degrees * phase
        elevation = 14.0 + 7.0 * np.sin(np.pi * phase)
        render_frame(
            args.outdir / f"frame_{frame_index:03d}.png",
            clouds,
            center,
            radius,
            azimuth,
            elevation,
            args.width,
            args.height,
            args.sample_label,
            theme,
            args.renderer,
            args.start_azim,
            args.camera_space,
            intrinsics,
            source_size,
        )
    metadata = {
        "night3r_source": str(args.night3r),
        "ground_truth_source": str(args.ground_truth),
        "night_points_key": args.night_points_key,
        "night_depth_key": args.night_depth_key,
        "night_valid_key": args.night_valid_key,
        "depth_stride": args.depth_stride,
        "night_rgb_image": str(args.night_rgb_image) if args.night_rgb_image else None,
        "intrinsics_json": str(args.intrinsics_json) if args.intrinsics_json else None,
        "frames": args.frames,
        "max_points_per_cloud": args.max_points,
        "coordinate_limits_center_xyz": center.tolist(),
        "shared_radius": radius,
        "camera_orbit": {
            "start_azimuth_degrees": args.start_azim,
            "orbit_degrees": args.orbit_degrees,
            "elevation_range_degrees": [14.0, 21.0],
        },
        "sample_label": args.sample_label,
        "ground_truth_title": args.ground_truth_title,
        "single_panel": args.single_panel,
        "light_background": args.light_background,
        "display_gamma": args.display_gamma,
        "renderer": args.renderer,
        "camera_space": args.camera_space,
    }
    (args.outdir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
