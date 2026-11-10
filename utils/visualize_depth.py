#!/usr/bin/env python3
"""
Visualize sparse depth maps exported from Dark3R.

This script loads depth .npy files and creates visualizations using a viridis colormap.
Optionally creates a video from the depth visualizations.
"""

import os
import argparse
import numpy as np
import gzip
import matplotlib.pyplot as plt
import subprocess
import tqdm


def load_depth(path):
    """
    Load a depth array from a .npy or .npy.gz file.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return np.load(f)
    else:
        return np.load(path)


def find_depth_files(directory):
    """
    Return a list of all .npy and .npy.gz files in the given directory (non-recursive).
    """
    exts = (".npy", ".npy.gz")
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(exts)
    ]


def compute_global_percentiles(files, lower=5, upper=95):
    """
    Compute the global lower and upper percentiles over all files,
    showing progress and printing the results.
    """
    samples = []
    for fp in tqdm.tqdm(files, desc="Calculating percentiles"):
        arr = load_depth(fp)
        samples.append(arr.flatten())
    all_data = np.concatenate(samples)
    lo = np.percentile(all_data, lower)
    hi = np.percentile(all_data, upper)
    min_val, max_val = np.min(all_data), np.max(all_data)
    print(f"Global min: {min_val:.2f}, max: {max_val:.2f}")
    print(f"Computed global percentiles: {lower}th={lo:.2f}, {upper}th={hi:.2f}")
    return lo, hi


def main():
    parser = argparse.ArgumentParser(
        description="Visualize depth files with viridis colormap and optionally create a video."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Folder containing .npy or .npy.gz depth files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Folder to save .png visualizations (and video if requested)",
        default=None,
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum depth value for colormap scaling",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum depth value for colormap scaling",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Frame rate for video if --video is set",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Output a histogram (distance vs frequency) for each depth file instead of images",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip creating a video from the visualizations",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "depth_vis")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.histogram:
        histogram_dir = os.path.join(os.path.dirname(args.output_dir), "histograms")
        os.makedirs(histogram_dir, exist_ok=True)

    files = find_depth_files(args.input_dir)
    if not files:
        print(f"No .npy or .npy.gz files found in {args.input_dir}")
        return

    # Determine vmin/vmax
    if args.vmin is None or args.vmax is None:
        _, _ = compute_global_percentiles(files, 2, 98)
        lower, upper = compute_global_percentiles(files, 1, 99)
        vmin = args.vmin if args.vmin is not None else lower
        vmax = args.vmax if args.vmax is not None else upper
    else:
        vmin, vmax = args.vmin, args.vmax

    # Process and save each file as PNG
    for fp in tqdm.tqdm(files, desc="Saving Depth Files"):
        depth = np.squeeze(load_depth(fp))

        if depth.ndim == 3:
            depth = np.mean(depth, axis=-1)

        name = os.path.basename(fp).split(".")[0]
        out_path = os.path.join(args.output_dir, name + ".png")

        plt.imsave(out_path, depth, cmap="viridis", vmin=vmin, vmax=vmax)

        if args.histogram:
            # Histogram mode
            dmean, dmedian, dmin, dmax = (
                np.mean(depth),
                np.median(depth),
                np.min(depth),
                np.max(depth),
            )
            plt.figure()
            plt.hist(depth.flatten(), bins=256, edgecolor="none", range=(0, 24))
            plt.xlabel("Depth")
            plt.ylabel("Frequency")
            plt.title(
                f"Histogram of {name}\n"
                f"mean={dmean:.2f}, med={dmedian:.2f}, min={dmin:.2f}, max={dmax:.2f}"
            )
            out_path = os.path.join(histogram_dir, name + ".png")
            plt.savefig(out_path)
            plt.close()

    # Create video if requested
    if not args.no_video:
        video_path = os.path.join(
            os.path.dirname(args.output_dir), "000_depth_video.mp4"
        )

        # Run from the PNG folder so globbing works and we don't need -safe 0
        cmd = [
            "ffmpeg",
            "-y",  # overwrite existing files
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-pattern_type",
            "glob",
            "-i",
            "*.png",
            "-c:v",
            "libx264",
            "-preset",
            "slow",  # speed/quality tradeoff
            "-crf",
            "18",  # visually lossless-ish
            "-pix_fmt",
            "yuv420p",  # broad compatibility (PowerPoint, web)
            "-movflags",
            "+faststart",  # better streaming/startup
            video_path,
        ]

        try:
            subprocess.run(cmd, check=True, cwd=args.output_dir)
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"FFmpeg failed to create video: {e}")
            print(
                "Note: FFmpeg is required to create videos. Install it or use --no_video to skip."
            )


if __name__ == "__main__":
    main()
