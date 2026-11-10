#!/usr/bin/env python3
"""Create a virtual long MVSEC sequence from consecutive exported chunks."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Assemble an MVSEC long-sequence evaluation scene")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-name", default="screen-long-outdoor_night2-mvsec")
    return parser.parse_args()


def load_pose_array(path):
    payload = np.load(path)
    for key in ("T_world_body", "T_world_cam", "T_world_camera"):
        if key in payload:
            poses = np.asarray(payload[key], dtype=np.float32)
            if poses.ndim == 3 and poses.shape[-2:] == (4, 4):
                return key, poses
    raise RuntimeError(f"No [N,4,4] pose array was found in {path}")


def load_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_scene = Path(args.output_root).resolve() / args.scene_name
    if output_scene.exists():
        raise SystemExit(f"Output scene already exists: {output_scene}")

    chunks = []
    pose_key = None
    previous_timestamp = None
    for scene_name in args.scenes:
        scene_dir = source_root / scene_name
        images = sorted((scene_dir / "images").glob("*.png"))
        events = sorted((scene_dir / "events").glob("*.pt"))
        rows = load_rows(scene_dir / "meta" / "frames_meta.csv")
        key, poses = load_pose_array(scene_dir / "meta" / "poses_selected.npz")
        if pose_key is None:
            pose_key = key
        elif pose_key != key:
            raise RuntimeError(f"Pose key mismatch: expected {pose_key}, got {key}")
        count = min(len(images), len(events), len(rows), len(poses))
        if count == 0:
            raise RuntimeError(f"No aligned RGB/event/pose frames in {scene_dir}")
        rows = rows[:count]
        current_timestamp = int(rows[0]["center_event_t"])
        chunks.append(
            {
                "name": scene_name,
                "dir": scene_dir,
                "images": images[:count],
                "events": events[:count],
                "rows": rows,
                "poses": poses[:count],
                "start_timestamp_us": current_timestamp,
                "end_timestamp_us": int(rows[-1]["center_event_t"]),
                "gap_from_previous_us": (
                    None
                    if previous_timestamp is None
                    else current_timestamp - previous_timestamp
                ),
            }
        )
        previous_timestamp = int(rows[-1]["center_event_t"])

    output_images = output_scene / "images"
    output_events = output_scene / "events"
    output_meta = output_scene / "meta"
    output_images.mkdir(parents=True)
    output_events.mkdir(parents=True)
    output_meta.mkdir(parents=True)

    camera_source = chunks[0]["dir"] / "cameras.yaml"
    if not camera_source.is_file():
        raise RuntimeError(f"Missing camera calibration: {camera_source}")
    os.symlink(camera_source, output_scene / "cameras.yaml")

    all_poses = []
    output_rows = []
    source_index = 0
    for chunk in chunks:
        for local_index, (image, event, row) in enumerate(
            zip(chunk["images"], chunk["events"], chunk["rows"])
        ):
            filename = f"{source_index:06d}"
            os.symlink(image, output_images / f"{filename}.png")
            os.symlink(event, output_events / f"{filename}.pt")
            output_rows.append(
                {
                    "global_index": source_index,
                    "source_scene": chunk["name"],
                    "source_local_index": local_index,
                    "center_event_t": row["center_event_t"],
                    "rgb_ts": row.get("rgb_ts", ""),
                    "pose_ts": row.get("pose_ts", ""),
                    "num_events": row.get("num_events", ""),
                }
            )
            all_poses.append(chunk["poses"][local_index])
            source_index += 1

    with (output_meta / "frames_meta.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    np.savez_compressed(output_meta / "poses_selected.npz", **{pose_key: np.stack(all_poses)})
    meta = {
        "source_dataset": "mvsec",
        "source_scene": "outdoor_night2",
        "sequence_kind": "virtual_consecutive_exported_chunks",
        "contains_depth_gt": False,
    }
    (output_scene / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    manifest = {
        "scene": args.scene_name,
        "source_root": str(source_root),
        "source_scenes": [chunk["name"] for chunk in chunks],
        "num_frames": len(all_poses),
        "pose_key": pose_key,
        "chunk_boundaries": [
            {
                "scene": chunk["name"],
                "num_frames": len(chunk["poses"]),
                "start_timestamp_us": chunk["start_timestamp_us"],
                "end_timestamp_us": chunk["end_timestamp_us"],
                "gap_from_previous_us": chunk["gap_from_previous_us"],
            }
            for chunk in chunks
        ],
        "max_inter_chunk_gap_us": max(
            [chunk["gap_from_previous_us"] or 0 for chunk in chunks]
        ),
        "storage": "image and event files are absolute symlinks to source chunks",
    }
    (output_meta / "long_sequence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"output_scene": str(output_scene), **manifest}, indent=2))


if __name__ == "__main__":
    main()
