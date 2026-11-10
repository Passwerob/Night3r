#!/usr/bin/env python3
"""Assemble a virtual long DL3DV scene from consecutive exported clips."""

import argparse
import copy
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Assemble a long DL3DV evaluation scene")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-name", required=True)
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_scene = Path(args.output_root).resolve() / args.scene_name
    if output_scene.exists():
        raise SystemExit(f"Output scene already exists: {output_scene}")

    chunks = []
    reference_transforms = None
    for scene_name in args.scenes:
        scene_dir = source_root / scene_name
        transforms_path = scene_dir / "transforms.json"
        meta_path = scene_dir / "meta.json"
        images = sorted((scene_dir / "images").glob("*.png"))
        events = sorted((scene_dir / "events").glob("*.pt"))
        if not transforms_path.is_file() or not meta_path.is_file() or not images or not events:
            raise RuntimeError(f"Missing aligned DL3DV inputs in {scene_dir}")
        transforms = load_json(transforms_path)
        frames = transforms.get("frames", [])
        count = min(len(images), len(events), len(frames))
        if count == 0:
            raise RuntimeError(f"No aligned RGB/event/pose frames in {scene_dir}")
        if reference_transforms is None:
            reference_transforms = {key: value for key, value in transforms.items() if key != "frames"}
        else:
            for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "camera_model"):
                if transforms.get(key) != reference_transforms.get(key):
                    raise RuntimeError(f"Camera metadata mismatch for {key} in {scene_dir}")
        chunks.append(
            {
                "name": scene_name,
                "dir": scene_dir,
                "meta": load_json(meta_path),
                "images": images[:count],
                "events": events[:count],
                "frames": frames[:count],
            }
        )

    output_images = output_scene / "images"
    output_events = output_scene / "events"
    output_meta = output_scene / "meta"
    output_images.mkdir(parents=True)
    output_events.mkdir(parents=True)
    output_meta.mkdir(parents=True)

    combined_frames = []
    manifest_chunks = []
    global_index = 0
    for chunk in chunks:
        start_index = global_index
        for image, event, frame in zip(chunk["images"], chunk["events"], chunk["frames"]):
            stem = f"{global_index:06d}"
            os.symlink(image.resolve(), output_images / f"{stem}.png")
            os.symlink(event.resolve(), output_events / f"{stem}.pt")
            merged_frame = copy.deepcopy(frame)
            merged_frame["file_path"] = f"images/{stem}.png"
            combined_frames.append(merged_frame)
            global_index += 1
        manifest_chunks.append(
            {
                "scene": chunk["name"],
                "source_scene": chunk["meta"].get("scene"),
                "source_clip": chunk["meta"].get("clip"),
                "num_frames": global_index - start_index,
                "global_frame_range": [start_index, global_index - 1],
            }
        )

    transforms = dict(reference_transforms)
    transforms["frames"] = combined_frames
    (output_scene / "transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")

    meta = dict(chunks[0]["meta"])
    meta.update(
        {
            "source_dataset": "DL3DV",
            "sequence_kind": "virtual_consecutive_test_clips",
            "source_scenes": [chunk["name"] for chunk in chunks],
            "contains_depth_gt": False,
            "num_frames": global_index,
        }
    )
    (output_scene / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    manifest = {
        "scene": args.scene_name,
        "source_root": str(source_root),
        "source_scenes": [chunk["name"] for chunk in chunks],
        "num_frames": global_index,
        "chunks": manifest_chunks,
        "storage": "image and event files are absolute symlinks to held-out source clips",
    }
    (output_meta / "long_sequence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_scene": str(output_scene), **manifest}, indent=2))


if __name__ == "__main__":
    main()
