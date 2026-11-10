#!/usr/bin/env python3
"""Trim trajectory JSONs to their longest common frame range for rendering."""

import argparse
import copy
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def select_scene(payload, scene_name):
    matches = [scene for scene in payload.get("scenes", []) if scene.get("scene") == scene_name]
    if len(matches) != 1 or not isinstance(matches[0].get("trajectory"), dict):
        raise RuntimeError(f"Expected one exported trajectory for {scene_name}")
    return matches[0]


def longest_contiguous(values):
    values = sorted(values)
    if not values:
        return []
    best = current = [values[0]]
    for value in values[1:]:
        if value == current[-1] + 1:
            current.append(value)
        else:
            if len(current) > len(best):
                best = current
            current = [value]
    return current if len(current) > len(best) else best


def main():
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    payloads = [json.loads(path.read_text()) for path in input_paths]
    scenes = [select_scene(payload, args.scene) for payload in payloads]
    trajectories = [scene["trajectory"] for scene in scenes]
    frame_maps = []
    for trajectory in trajectories:
        frames = trajectory["frame_indices"]
        predicted = trajectory["pred_camera_to_world"]
        ground_truth = trajectory["gt_camera_to_world"]
        if not (len(frames) == len(predicted) == len(ground_truth)):
            raise RuntimeError("Trajectory arrays have inconsistent lengths")
        frame_maps.append(
            {
                int(frame): (prediction, gt)
                for frame, prediction, gt in zip(frames, predicted, ground_truth)
            }
        )
    common_frames = longest_contiguous(set.intersection(*(set(mapping) for mapping in frame_maps)))
    if len(common_frames) < 3:
        raise RuntimeError("The methods do not share a contiguous trajectory of at least three frames")

    reference = frame_maps[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, (path, payload, scene, mapping) in enumerate(
        zip(input_paths, payloads, scenes, frame_maps)
    ):
        trimmed = copy.deepcopy(payload)
        trimmed_scene = select_scene(trimmed, args.scene)
        trimmed_scene["trajectory"] = {
            "frame_indices": common_frames,
            "pred_camera_to_world": [mapping[frame][0] for frame in common_frames],
            "gt_camera_to_world": [reference[frame][1] for frame in common_frames],
        }
        trimmed_scene["num_frames"] = len(common_frames)
        trimmed_scene["num_edges_in_evaluated_trajectory"] = len(common_frames) - 1
        trimmed_scene["trajectory_start_frame"] = common_frames[0]
        output_path = output_dir / f"matched_{index}_{path.stem}.json"
        output_path.write_text(json.dumps(trimmed, indent=2, sort_keys=True) + "\n")
        outputs.append(str(output_path.resolve()))

    manifest = {
        "scene": args.scene,
        "source_inputs": [str(path.resolve()) for path in input_paths],
        "matched_frame_indices": common_frames,
        "num_matched_frames": len(common_frames),
        "alignment": (
            "Each method is subsequently aligned by one Sim(3) to the common GT "
            "trajectory during rendering. Metrics in the original JSONs remain full-run metrics."
        ),
        "outputs": outputs,
    }
    (output_dir / "matched_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
