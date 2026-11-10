#!/usr/bin/env python3
"""Package MVSEC pairs with RGB-baseline collapse and strong event response."""

import argparse
import csv
import json
import shutil
from pathlib import Path


SELECTIONS = (
    ("screen-000115-mvsec", 26, 27),
    ("screen-000156-mvsec", 30, 31),
    ("screen-000157-mvsec", 36, 37),
    ("screen-000205-mvsec", 47, 48),
    ("screen-000122-mvsec", 44, 45),
)

METRIC_FIELDS = (
    "master_raw_absrel",
    "master_raw_delta1",
    "master_median_absrel",
    "master_median_delta1",
    "night_raw_absrel",
    "night_raw_delta1",
    "night_median_absrel",
    "night_median_delta1",
    "zero_raw_absrel",
    "zero_raw_delta1",
    "zero_median_absrel",
    "zero_median_delta1",
    "event_activity_mean",
    "event_occupancy",
    "event_gate_mean",
    "lowlight_mean",
    "lowlight_noise",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/fcr/data/Night3R/test/MVSEC"),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(
            "/data/fcr/data/Night3R/outputs/"
            "all_test_visual_sweep_step100_final_mildnoise_20260723"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/fcr/data/Night3R/outputs/"
            "mvsec_collapse_event_rich_selected_20260726"
        ),
    )
    return parser.parse_args()


def load_rankings(sweep_root):
    rankings = {}
    ranking_files = sorted((sweep_root / "shards").glob("gpu*_mvsec_*/ranking.csv"))
    if not ranking_files:
        raise FileNotFoundError(f"No MVSEC ranking files under {sweep_root}")
    for path in ranking_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["scene"], int(row["idx1"]), int(row["idx2"]))
                rankings[key] = row
    return rankings


def copy_pair_source(source_scene, output_source, idx1, idx2):
    for folder, suffix in (
        ("images", ".png"),
        ("events", ".pt"),
        ("depth_gt", ".pt"),
    ):
        source_folder = source_scene / folder
        if not source_folder.is_dir():
            continue
        destination_folder = output_source / folder
        destination_folder.mkdir(parents=True, exist_ok=True)
        for index in (idx1, idx2):
            source = source_folder / f"{index:06d}{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination_folder / source.name)

    for name in ("cameras.yaml", "meta.json"):
        source = source_scene / name
        if source.is_file():
            shutil.copy2(source, output_source / name)


def numeric_metrics(row):
    return {field: float(row[field]) for field in METRIC_FIELDS}


def write_readme(output_root, records):
    lines = [
        "# MVSEC RGB-Collapse / Event-Response Selection",
        "",
        "These qualitative pairs were selected from the full MVSEC sweep using",
        "the final Night3R step-100 checkpoint. All scenes overlap the final",
        "MVSEC adaptation domain and must be labeled as qualitative/training-domain",
        "examples rather than strict-heldout benchmark evidence.",
        "",
        "| Pair | MASt3R med AbsRel | Night3R med AbsRel | Zero-event med AbsRel | Event occupancy |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"| {record['scene']}:{record['idx1']:06d}-{record['idx2']:06d} "
            f"| {metrics['master_median_absrel']:.4f} "
            f"| {metrics['night_median_absrel']:.4f} "
            f"| {metrics['zero_median_absrel']:.4f} "
            f"| {metrics['event_occupancy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Each pair directory contains original RGB/event/depth tensors under",
            "`source/` and the low-light RGB, red/blue event renders, MASt3R,",
            "Night3R-real, Night3R-zero, GT depth, and geometry arrays under",
            "`visualization/`.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings(args.sweep_root)
    records = []
    panels_dir = args.output_root / "panels"
    panels_dir.mkdir()

    for order, key in enumerate(SELECTIONS, start=1):
        scene, idx1, idx2 = key
        if key not in rankings:
            raise KeyError(f"Selected pair not present in ranking.csv: {key}")
        row = rankings[key]
        source_scene = args.source_root / scene
        asset_dir = Path(row["asset_dir"])
        if not source_scene.is_dir() or not asset_dir.is_dir():
            raise FileNotFoundError(f"Missing source or assets for {key}")

        pair_name = f"{order:02d}_{scene}_{idx1:06d}_{idx2:06d}"
        pair_root = args.output_root / pair_name
        output_source = pair_root / "source"
        output_source.mkdir(parents=True)
        copy_pair_source(source_scene, output_source, idx1, idx2)
        shutil.copytree(asset_dir, pair_root / "visualization")
        shutil.copy2(
            asset_dir / "selection_panel.png",
            panels_dir / f"{pair_name}.png",
        )

        metrics = numeric_metrics(row)
        record = {
            "order": order,
            "scene": scene,
            "idx1": idx1,
            "idx2": idx2,
            "sample_id": f"{scene}:{idx1:06d}-{idx2:06d}",
            "split_label": "adaptation-domain/train-scene-overlap",
            "source_pair_path": str(output_source.relative_to(args.output_root)),
            "visualization_path": str(
                (pair_root / "visualization").relative_to(args.output_root)
            ),
            "metrics": metrics,
            "mast3r_to_night_absrel_gain": (
                metrics["master_median_absrel"]
                - metrics["night_median_absrel"]
            ),
            "zero_to_real_event_absrel_gain": (
                metrics["zero_median_absrel"]
                - metrics["night_median_absrel"]
            ),
        }
        (pair_root / "pair_manifest.json").write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )
        records.append(record)

    (args.output_root / "manifest.json").write_text(
        json.dumps({"version": 1, "pairs": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = (
            "order",
            "sample_id",
            "split_label",
            "master_median_absrel",
            "night_median_absrel",
            "zero_median_absrel",
            "mast3r_to_night_absrel_gain",
            "zero_to_real_event_absrel_gain",
            "event_occupancy",
            "event_gate_mean",
            "lowlight_mean",
            "lowlight_noise",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                {
                    "order": record["order"],
                    "sample_id": record["sample_id"],
                    "split_label": record["split_label"],
                    "master_median_absrel": metrics["master_median_absrel"],
                    "night_median_absrel": metrics["night_median_absrel"],
                    "zero_median_absrel": metrics["zero_median_absrel"],
                    "mast3r_to_night_absrel_gain": record[
                        "mast3r_to_night_absrel_gain"
                    ],
                    "zero_to_real_event_absrel_gain": record[
                        "zero_to_real_event_absrel_gain"
                    ],
                    "event_occupancy": metrics["event_occupancy"],
                    "event_gate_mean": metrics["event_gate_mean"],
                    "lowlight_mean": metrics["lowlight_mean"],
                    "lowlight_noise": metrics["lowlight_noise"],
                }
            )
    write_readme(args.output_root, records)
    print(json.dumps({"output": str(args.output_root), "pairs": len(records)}))


if __name__ == "__main__":
    main()
