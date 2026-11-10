#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser("Summarize Night3R descriptor Sparse-GA sweep")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    panels = []
    for directory in sorted(args.run_dir.glob("sparse_ga_step*")):
        manifest_path = directory / "manifest.json"
        image_path = directory / "night3r_sparse_global_rgb.png"
        if not manifest_path.is_file() or not image_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        step = int(directory.name.rsplit("step", 1)[1])
        trajectory = manifest.get("trajectory_diagnostics", {})
        rows.append(
            {
                "step": step,
                "point_count": manifest.get("point_count"),
                "final_focal_px": manifest.get("final_focal_px"),
                "path_length_internal_units": trajectory.get(
                    "path_length_internal_units"
                ),
                "step_coefficient_of_variation": trajectory.get(
                    "step_coefficient_of_variation"
                ),
                "manifest": str(manifest_path.resolve()),
                "render": str(image_path.resolve()),
            }
        )
        panels.append((step, Image.open(image_path).convert("RGB")))

    payload = {
        "status": "requires_visual_validation",
        "selection_rule": (
            "Do not select by trajectory CV alone; accept only a checkpoint with "
            "both coherent scene structure and stable trajectory diagnostics."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "| Step | Points | Focal (px) | Path (internal) | Step CV |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['point_count']} | "
            f"{row['final_focal_px']:.3f} | "
            f"{row['path_length_internal_units']:.4f} | "
            f"{row['step_coefficient_of_variation']:.4f} |"
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    if panels:
        target_width = 900
        resized = []
        for step, panel in panels:
            height = round(panel.height * target_width / panel.width)
            resized.append((step, panel.resize((target_width, height))))
        label_height = 48
        canvas = Image.new(
            "RGB",
            (target_width, sum(panel.height + label_height for _, panel in resized)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        top = 0
        for step, panel in resized:
            draw.text((16, top + 14), f"Descriptor recovery: step {step}", fill="black")
            top += label_height
            canvas.paste(panel, (0, top))
            top += panel.height
        canvas.save(args.output_dir / "sweep_contact_sheet.png")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
