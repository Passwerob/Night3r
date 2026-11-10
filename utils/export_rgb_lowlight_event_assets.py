#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.lowlight_augmentation import ContinuousLowLightAug


DATASETS = ("MVSEC", "DSEC", "DL3DV")


def parse_args():
    parser = argparse.ArgumentParser("Export RGB/low-light/event figure assets")
    parser.add_argument(
        "--train-root", type=Path, default=Path("/data/fcr/data/train_data")
    )
    parser.add_argument("--test-root", type=Path, default=Path("test"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/paper_assets_rgb_lowlight_event_20260721"),
    )
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--frames-per-scene", type=int, default=7)
    parser.add_argument("--exposure", type=float, default=0.10)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_event(path):
    try:
        event = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        event = torch.load(path, map_location="cpu")
    if not torch.is_tensor(event) or event.ndim != 3 or event.shape[0] % 2:
        raise ValueError(f"Invalid polarity-separated event voxel: {path}")
    return event.float()


def image_tensor(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def to_image(tensor):
    array = (
        tensor.detach().float().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    )
    return Image.fromarray(array)


def event_activity(event):
    half = event.shape[0] // 2
    positive = event[:half].abs().sum(dim=0)
    negative = event[half:].abs().sum(dim=0)
    return positive, negative


def robust_normalize(activity):
    nonzero = activity[activity > 0]
    if nonzero.numel() == 0:
        return torch.zeros_like(activity)
    scale = torch.quantile(torch.log1p(nonzero), 0.995).clamp_min(1.0e-6)
    return (torch.log1p(activity) / scale).clamp(0, 1)


def event_red_blue(event, background="black"):
    positive, negative = event_activity(event)
    pos = robust_normalize(positive)
    neg = robust_normalize(negative)
    if background == "black":
        rgb = torch.stack([pos, torch.zeros_like(pos), neg], dim=0)
    elif background == "white":
        strength = torch.maximum(pos, neg)
        total = (pos + neg).clamp_min(1.0e-6)
        pos_ratio = pos / total
        neg_ratio = neg / total
        color = torch.stack(
            [
                pos_ratio + 0.08 * neg_ratio,
                0.12 * pos_ratio + 0.18 * neg_ratio,
                0.08 * pos_ratio + neg_ratio,
            ],
            dim=0,
        )
        rgb = (1.0 - strength.unsqueeze(0)) + strength.unsqueeze(0) * color
    else:
        raise ValueError(background)
    return to_image(rgb), pos, neg


def event_overlay(lowlight, pos, neg):
    strength = torch.maximum(pos, neg).pow(0.65)
    total = (pos + neg).clamp_min(1.0e-6)
    event_color = torch.stack(
        [pos / total, torch.zeros_like(pos), neg / total], dim=0
    )
    alpha = (0.72 * strength).unsqueeze(0)
    return to_image(lowlight * (1.0 - alpha) + event_color * alpha)


def lowlight_image(clean, args, token):
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    local_seed = args.seed + int.from_bytes(digest[:4], "little")
    state = torch.random.get_rng_state()
    torch.manual_seed(local_seed)
    try:
        augment = ContinuousLowLightAug(
            min_exposure=args.exposure,
            max_exposure=args.exposure,
            read_noise_std=args.read_noise_std,
            shot_noise=True,
            shot_noise_std=args.shot_noise_std,
            gamma_space_input=True,
        )
        return augment(clean)
    finally:
        torch.random.set_rng_state(state)


def candidate_score(rgb, event):
    gray = rgb.mean(dim=0)
    contrast = float(gray.std())
    gradient = float(
        0.5
        * (
            (gray[:, 1:] - gray[:, :-1]).abs().mean()
            + (gray[1:, :] - gray[:-1, :]).abs().mean()
        )
    )
    positive, negative = event_activity(event)
    occupied = (positive + negative) > 0
    occupancy = float(occupied.float().mean())
    pos_sum = float(positive.sum())
    neg_sum = float(negative.sum())
    balance = 1.0 - abs(pos_sum - neg_sum) / max(pos_sum + neg_sum, 1.0e-6)
    occupancy_preference = float(np.exp(-((occupancy - 0.30) / 0.28) ** 2))
    score = 2.5 * contrast + 8.0 * gradient + 0.6 * balance + occupancy_preference
    return score, {
        "rgb_mean": float(gray.mean()),
        "rgb_contrast": contrast,
        "rgb_gradient": gradient,
        "event_occupancy": occupancy,
        "event_positive_sum": pos_sum,
        "event_negative_sum": neg_sum,
        "event_polarity_balance": balance,
    }


def scene_candidates(root, excluded_scene_names, frames_per_scene):
    candidates = []
    for scene_dir in sorted(root.glob("screen-*")):
        if scene_dir.name in excluded_scene_names:
            continue
        images = sorted((scene_dir / "images").glob("*.png"))
        events = sorted((scene_dir / "events").glob("*.pt"))
        count = min(len(images), len(events))
        if count == 0:
            continue
        indices = np.linspace(0, count - 1, min(frames_per_scene, count), dtype=int)
        best = None
        for index in sorted(set(int(value) for value in indices)):
            rgb = image_tensor(images[index])
            event = load_event(events[index])
            score, stats = candidate_score(rgb, event)
            item = {
                "scene": scene_dir.name,
                "frame_index": index,
                "image_path": images[index],
                "event_path": events[index],
                "score": score,
                "stats": stats,
            }
            if best is None or score > best["score"]:
                best = item
        if best is not None:
            candidates.append(best)
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def make_triptych(clean_image, lowlight_image, event_image):
    width = min(clean_image.width, lowlight_image.width, event_image.width)
    height = min(clean_image.height, lowlight_image.height, event_image.height)
    panels = [
        clean_image.resize((width, height)),
        lowlight_image.resize((width, height)),
        event_image.resize((width, height)),
    ]
    label_height = 42
    canvas = Image.new("RGB", (3 * width, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["Reference RGB", "Low-light RGB", "Events (+ red / - blue)"]
    for index, (label, panel) in enumerate(zip(labels, panels)):
        left = index * width
        draw.text((left + 12, 13), label, fill="black")
        canvas.paste(panel, (left, label_height))
    return canvas


def export_candidate(item, split, dataset, rank, args):
    name = f"rank{rank:02d}_{item['scene']}_frame{item['frame_index']:06d}"
    output = args.outdir / split / dataset / name
    output.mkdir(parents=True, exist_ok=True)
    clean = image_tensor(item["image_path"])
    event = load_event(item["event_path"])
    lowlight = lowlight_image(clean, args, f"{split}/{dataset}/{name}")
    clean_image = to_image(clean)
    lowlight_render = to_image(lowlight)
    event_black, pos, neg = event_red_blue(event, background="black")
    event_white, _, _ = event_red_blue(event, background="white")
    overlay = event_overlay(lowlight, pos, neg)

    clean_image.save(output / "reference_rgb.png")
    lowlight_render.save(output / "lowlight_rgb_exp010.png")
    event_black.save(output / "events_pos_red_neg_blue_black.png")
    event_white.save(output / "events_pos_red_neg_blue_white.png")
    overlay.save(output / "lowlight_event_overlay.png")
    make_triptych(clean_image, lowlight_render, event_white).save(
        output / "triptych_labeled.png"
    )
    metadata = {
        "split": split,
        "dataset": dataset,
        "scene": item["scene"],
        "frame_index": item["frame_index"],
        "source_image": str(item["image_path"].resolve()),
        "source_event": str(item["event_path"].resolve()),
        "selection_score": item["score"],
        "statistics": item["stats"],
        "lowlight": {
            "exposure": args.exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise_std": args.shot_noise_std,
            "gamma_space_input": True,
        },
        "event_color_convention": {
            "positive": "red",
            "negative": "blue",
            "magnitude": "log1p, normalized by nonzero 99.5 percentile",
        },
        "note": (
            "MVSEC reference_rgb is the original captured nighttime frame; "
            "DSEC/DL3DV reference_rgb is the source frame used for synthetic darkening."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {**metadata, "asset_dir": str(output.resolve())}


def contact_sheet(records, output):
    panels = []
    for record in records:
        path = Path(record["asset_dir"]) / "triptych_labeled.png"
        panels.append(Image.open(path).convert("RGB"))
    if not panels:
        return
    width = min(1800, max(panel.width for panel in panels))
    resized = [
        panel.resize((width, round(panel.height * width / panel.width))) for panel in panels
    ]
    canvas = Image.new("RGB", (width, sum(panel.height for panel in resized)), "white")
    top = 0
    for panel in resized:
        canvas.paste(panel, (0, top))
        top += panel.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for dataset in DATASETS:
        train_dataset_root = args.train_root / dataset
        test_dataset_root = args.test_root / dataset
        train_names = {
            path.name for path in train_dataset_root.glob("screen-*") if path.is_dir()
        }
        test_names = {
            path.name for path in test_dataset_root.glob("screen-*") if path.is_dir()
        }
        group_specs = (
            ("train", train_dataset_root, test_names),
            ("test", test_dataset_root, set()),
        )
        for split, root, excluded in group_specs:
            candidates = scene_candidates(root, excluded, args.frames_per_scene)
            if split == "test":
                candidates = [item for item in candidates if item["scene"] not in set()]
            selected = candidates[: args.per_group]
            records = [
                export_candidate(item, split, dataset, rank, args)
                for rank, item in enumerate(selected, start=1)
            ]
            all_records.extend(records)
            contact_sheet(
                records,
                args.outdir / "contact_sheets" / f"{split}_{dataset}.png",
            )

    manifest = {
        "status": "figure_asset_candidates",
        "train_test_selection_policy": (
            "Training candidates exclude every scene name present under the corresponding "
            "test directory."
        ),
        "records": all_records,
    }
    (args.outdir / "index.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    lines = [
        "| Split | Dataset | Scene | Frame | Score | Event occupancy | Asset directory |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for record in all_records:
        lines.append(
            f"| {record['split']} | {record['dataset']} | {record['scene']} | "
            f"{record['frame_index']} | {record['selection_score']:.4f} | "
            f"{record['statistics']['event_occupancy']:.4f} | "
            f"`{record['asset_dir']}` |"
        )
    (args.outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir.resolve()), "count": len(all_records)}, indent=2))


if __name__ == "__main__":
    main()
