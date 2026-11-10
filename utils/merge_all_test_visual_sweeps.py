#!/usr/bin/env python3
"""Merge independent GPU sweep manifests into one browsing index."""

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.root = args.root.resolve()
    manifests = []
    for path in sorted((args.root / "shards").glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_path"] = str(path.resolve())
        manifests.append(payload)
    if not manifests:
        raise RuntimeError(f"No completed shard manifests under {args.root / 'shards'}")

    records = [record for manifest in manifests for record in manifest.get("records", [])]
    records.sort(key=lambda item: (item["dataset"], item["scene"], item["scene_rank"]))
    fields = [
        "dataset", "scene", "sample_id", "scene_rank", "idx1", "idx2", "score",
        "selection_score", "score_kind", "split_label", "lowlight_mean", "lowlight_noise",
        "event_occupancy", "event_gate_mean", "asset_dir", "panel",
    ]
    with (args.root / "candidate_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    totals = {
        "shard_count": len(manifests),
        "processed_pairs": sum(int(item.get("processed_pairs", 0)) for item in manifests),
        "requested_pairs": sum(int(item.get("requested_pairs", 0)) for item in manifests),
        "selected_candidates": len(records),
        "datasets": dict(Counter(item["dataset"] for item in records)),
        "split_labels": dict(Counter(item["split_label"] for item in records)),
    }
    merged = {"totals": totals, "shards": manifests, "records": records}
    (args.root / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["dataset"]][record["scene"]].append(record)
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Night3R all-test visual sweep</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f1f5f9;color:#0f172a}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}"
        ".card{background:white;border:1px solid #cbd5e1;padding:12px}img{width:100%;height:auto}"
        ".badge{font-weight:700;color:#9a3412}.strict{color:#166534}code{background:#e2e8f0;padding:2px 5px}</style></head><body>",
        "<h1>Night3R all-test visual candidate sweep</h1>",
        f"<p>Processed <strong>{totals['processed_pairs']}</strong> / {totals['requested_pairs']} adjacent pairs; retained {len(records)} temporally diverse candidates.</p>",
        "<p><strong>Interpretation:</strong> DSEC/MVSEC panels use available depth GT for ranking. DL3DV has no depth GT, so its ranking only indicates model/event sensitivity and cannot establish better geometry.</p>",
    ]
    for dataset in ("DSEC", "MVSEC", "DL3DV"):
        if dataset not in grouped:
            continue
        lines.append(f"<h2>{dataset}</h2>")
        for scene, items in sorted(grouped[dataset].items()):
            lines.append(f"<h3>{html.escape(scene)}</h3><div class='grid'>")
            for item in items:
                panel = Path(item["panel"]).relative_to(args.root)
                strict_class = " strict" if item["split_label"] == "strict-heldout" else ""
                lines.append(
                    "<div class='card'>"
                    f"<h4>#{item['scene_rank']} {html.escape(item['sample_id'])}</h4>"
                    f"<p class='badge{strict_class}'>{html.escape(item['split_label'])}</p>"
                    f"<p>score=<code>{item['score']:.4f}</code>; {html.escape(item['score_kind'])}</p>"
                    f"<a href='{html.escape(str(panel))}'><img loading='lazy' src='{html.escape(str(panel))}'></a>"
                    "</div>"
                )
            lines.append("</div>")
    lines.append("</body></html>")
    (args.root / "index.html").write_text("\n".join(lines), encoding="utf-8")

    synthetic_protocols = sorted(
        {
            (
                item.get("dataset"),
                item.get("lowlight", {}).get("exposure"),
                item.get("lowlight", {}).get("read_noise_std"),
                item.get("lowlight", {}).get("shot_noise_std"),
                item.get("lowlight", {}).get("mvsec_synthetic_darkening_opt_in", False),
            )
            for item in manifests
            if item.get("lowlight", {}).get("exposure") is not None
        }
    )
    protocol_lines = [
        f"- {dataset} input: controlled low light at exposure {exposure:.3f}, "
        f"read noise {read_noise:g}, and shot-noise scale {shot_noise:g}"
        f"{' (explicit synthetic darkening opt-in)' if dataset == 'MVSEC' and mvsec_opt_in else ''}."
        for dataset, exposure, read_noise, shot_noise, mvsec_opt_in in synthetic_protocols
    ]
    readme = [
        "# Night3R all-test visual sweep",
        "",
        f"- Processed adjacent pairs: {totals['processed_pairs']} / {totals['requested_pairs']}",
        f"- Exported candidates: {len(records)}",
        "- Per-scene selection: best candidate in each of 8 temporal buckets, followed by top 4 buckets.",
        *protocol_lines,
        "- DSEC/MVSEC ranking: median-aligned AbsRel improvement over MASt3R plus real-vs-zero event gain.",
        "- DL3DV ranking: scale-invariant prediction discrepancy only; this is not a quality metric because depth GT is unavailable.",
        "",
        "Split labels are mandatory when choosing figures:",
        "",
        "- `strict-heldout`: held-out DSEC scenes.",
        "- `strict-heldout/no-depth-gt`: held-out DL3DV scenes without depth GT.",
        "- `adaptation-domain/train-scene-overlap`: MVSEC scenes overlap the final MVSEC adaptation domain and are qualitative only.",
        "- `excluded-overlap`: DSEC screen-000002 overlaps training and must not be presented as held-out evidence.",
        "",
        "Open `index.html` to browse all selected panels.",
    ]
    (args.root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(args.root.resolve()), **totals}, indent=2))


if __name__ == "__main__":
    main()
