# Night3R Pair Review

Open `http://127.0.0.1:<port>/review_app/` while the local review server is running. The original sweep uses port `17892`; the dark low-noise sweep uses port `17894`.

## Shortcuts

- `K` or `Enter`: keep the current pair and advance.
- `X` or `Backspace`: skip the current pair and advance.
- `U`: undo the latest Keep/Skip decision.
- `Left`, `Right`, or `J`: navigate candidates.
- `1`: the default eight-panel view with two RGB images, two event maps, two depth predictions, and two 3D reconstructions.
- `2`, `3`, `4`, `5`: detailed Inputs, Depth, 3D, and comparison-sheet views.
- `F`: toggle the main viewer fullscreen.

The left queue shows the active filters and persistent Keep/Skip status. The center viewer uses the largest available area for Inputs, Depth, same-camera 3D, or the full comparison sheet. The right inspector keeps metrics and decisions visible while reviewing.

Shortcuts are disabled while typing in the search box or selection note.

## Selection Output

Every decision is written immediately under `../review_selection/`:

- `decisions.json`: complete persistent review state and undo history.
- `kept_manifest.json`: structured metadata for all kept pairs.
- `kept.csv`: compact kept-pair index.
- `kept/<dataset>/<pair>/`: one directory per kept pair, containing source and geometry asset links plus `selection.json`.

The links point to the existing sweep assets, so keeping a pair does not duplicate large image files.

## 3D Comparison Protocol

All methods in a pair use the same crop, camera, color range, and point stride. For GT-enabled DSEC and MVSEC pairs, predictions are independently median-aligned to GT for qualitative shape comparison. DL3DV has no depth GT; its Night3R geometry is median-aligned to MASt3R and must be treated as a visual discrepancy diagnostic, not a quality ranking.

Start a review root directly with:

```bash
python utils/serve_pair_review.py --root <sweep-root> --port <port>
```
