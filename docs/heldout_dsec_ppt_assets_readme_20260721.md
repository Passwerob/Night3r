# Night3R held-out DSEC PPT assets

This package contains presentation-ready PNG assets for the source-disjoint
DSEC `zurich_city_11_b` sequence (`screen-000004-dsec`). The final Night3R
checkpoint predates the test export, and the sequence-level overlap audit is
included in `00_Overview/`.

## Evaluation protocol

- Checkpoint: final MVSEC-1k Night3R checkpoint, step 1000.
- Test sequence: DSEC `zurich_city_11_b`.
- Low-light transform: exposure 0.10, read-noise standard deviation 0.01,
  shot noise enabled, gamma-space input enabled.
- Event representation: four temporal bins per polarity, eight channels total.
- Pair positions: 120--121, 360--361, 600--601, and 840--841.
- Pair positions were fixed before inspecting model metrics.
- Pair geometry uses the same ground-truth relative pose and support mask.
- The 3D comparison uses raw metric depth without scale alignment.
- The 2D depth and error panels are explicitly marked as median aligned.

## Raw AbsRel

| Pair | MASt3R | Night3R |
|---|---:|---:|
| 120--121 | 0.5896 | 0.2479 |
| 360--361 | 0.7154 | 0.3012 |
| 600--601 | 0.7122 | 0.2927 |
| 840--841 | 0.8646 | 0.4388 |

## Suggested PPT order

1. Use `00_Overview/01_RGB_Event_Geometry_Overview.png` as the complete result.
2. Use a pair folder's clean and low-light RGB images to introduce the setting.
3. Add the corresponding event images.
4. Compare `MASt3R`, `Night3R`, and `GT` elevated geometry images.
5. Use `15_3D_Comparison_Front_Elevated.png` when a single comparison panel is preferred.

The result supports improved metric-scale recovery. It must not be described
as showing that Night3R wins every median-aligned or scale-invariant shape
metric.
