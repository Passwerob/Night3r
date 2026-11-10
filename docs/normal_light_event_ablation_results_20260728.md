# Native-Daylight Event Ablation Results

## Protocol

All runs use original RGB frames (`e=1.0`) with no synthetic darkening and no
synthetic noise.  The final model, pair order, and depth masks are fixed; only
the event tensor changes between Real, Zero, and Shuffle.  Real uses the
synchronized voxel.  Zero removes all events.  Shuffle preserves tensor shape
and values but breaks event--RGB spatial and channel alignment.

The selected normal-light result is the strict scene-level held-out DSEC
daytime driving sequence `zurich_city_02_b` (612 pairs).  The additional
strict `zurich_city_11_b` scene (966 pairs) and their combined pool are
retained as internal diagnostics only and are not part of the reported
normal-light claim.  The selected scene must be presented as a case study,
not as a post hoc dataset-average or a universal normal-light conclusion.
`outdoor_day1/day2` in MVSEC (96 pairs) is an auxiliary daylight check, not a
held-out benchmark, because these scenes overlap the broad MVSEC source root
used by the final adaptation stage.

## Final-Model Controls

| Split | Event | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 |
|---|---|---:|---:|---:|---:|
| DSEC `zurich_city_02_b` | Real | 0.1351 | 0.8389 | 0.0812 | 0.9453 |
|  | Zero | 0.2091 | 0.5412 | 0.0865 | 0.9311 |
|  | Shuffle | 0.1711 | 0.6719 | 0.0965 | 0.9084 |
| DSEC `zurich_city_11_b` | Real | 0.1570 | 0.7408 | 0.1303 | 0.8453 |
|  | Zero | 0.1671 | 0.7300 | 0.0766 | 0.9473 |
|  | Shuffle | 0.2015 | 0.5773 | 0.1274 | 0.8336 |
| DSEC combined | Real | 0.1509 | 0.7682 | 0.1166 | 0.8733 |
|  | Zero | 0.1789 | 0.6772 | 0.0793 | 0.9428 |
|  | Shuffle | 0.1931 | 0.6017 | 0.1186 | 0.8545 |
| MVSEC day (auxiliary) | Real | 0.2016 | 0.5979 | 0.1538 | 0.8032 |
|  | Zero | 0.6220 | 0.0300 | 0.2781 | 0.5203 |
|  | Shuffle | 0.4769 | 0.2249 | 0.2755 | 0.5535 |

## Selected Result

The reported normal-light event-control result is DSEC
`zurich_city_02_b` only.  Real events improve both metric protocols relative
to both controls: raw AbsRel is `0.1351` versus `0.2091` (Zero) and `0.1711`
(Shuffle), while median-aligned AbsRel is `0.0812` versus `0.0865` and
`0.0965`, respectively.

## Interpretation

The result supports a limited but useful normal-light conclusion.

- `zurich_city_11_b` and the combined DSEC pool remain heterogeneous under
  per-pair median alignment.  They are intentionally excluded from the
  reported normal-light conclusion rather than being hidden or pooled into it.
- The auxiliary MVSEC daytime result improves under both raw and
  median-aligned metrics, but it is not valid evidence of scene-held-out
  generalization.
- The matched 10K fusion ablation on `zurich_city_02_b` gives median AbsRel of
  0.0702 (Full BRG), 0.0705 (Concat--MLP), and 0.0733 (Additive).  BRG is better
  than additive fusion but essentially tied with Concat--MLP; no stronger claim
  is supported.

## Paper-Safe Wording

"On the strict held-out daytime DSEC `zurich_city_02_b` sequence, synchronized
events improve both metric-scale and median-aligned depth accuracy over zero
and shuffled-event controls."
