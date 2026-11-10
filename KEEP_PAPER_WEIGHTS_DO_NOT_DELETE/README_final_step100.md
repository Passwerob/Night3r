# Night3R Final Checkpoint

## Selection

- Final checkpoint: `event_dark3r_night3r_final.pth`
- Selected training point: step 100
- Initialization: three-dataset global-final `blend700`
- Training sources: MVSEC, DSEC train split, and DL3DV with DA3 pseudo-depth
- Source sampling ratio: DL3DV:DSEC:MVSEC = 4:2:1
- The final file is a hard link to the immutable step-100 training artifact, so
  both paths contain exactly the same checkpoint bytes without duplicating 3.8 GB.

## Metrics

| Dataset | Reference | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 |
|---|---|---:|---:|---:|---:|
| MVSEC | GT | 0.2008 | 0.7591 | 0.1933 | 0.7976 |
| DSEC | GT | 0.1599 | 0.7252 | 0.0684 | 0.9686 |
| DL3DV | Held-out DA3 pseudo-depth | 0.6820 | 0.1988 | 0.3061 | 0.6367 |

MVSEC and DSEC are evaluated against dataset/sensor depth. DL3DV measures
agreement with held-out DA3 pseudo-depth and must not be described as
ground-truth depth accuracy.

## Provenance

- Source checkpoint:
  `../night3r_global_final_continue_200_4gpu_20260723/event_dark3r_step00000100.pth`
- Training record:
  `../night3r_global_final_continue_200_4gpu_20260723/README.md`
- Evaluation:
  `../night3r_global_final_continue_200_eval_20260723/summary.md`
