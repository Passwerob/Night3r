# KEEP — Night3R paper weights. DO NOT DELETE

Copied on 2026-10-28 09:02  to survive cleanup of outputs/.
Large .pth files are hard links on the same filesystem: they do not use extra 3.7GB,
but they still keep the bytes if the original path under outputs/ is removed.

## Files

- `event_dark3r_night3r_final.pth`
  Paper-final geometry model (global-final continue step 100, 2026-07-23).
  Original: `outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth`
- `event_dark3r_night3r_final_with_pose_head.pth`
  Same geometry plus adapted pose head (2026-07-26). Use for pose/trajectory figures.
  Original: `outputs/night3r_final_complete_pose_step100_20260726/event_dark3r_night3r_final_with_pose_head.pth`
- `pose_head_step00000100.pth`
  Pose head only, if you need to load it separately.
- `README_final_step100.md`
  Original selection/metrics note for the geometry checkpoint.

Backbone still needed at load time:
`checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
