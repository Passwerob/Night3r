# Depth and Geometry Alignment Notes

## Recommended Caption Language

**GT-median-aligned geometry comparison.** For each method and image pair, we
compute one scale from pixels where both the prediction and sensor depth are
valid, and apply it to the complete depth map and the corresponding 3D point
cloud. All panels share the same camera and the same Viridis depth range
derived from sensor depth. The visualization therefore compares local geometry
and structural fidelity after resolving the global scale ambiguity.

Shorter version:

**GT-median-aligned 3D geometry.** A single scale is estimated per method from
valid sensor-depth pixels and applied to both its depth map and point cloud;
all methods share the same camera and GT-derived color range.

## Actual GT-Median Alignment Used in the New Package

For method `m`, let `D_m` be its predicted depth, `D_GT` the sensor depth, and
`Omega_m` the pixels where both maps are finite and positive. We calculate one
scalar for the full pair:

```text
s_m = median(D_GT[Omega_m]) / median(D_m[Omega_m])
```

We then apply this exact scalar everywhere:

```text
D_m_aligned = s_m * D_m
P_m_aligned = s_m * P_m
```

Here `P_m = (X_m, Y_m, Z_m)` is the predicted point cloud in the view-1 camera
coordinate frame. Thus the same scale is applied to `X`, `Y`, and `Z`, rather
than scaling a rendered image after the fact.

After alignment:

- Every method uses the same camera for a given scene.
- All depth and point-cloud colors use `Viridis`.
- The displayed color range is the 2nd--98th percentile of valid sensor-depth
  GT for that scene.
- There is no further per-method translation, centering, or display rescaling.

This protocol is appropriate for comparing local shape, surface ordering,
boundaries, and structural coherence after removing global scale ambiguity. It
must not be used to claim superiority in absolute metric scale.

## Earlier Depth Alignment Used in Some Legacy Assets

Some early two-view visual assets estimated one scalar from view 1 and applied
the same scalar to both views of a method:

```text
s_m = median(D_GT_view1) / median(D_m_view1)
D_m_view1_aligned = s_m * D_m_view1
D_m_view2_aligned = s_m * D_m_view2
```

This is also a scale-only alignment, but differs from the current package
because the current protocol explicitly restricts the median calculation to
the valid GT/prediction overlap for every method and applies the result to its
3D point cloud as well.

## Old Geometry-Only Display Normalization Is Not GT Alignment

The earlier `geometry_gt_minmax_display_normalized_20260727` package did **not**
align predicted depth or point-cloud scale to GT. It only changed point-cloud
coordinates for display:

```text
P_m_display = a_m * (P_m - center_m) + center_GT
```

where `a_m` makes each cloud occupy a similar robust visual extent. Its colors
were computed from the raw method Z-depth using the GT color range. This makes
the objects easy to inspect side by side, but it is a presentation transform,
not a valid geometric alignment and must not be described as GT-aligned.

## Shared-Scale Depth Figures Are Different Again

The shared-scale depth figures use a single scalar derived from Night3R and
apply that unchanged to every method. They intentionally retain relative
absolute-scale errors across methods. They are suitable for a scale-stability
qualitative figure, not for the aligned local-shape comparison above.

## Do Not Mix These Claims

- Use **GT-median-aligned geometry** to discuss local geometric structure.
- Use **shared-scale depth figures** or non-aligned metrics to discuss absolute
  scale robustness.
- Do not call display-only normalization an alignment.
- Do not compare colors between figures that use different color-range rules.

## Generated GT-Aligned Package

- Directory: `outputs/shared_scale_comparison_selected_20260727/gt_median_aligned_geometry_depth_20260728/`
- Each scene-level `manifest.json` records the method scales, valid overlap
  counts, shared camera parameters, and GT-derived color range.
