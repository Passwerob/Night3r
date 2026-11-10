#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

SOURCE="outputs/paper_main_geometry_heldout_dsec_20260721"
SCENE="test/DSEC/screen-000004-dsec"
PACKAGE_NAME="night3r_heldout_dsec_zurich11b_ppt_assets_20260721"
PACKAGE="outputs/${PACKAGE_NAME}"
ARCHIVE="outputs/${PACKAGE_NAME}.zip"

if [[ "${UPDATE_EXISTING:-0}" != "1" && ( -e "${PACKAGE}" || -e "${ARCHIVE}" ) ]]; then
  echo "Refusing to overwrite an existing package: ${PACKAGE} or ${ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${PACKAGE}/00_Overview"
cp docs/heldout_dsec_ppt_assets_readme_20260721.md "${PACKAGE}/README.md"
cp "${SOURCE}/composed/main_rgb_event_geometry_comparison_white.png" \
  "${PACKAGE}/00_Overview/01_RGB_Event_Geometry_Overview.png"
cp "${SOURCE}/composed/main_geometry_comparison_white.png" \
  "${PACKAGE}/00_Overview/02_Geometry_Only_Overview.png"
cp "${SOURCE}/composed/selection_metrics.csv" \
  "${PACKAGE}/00_Overview/03_Raw_AbsRel.csv"
cp "${SOURCE}/composed/main_figure_caption.tex" \
  "${PACKAGE}/00_Overview/04_Figure_Caption.tex"
cp "${SCENE}/meta/test_overlap_audit.json" \
  "${PACKAGE}/00_Overview/05_Split_Overlap_Audit.json"

pairs=(120 360 600 840)
labels=(01 02 03 04)

for slot in "${!pairs[@]}"; do
  idx1="${pairs[$slot]}"
  idx2="$((idx1 + 1))"
  label="${labels[$slot]}"
  source_pair="${SOURCE}/pair_$(printf '%03d' "${idx1}")_$(printf '%03d' "${idx2}")"
  target="${PACKAGE}/${label}_Pair_$(printf '%03d' "${idx1}")_$(printf '%03d' "${idx2}")"
  mkdir -p "${target}"

  cp "${SCENE}/images/$(printf '%06d' "${idx1}").png" "${target}/01_View1_CleanRGB.png"
  cp "${SCENE}/images/$(printf '%06d' "${idx2}").png" "${target}/02_View2_CleanRGB.png"
  cp "${source_pair}/single_images/view01/01_RGB_native.png" "${target}/03_View1_LowLightRGB_e010.png"
  cp "${source_pair}/single_images/view02/01_RGB_native.png" "${target}/04_View2_LowLightRGB_e010.png"
  cp "${source_pair}/single_images/view01/02_Event_recent4bins_red_blue.png" "${target}/05_View1_Event_Recent4Bins.png"
  cp "${source_pair}/single_images/view02/02_Event_recent4bins_red_blue.png" "${target}/06_View2_Event_Recent4Bins.png"

  cp "${source_pair}/single_images/view01/03_MASt3R_depth_median_aligned.png" "${target}/07_View1_MASt3R_Depth_MedianAligned.png"
  cp "${source_pair}/single_images/view01/04_Night3R_depth_median_aligned.png" "${target}/08_View1_Night3R_Depth_MedianAligned.png"
  cp "${source_pair}/single_images/view01/05_GT_depth.png" "${target}/09_View1_GT_Depth.png"

  cp "${source_pair}/geometry_single_images_white/mast3r_front.png" "${target}/10_MASt3R_3D_Front_RawScale.png"
  cp "${source_pair}/geometry_single_images_white/night3r_front.png" "${target}/11_Night3R_3D_Front_RawScale.png"
  cp "${source_pair}/geometry_single_images_white/ground_truth_front.png" "${target}/12_GT_3D_Front.png"
  cp "${source_pair}/geometry_single_images_white/mast3r_elevated.png" "${target}/13_MASt3R_3D_Elevated_RawScale.png"
  cp "${source_pair}/geometry_single_images_white/night3r_elevated.png" "${target}/14_Night3R_3D_Elevated_RawScale.png"
  cp "${source_pair}/geometry_single_images_white/ground_truth_elevated.png" "${target}/15_GT_3D_Elevated.png"

  cp "${source_pair}/comparison_pair_geometry_same_view_white.png" "${target}/16_3D_Comparison_Front_Elevated.png"
  cp "${source_pair}/comparison_pair_geometry_elevated_same_view_white.png" "${target}/17_3D_Comparison_Elevated.png"
  cp "${source_pair}/comparison_pair_depths.png" "${target}/18_RGB_Event_Depth_Comparison.png"
  cp "${source_pair}/comparison_pair_errors.png" "${target}/19_MedianAligned_Error_Comparison.png"
  cp "${source_pair}/manifest.json" "${target}/20_Metrics_And_Provenance.json"
done

(
  cd outputs
  zip -q -r "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
)

unzip -t "${ARCHIVE}" >/dev/null
echo "${ARCHIVE}"
