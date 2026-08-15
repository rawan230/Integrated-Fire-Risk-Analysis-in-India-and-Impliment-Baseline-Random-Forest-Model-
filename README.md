# 🔥🌿🌡️🗺️🎯 Integrated Fire-Risk Alignment + Susceptibility Model

**Notebooks:** [`Step4_Integrated_FireRisk_Analysis.ipynb`](Step4_Integrated_FireRisk_Analysis.ipynb), [`Step5_FireRisk_Susceptibility_Model.ipynb`](Step5_FireRisk_Susceptibility_Model.ipynb)
**Kernel:** `firerisk-anaconda3` (Python 3.12.7, base `C:\Users\Admin\anaconda3\python.exe`)

## Step 4 — Integrated Multi-Factor Fire-Risk Feature Alignment

Combines every other step's per-pixel feature rasters — already on (or reprojected onto)
the **NDVI grid** (3641×3504, EPSG:4326, ~0.01°/1km, established in Step 2) — into one
aligned dataset:

| Source | Features |
|---|---|
| Step 1 (fire points) | fire count + binary fire-ever label, rasterized onto the grid |
| Step 2 (NDVI) | 9 features: QA mean, climatology, anomaly, trend, residual, Mann-Kendall τ, CVSI (`ndvi_cvsi_k8`, k=8 optimal lag), LISA cluster, breakpoint threshold |
| Step 3 (LST) | 5 features: Day/Night anomaly, DTR anomaly, monthly Mann-Kendall τ (Day/Night) |
| Step 6 (FLDAS) | 12 features: air temp/wind/precip/RH/soil moisture/net LW radiation — anomaly + monthly Mann-Kendall τ each |
| Step 6 (land cover) | 22 features: ESA CCI/C3S 2020 base-class fractional cover per pixel |
| This notebook | 4 features: LULC forest fraction (2001/2020/2022) + forest loss — the one input not aligned anywhere else |

**Total: 54 feature layers**, plus `lon`/`lat` = 56 columns in the flattened pixel table.

**2026-08-15 fixes:**
- **CVSI k6 → k8**: Step 2's CVSI optimal-lag was corrected from k=6 to k=8 after extending
  and properly resolving the mutual-information sweep (k=8 is a confirmed interior optimum,
  not a boundary artifact). Step 4's NDVI-loading cell now reads `F7_CVSI_k8.tif` into a
  `ndvi_cvsi_k8` column (renamed from `ndvi_cvsi_k6` / `F7_CVSI_k6.tif`).
- **Forest-class reconciliation**: Step 4's `FOREST_CODES` set (used to build
  `forest_frac_baseline/recent/current` and `forest_loss_baseline_to_recent`) previously
  omitted LCCS codes 100 (`mosaic_tree_and_shrub`) and 110 (`mosaic_herbaceous`) — an
  undocumented divergence from Step 1's own `FOREST_CODES`, which includes them per
  Sannigrahi et al. 2018 (cited via Biswas et al. 2025, p.4863). Step 4 now uses the same
  13-code definition as Step 1 (`{50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110}`),
  so the fire-point ground-truth label and the forest-fraction predictor feature share one
  consistent operational definition of "forest." This raised the national-mean
  forest-fraction values: baseline (2001) 7.8% → 10.2%, recent (2020) 7.9% → 10.5%,
  current (2022) 8.0% → 10.7%. Band/column counts are unchanged (54 layers, 56 columns) —
  this only changes the *values* of four existing features, not their count.

### What it produces

1. `Integrated_FireRisk_Stack.tif` — 54-band GeoTIFF, one band per feature, identical
   pixel grid across bands.
2. `Integrated_FireRisk_Pixels.parquet` — same stack flattened to one row per valid
   in-India pixel (4,161,009 pixels × 56 columns), `fire_ever` as the label.
3. `Integrated_Monthly_TimeSeries.csv` — national-mean NDVI + LST Day/Night + FLDAS
   climatic variables + fire counts, joined on `(year, month)` (266 months × 25 columns).

Every documented variable source (NDVI, LST, FLDAS climatic variables, land cover) is
wired in as of this run — this was the pending gap flagged in Step 6/FLDAS's own closing
note ("wire into `Integrated_Analysis/Step4`"), closed by adding a Step 4b cell that loads
FLDAS's `NDVI_Aligned_GeoTIFFs/*.tif` and the 22-band land-cover GeoTIFF the same way the
existing LST-loading cell works.

## Step 5 — Fire Susceptibility Model + Reproducibility Report

**Input:** Step 4's `Integrated_FireRisk_Pixels.parquet` — dynamically picks up every
feature column present (`feature_cols = [c for c in df.columns if c not in DROP_COLS]`),
so it automatically retrained on the full 52-feature set (56 columns minus `lon`, `lat`,
`fire_count`, `fire_ever`) once Step 4 was expanded with FLDAS + land cover.

Delivers:
1. A Random Forest fire-susceptibility classifier evaluated on a held-out test set
   (ROC-AUC, Average Precision, confusion matrix).
2. **Reproducibility evidence**: same fixed seed re-trained and compared bit-for-bit, plus
   5-fold cross-validation to show AUC stability across data subsets.
3. **Computational cost accounting**: wall-clock time and memory for every stage.
4. A full-country fire susceptibility probability map (`Fire_Susceptibility_Probability.tif`).

### Results (retrained 2026-08-15 on the 52-feature set, after the forest-class + CVSI k8 fixes below)

| Metric | Value |
|---|---|
| ROC-AUC (held-out test) | **0.9674** |
| Average Precision | 0.6761 (no-skill baseline = 0.0649) |
| 5-fold CV mean AUC | 0.9670 ± 0.0002 (CV = 0.02% — stable across folds) |
| Bit-exact reproducibility | max abs diff 8.88e-16 (float64 summation-order noise from `n_jobs=-1`, not a real divergence) |
| Train / test split | 3,328,807 / 832,202 pixels (80/20, stratified, 6.49% fire rate both sides) |
| Training time | 188.1 sec (200 trees, max depth 20, 24 cores) |

**2026-08-15 re-run note:** this result reflects two upstream Step 4 fixes — (1) the
`FOREST_CODES` reconciliation to Step 1's 13-code definition (adds LCCS 100/110 to the
forest-fraction features), and (2) the CVSI feature rename from `ndvi_cvsi_k6` to
`ndvi_cvsi_k8` (Step 2's corrected optimal lag). Headline metrics moved by <0.001 (ROC-AUC
0.9676 → 0.9674, AP 0.6765 → 0.6761, CV mean 0.9671 → 0.9670 ± 0.0002) — not a red flag,
just the model re-fit on slightly revised feature values. The more notable shift is in
**feature importance**: `forest_frac_recent`/`forest_frac_current`/`forest_frac_baseline`
jumped from mid-pack (~0.08–0.11) to the top 3 (0.160, 0.141, 0.112 respectively, ahead of
`ndvi_mean` at 0.094), consistent with the corrected forest definition now capturing more
fire-relevant forest area via the mosaic classes.

**2026-08-15 fix:** the 5-fold CV cell previously re-fit with `n_estimators=100` (half the
trees of the headline `rf_model`, undocumented), so the reported CV stability numbers
technically described a different, weaker model than the one whose AUC was reported as the
headline result. The CV cell now uses identical hyperparameters to the reported model
(`n_estimators=200, max_depth=20, min_samples_leaf=5, class_weight="balanced",
random_state=42`) — a true apples-to-apples cross-validation. The corrected numbers round
to the same 0.9671 ± 0.0002 as before the fix (per-fold AUCs shifted by <0.0002), so the
earlier CV conclusion was directionally fine, but it's now backed by evidence about the
actual reported model rather than a cheaper stand-in. This roughly doubled the CV cell's
runtime (347.7 sec &rarr; 863.3 sec), pushing total notebook wall time from ~12.6 min to
~21.7 min.

Top 5 features by Gini importance (2026-08-15 re-run, post forest-class reconciliation):
`forest_frac_recent` (0.160), `forest_frac_current` (0.141), `forest_frac_baseline` (0.112),
`ndvi_mean` (0.094), `ndvi_clim_june` (0.072) — forest-fraction and NDVI variables
dominate; no single FLDAS or land-cover-class feature cracked the top 5, though they
contribute in aggregate (52 features total vs. 20 before the FLDAS/land-cover expansion).

This model is **not a preprocessing dependency for a PINN** — nothing downstream reads its
outputs. It's kept deliberately as a standalone classical-ML baseline to compare a future
PINN against (decision made 2026-08-04; see the `integrated-fire-risk-model` skill for the
full reasoning). Step 4, by contrast, *is* necessary preprocessing — it's the one place
LULC features get built and every other step's output gets assembled into a single
ML-ready table, which a PINN needs regardless of model architecture.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=1800 "Step4_Integrated_FireRisk_Analysis.ipynb"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=1800 "Step5_FireRisk_Susceptibility_Model.ipynb"
```

Step 4 requires Steps 1, 2, 3, and 6 to have already run (reads their outputs directly).
Step 5 requires Step 4's parquet.

## Outputs

```
Integrated_Outputs/
├── Integrated_FireRisk_Stack.tif              # 54-band GeoTIFF (not tracked, ~large)
├── Integrated_FireRisk_Pixels.parquet          # ML-ready table (not tracked, ~large)
├── Integrated_FireRisk_Pixels_sample200k.csv   # 200k-row sample (tracked)
├── Integrated_Monthly_TimeSeries.csv           # monthly join table (tracked)
├── Feature_Correlation_Matrix.png              # tracked
└── Fire_vs_NoFire_Feature_Distributions.png    # tracked

Model_Outputs/
├── Fire_Susceptibility_Probability.tif         # full-country probability map (not tracked)
├── Feature_Importance.png                      # tracked
├── Fire_Susceptibility_Map.png                 # tracked
├── ROC_PR_Curves.png                           # tracked
├── Computational_Cost_Dashboard.png             # tracked
└── Computational_Cost_Reproducibility_Report.json  # tracked
```

## Citation

- Biswas, U., Mahato, S., & Joshi, P.K. (2025). Spatial prediction of forest fires
  in India: a machine learning approach for improved risk assessment and early
  warning systems. *Environmental Science and Pollution Research*, 32(8), 4856–4878.
  DOI: 10.1007/s11356-025-35982-8. (Verified via Crossref, see `METHODOLOGY.md`.)

## License

No license has been chosen yet for this repository's code.
