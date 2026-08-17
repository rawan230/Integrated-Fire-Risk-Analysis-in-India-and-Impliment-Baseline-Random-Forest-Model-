# 🔥🌿🌡️🗺️🎯 Integrated Fire-Risk Alignment + Susceptibility Model

**Notebooks:** [`Step5_Integrated_FireRisk_Analysis.ipynb`](Step5_Integrated_FireRisk_Analysis.ipynb), [`Step6_FireRisk_Susceptibility_Model.ipynb`](Step6_FireRisk_Susceptibility_Model.ipynb)
**Kernel:** `firerisk-anaconda3` (Python 3.12.7, base `C:\Users\Admin\anaconda3\python.exe`)

> **Renumbered 2026-08-17**: these notebooks were `Step4_...`/`Step5_...` before — renamed
> to `Step5_...`/`Step6_...` so training is genuinely the last step in both execution order
> and documentation labels. FLDAS (previously "Step 6") is now Step 4. No content, code, or
> results changed, only the labels and filenames.

## Step 5 — Integrated Multi-Factor Fire-Risk Feature Alignment

Combines every other step's per-pixel feature rasters — already on (or reprojected onto)
the **NDVI grid** (3641×3504, EPSG:4326, ~0.01°/1km, established in Step 2) — into one
aligned dataset:

| Source | Features |
|---|---|
| Step 1 (fire points) | fire count + binary fire-ever label, rasterized onto the grid |
| Step 2 (NDVI) | 9 features: QA mean, climatology, anomaly, trend, residual, Mann-Kendall τ, CVSI (`ndvi_cvsi_k8`, k=8 optimal lag), LISA cluster, breakpoint threshold |
| Step 3 (LST) | 5 features: Day/Night anomaly, DTR anomaly, monthly Mann-Kendall τ (Day/Night) |
| Step 4 (FLDAS) | 12 features: air temp/wind/precip/RH/soil moisture/net LW radiation — anomaly + monthly Mann-Kendall τ each |
| Step 4 (land cover) | 22 features: ESA CCI/C3S 2020 base-class fractional cover per pixel |
| This notebook | 4 features: LULC forest fraction (2001/2020/2022) + forest loss — the one input not aligned anywhere else |

**Total: 54 feature layers**, plus `lon`/`lat` = 56 columns in the flattened pixel table.

**2026-08-15 fixes:**
- **CVSI k6 → k8**: Step 2's CVSI optimal-lag was corrected from k=6 to k=8 after extending
  and properly resolving the mutual-information sweep (k=8 is a confirmed interior optimum,
  not a boundary artifact). Step 5's NDVI-loading cell now reads `F7_CVSI_k8.tif` into a
  `ndvi_cvsi_k8` column (renamed from `ndvi_cvsi_k6` / `F7_CVSI_k6.tif`).
- **Forest-class reconciliation**: Step 5's `FOREST_CODES` set (used to build
  `forest_frac_baseline/recent/current` and `forest_loss_baseline_to_recent`) previously
  omitted LCCS codes 100 (`mosaic_tree_and_shrub`) and 110 (`mosaic_herbaceous`) — an
  undocumented divergence from Step 1's own `FOREST_CODES`, which includes them per
  Sannigrahi et al. 2018 (cited via Biswas et al. 2025, p.4863). Step 5 now uses the same
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
wired in as of this run — this was the pending gap flagged in Step 4/FLDAS's own closing
note ("wire into `Integrated_Analysis/Step5`"), closed by adding a Step 4b cell that loads
FLDAS's `NDVI_Aligned_GeoTIFFs/*.tif` and the 22-band land-cover GeoTIFF the same way the
existing LST-loading cell works.

## Step 6 — Fire Susceptibility Model + Reproducibility Report

**Input:** Step 5's `Integrated_FireRisk_Pixels.parquet` — dynamically picks up every
feature column present (`feature_cols = [c for c in df.columns if c not in DROP_COLS]`),
so it automatically retrained on the full 52-feature set (56 columns minus `lon`, `lat`,
`fire_count`, `fire_ever`) once Step 5 was expanded with FLDAS + land cover.

Delivers:
1. A Random Forest fire-susceptibility classifier evaluated on a held-out test set
   (ROC-AUC, Average Precision, confusion matrix).
2. **Reproducibility evidence**: same fixed seed re-trained and compared bit-for-bit, plus
   5-fold cross-validation to show AUC stability across data subsets.
3. **Computational cost accounting**: wall-clock time and memory for every stage.
4. A full-country fire susceptibility probability map (`Fire_Susceptibility_Probability.tif`).
5. A real, trained **MaxEnt baseline** (`elapid`), evaluated on the identical held-out test
   set, as a direct methodological comparison against Biswas et al. (2025) — the paper this
   pipeline extends, which itself uses MaxEnt for India forest-fire susceptibility mapping.

### Results (retrained 2026-08-15 on the 52-feature set, after the forest-class + CVSI k8 fixes below)

| Metric | Value |
|---|---|
| ROC-AUC (held-out test) | **0.9674** |
| Average Precision | 0.6761 (no-skill baseline = 0.0649) |
| 5-fold CV mean AUC | 0.9670 ± 0.0002 (CV = 0.02% — stable across folds) |
| Bit-exact reproducibility | max abs diff 8.88e-16 (float64 summation-order noise from `n_jobs=-1`, not a real divergence) |
| Train / test split | 3,328,807 / 832,202 pixels (80/20, stratified, 6.49% fire rate both sides) |
| Training time | 188.1 sec (200 trees, max depth 20, 24 cores) |

**2026-08-15 re-run note:** this result reflects two upstream Step 5 fixes — (1) the
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

### MaxEnt baseline (Biswas et al. 2025 comparison, added 2026-08-17)

This pipeline explicitly extends **Biswas, U., Mahato, S., & Joshi, P.K. (2025)**, the
paper cited at the bottom of this README, which uses **MaxEnt (Maximum Entropy)** for
forest-fire susceptibility mapping in India. Until this run, no direct MaxEnt comparison
existed anywhere in this project. Step 6 now trains a real MaxEnt model
(`elapid.MaxentModel` v1.0.4 — a scikit-learn-compatible reimplementation of Phillips et
al.'s algorithm, `feature_types=['linear','hinge','product']`, cloglog transform) on this
project's own 52-feature table and evaluates it on the **identical held-out test set**
(832,202 pixels) as the Random Forest above — a direct, apples-to-apples comparison, not a
citation of Biswas et al.'s own reported numbers (which would be a weaker comparison given
their different feature set/resolution).

**MaxEnt is classically trained on presence + background samples, not exhaustive
full-image training** — standard practice in the species/habitat distribution modeling
literature MaxEnt comes from. A preliminary 40,000-row benchmark measured 202.6 sec, naively
extrapolating to ~40–50 min for an originally-planned 450,000-row training subsample.
**A direct timing probe on this project's own data disproved that extrapolation**: MaxEnt's
fit time scales *super-linearly* with sample size here (20k → 356 rows/sec, 40k → ~275,
80k → ~205, 150k → 121, 250k → 67 rows/sec) — a 450,000-row fit did not finish within a
7,200 sec (2 hr) cell timeout. Training size was recalibrated down to **150,000 rows**
(confirmed ~20.6 min fit time), a disclosed deviation from the original target, not a
silent downgrade — still a large stratified presence/background sample by the standards of
the literature this method comes from.

| Metric | Random Forest (headline) | MaxEnt (elapid) |
|---|---|---|
| ROC-AUC (held-out test, 832,202 px) | **0.9674** | 0.9576 |
| Average Precision | **0.6761** | 0.6111 |
| Training rows | 3,328,807 (100% of train split) | 150,000 (4.51% of train split, stratified) |
| Training time | 195.2 sec | 1,396.4 sec (23.3 min) |
| Test-set inference time | 1.4 sec | 33.3 sec |

Random Forest outperforms MaxEnt on this feature set by 0.0098 ROC-AUC (0.9674 vs. 0.9576)
and 0.0650 Average Precision — a modest but consistent RF advantage, plausible given RF
trains on the full 3.3M-row training set with a more flexible (non-linear, non-additive)
decision boundary, while MaxEnt here is a linear/hinge/product-feature exponential-family
model trained on a 4.5%-of-training-set stratified subsample. This is reported as a
straightforward measured result, not tuned in either model's favor. Outputs:
`ROC_PR_Curves_RF_vs_MaxEnt.png`, `RF_vs_MaxEnt_Comparison.csv`,
`MaxEnt_Feature_Importance.png` (permutation importance), `MaxEnt_Susceptibility_Probability.tif`,
`Fire_Susceptibility_Map_RF_vs_MaxEnt.png`.

This model is **not a preprocessing dependency for a PINN** — nothing downstream reads its
outputs. It's kept deliberately as a standalone classical-ML baseline (now with two
reference points — Random Forest and MaxEnt) to compare a future PINN against (decision
made 2026-08-04; see the `integrated-fire-risk-model` skill for the full reasoning). Step 5,
by contrast, *is* necessary preprocessing — it's the one place LULC features get built and
every other step's output gets assembled into a single ML-ready table, which a PINN needs
regardless of model architecture.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=1800 "Step5_Integrated_FireRisk_Analysis.ipynb"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=3600 "Step6_FireRisk_Susceptibility_Model.ipynb"
```

Step 5 requires Steps 1, 2, 3, and 4 to have already run (reads their outputs directly).
Step 6 requires Step 5's parquet. Step 6's total wall time is now ~50 min (measured
2026-08-17: 2,996.7 sec) — up from ~22 min before the MaxEnt baseline was added, since
MaxEnt's 150,000-row fit alone takes ~23.3 min; the `--ExecutePreprocessor.timeout` above
is a per-cell limit, not a total-notebook limit, and 3600 sec comfortably covers the
MaxEnt training cell with margin.

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
├── Fire_Susceptibility_Probability.tif         # full-country probability map, Random Forest (not tracked)
├── MaxEnt_Susceptibility_Probability.tif       # full-country probability map, MaxEnt (not tracked)
├── Feature_Importance.png                      # tracked -- Random Forest Gini importance
├── MaxEnt_Feature_Importance.png               # tracked -- MaxEnt permutation importance
├── Fire_Susceptibility_Map.png                 # tracked
├── Fire_Susceptibility_Map_RF_vs_MaxEnt.png    # tracked -- side-by-side maps, both models
├── ROC_PR_Curves.png                           # tracked
├── ROC_PR_Curves_RF_vs_MaxEnt.png              # tracked -- overlaid ROC/PR, both models
├── RF_vs_MaxEnt_Comparison.csv                 # tracked -- AUC/AP/timing/sample-size table
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
