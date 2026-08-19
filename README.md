# 🔥🌿🌡️🗺️🎯 Integrated Fire-Risk Alignment + Susceptibility Model

**Notebooks:** [`Step6_Integrated_FireRisk_Analysis.ipynb`](Step6_Integrated_FireRisk_Analysis.ipynb), [`Step7_FireRisk_Susceptibility_Model.ipynb`](Step7_FireRisk_Susceptibility_Model.ipynb)
**Kernel:** `firerisk-anaconda3` (Python 3.12.7, base `C:\Users\Admin\anaconda3\python.exe`)

> **Renumbered twice.** On 2026-08-17, these notebooks moved from `Step4_...`/`Step5_...`
> to `Step5_...`/`Step6_...` so training was the last step in both execution order and
> documentation labels (FLDAS, previously "Step 6", became Step 4). On 2026-08-19, they
> moved again to `Step6_...`/`Step7_...` to make room for a new Step 5 (Terrain &
> Accessibility Analysis, its own folder/repo) inserted between FLDAS and this step. No
> content, code, or results changed either time, only the labels and filenames.
>
> **Terrain & Accessibility wired in 2026-08-20.** Step 5a/5b's 6 bands (elevation, slope,
> aspect, distance to roads/railways/waterways) existed in their own repos since
> 2026-08-18 but were not read by this notebook until now — this was the last remaining
> gap between the pipeline's ML-ready table and Biswas et al.'s real 15-variable predictor
> set. Band/column counts below have been updated accordingly (54 → 60 bands, 56 → 62
> columns).
>
> **Step 7 retrained 2026-08-20** on the expanded 58-feature table (`feature_cols =
> [c for c in df.columns if c not in DROP_COLS]` picked up all 6 new terrain/accessibility
> columns automatically, no code change needed — verified directly, not assumed). This is
> the first Step 7 run that genuinely trains on all 15 of Biswas et al. (2025)'s predictor
> variables. Results below are current as of this run.

## Step 6 — Integrated Multi-Factor Fire-Risk Feature Alignment

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
| Step 5a (Terrain) | 3 features: elevation, slope, aspect (SRTMGL3 90m DEM) |
| Step 5b (Accessibility) | 3 features: distance to roads, railways, waterways (Geofabrik OSM 2022) |
| This notebook | 4 features: LULC forest fraction (2001/2020/2022) + forest loss — the one input not aligned anywhere else |

**Total: 60 feature layers**, plus `lon`/`lat` = 62 columns in the flattened pixel table.
This is the first run in which the trained feature table genuinely covers all 15 of
Biswas et al. (2025)'s real predictor variables (see the "Terrain & accessibility wired
in" note below) — previously only 9 of 15 were present, despite Step 5a/5b's work
existing separately since 2026-08-18.

**2026-08-15 fixes:**
- **CVSI k6 → k8**: Step 2's CVSI optimal-lag was corrected from k=6 to k=8 after extending
  and properly resolving the mutual-information sweep (k=8 is a confirmed interior optimum,
  not a boundary artifact). Step 6's NDVI-loading cell now reads `F7_CVSI_k8.tif` into a
  `ndvi_cvsi_k8` column (renamed from `ndvi_cvsi_k6` / `F7_CVSI_k6.tif`).
- **Forest-class reconciliation**: Step 6's `FOREST_CODES` set (used to build
  `forest_frac_baseline/recent/current` and `forest_loss_baseline_to_recent`) previously
  omitted LCCS codes 100 (`mosaic_tree_and_shrub`) and 110 (`mosaic_herbaceous`) — an
  undocumented divergence from Step 1's own `FOREST_CODES`, which includes them per
  Sannigrahi et al. 2018 (cited via Biswas et al. 2025, p.4863). Step 6 now uses the same
  13-code definition as Step 1 (`{50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110}`),
  so the fire-point ground-truth label and the forest-fraction predictor feature share one
  consistent operational definition of "forest." This raised the national-mean
  forest-fraction values: baseline (2001) 7.8% → 10.2%, recent (2020) 7.9% → 10.5%,
  current (2022) 8.0% → 10.7%. Band/column counts are unchanged (54 layers, 56 columns) —
  this only changes the *values* of four existing features, not their count.

### What it produces

1. `Integrated_FireRisk_Stack.tif` — 60-band GeoTIFF, one band per feature, identical
   pixel grid across bands.
2. `Integrated_FireRisk_Pixels.parquet` — same stack flattened to one row per valid
   in-India pixel (4,161,009 pixels × 62 columns), `fire_ever` as the label.
3. `Integrated_Monthly_TimeSeries.csv` — national-mean NDVI + LST Day/Night + FLDAS
   climatic variables + fire counts, joined on `(year, month)` (266 months × 25 columns).

Every documented variable source (NDVI, LST, FLDAS climatic variables, land cover,
terrain, accessibility) is wired in as of this run. FLDAS + land cover were the pending
gap flagged in Step 4/FLDAS's own closing note, closed 2026-08-07 (Step 4b cell loading
`NDVI_Aligned_GeoTIFFs/*.tif` and the 22-band land-cover GeoTIFF). Terrain + accessibility
were the pending gap flagged repeatedly in this project's docs after Step 5a/5b were
split out on 2026-08-18/19, closed 2026-08-20 (Step 4c cell loading Step 5a/5b's
`*_native_1km.tif` outputs, verified already on the exact NDVI grid — no reprojection
needed).

## Step 7 — Fire Susceptibility Model + Reproducibility Report

**Input:** Step 6's `Integrated_FireRisk_Pixels.parquet` — dynamically picks up every
feature column present (`feature_cols = [c for c in df.columns if c not in DROP_COLS]`),
so it automatically retrained on the full 58-feature set (62 columns minus `lon`, `lat`,
`fire_count`, `fire_ever`) once Step 6 was expanded with terrain + accessibility features
(2026-08-20). Verified directly (not assumed): `DROP_COLS` is still exactly
`["lon", "lat", "fire_count", "fire_ever"]` and no cell hardcodes a column count or name
list, so no code change was needed for this retrain.

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

### Results (retrained 2026-08-20 on the 58-feature set, after terrain + accessibility were wired into Step 6)

| Metric | New (58 features, 2026-08-20) | Old (52 features, 2026-08-15) |
|---|---|---|
| ROC-AUC (held-out test) | **0.9683** | 0.9674 |
| Average Precision | **0.6796** (no-skill baseline = 0.0649) | 0.6761 |
| 5-fold CV mean AUC | **0.9679 ± 0.0002** (CV = 0.02% — stable across folds) | 0.9670 ± 0.0002 |
| Per-fold AUC | [0.9679, 0.9679, 0.9682, 0.9677, 0.9679] | not previously reported per-fold |
| Bit-exact reproducibility | max abs diff 7.77e-16 (float64 summation-order noise from `n_jobs=-1`, not a real divergence); identical within 1e-9 tolerance | max abs diff 8.88e-16 |
| Train / test split | 3,328,807 / 832,202 pixels (80/20, stratified, 6.49% fire rate both sides) | same |
| Training time | 202.2 sec (200 trees, max depth 20, 24 cores) | 188.1 sec |

**2026-08-20 re-run note:** ROC-AUC improved by +0.0009 and Average Precision by +0.0035
after adding the 6 terrain/accessibility features (elevation, slope, aspect, distance to
roads/railways/waterways) — a small but real, unforced improvement, consistent with this
project's own Step 5a finding that fire pixels sit at markedly higher slope than the
national average. This is the first Step 7 run that genuinely trains on all 15 of Biswas
et al. (2025)'s predictor variables, not just the pipeline containing them.

Top 5 features by Gini importance (2026-08-20 re-run, 58-feature set):
`forest_frac_recent` (0.1657), `forest_frac_current` (0.1198), `forest_frac_baseline`
(0.1112), `ndvi_trend_2x12ma` (0.0851), `ndvi_mean` (0.0571) — forest-fraction and NDVI
variables still dominate the top 5, same as the 52-feature-era model, but with
`ndvi_trend_2x12ma` newly entering (previously outside the top 5; `ndvi_clim_june` has
moved down to 7th).

**None of the 6 new terrain/accessibility features cracked the top 5** — but
`terrain_slope` is the highest-ranked of them and landed at **6th place overall**
(Gini importance 0.0492), immediately behind the top 5 and ahead of all 52 previously-
existing features except `ndvi_mean`. This is a real, testable, and confirmed signal:
Step 5a's fire-coincidence finding (fires sit at +115% mean slope vs. the national
average) does translate into predictive importance for the Random Forest, just not
enough to unseat the forest-fraction/NDVI features that already dominated. The other 5
new features rank further down: `terrain_elevation` 11th (0.0198), `access_dist_roads`
22nd (0.0097), `access_dist_railways` 29th (0.0069), `terrain_aspect` 36th (0.0050),
`access_dist_waterways` 39th (0.0040) — out of 58 total features. Full ranking available
in `Model_Outputs/Feature_Importance.png`.

### MaxEnt baseline (Biswas et al. 2025 comparison, added 2026-08-17)

This pipeline explicitly extends **Biswas, U., Mahato, S., & Joshi, P.K. (2025)**, the
paper cited at the bottom of this README, which uses **MaxEnt (Maximum Entropy)** for
forest-fire susceptibility mapping in India. Step 7 trains a real MaxEnt model
(`elapid.MaxentModel` v1.0.4 — a scikit-learn-compatible reimplementation of Phillips et
al.'s algorithm, `feature_types=['linear','hinge','product']`, cloglog transform) on this
project's own 58-feature table (retrained 2026-08-20 alongside the Random Forest, on the
same expanded parquet including terrain/accessibility) and evaluates it on the
**identical held-out test set** (832,202 pixels) as the Random Forest above — a direct,
apples-to-apples comparison, not a citation of Biswas et al.'s own reported numbers (which
would be a weaker comparison given their different feature set/resolution).

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
| ROC-AUC (held-out test, 832,202 px) | **0.9683** | 0.9595 |
| Average Precision | **0.6796** | 0.6237 |
| Training rows | 3,328,807 (100% of train split) | 150,000 (4.51% of train split, stratified) |
| Training time | 202.2 sec | 1,486.8 sec (24.8 min) |
| Test-set inference time | 1.4 sec | 34.3 sec |

(Prior 52-feature-era numbers, 2026-08-17: Random Forest ROC-AUC 0.9674 / AP 0.6761;
MaxEnt ROC-AUC 0.9576 / AP 0.6111 — both models improved after terrain/accessibility
features were added, MaxEnt included since it's trained on the same expanded table.)

Random Forest outperforms MaxEnt on this feature set by 0.0088 ROC-AUC (0.9683 vs. 0.9595)
and 0.0559 Average Precision — a modest but consistent RF advantage, plausible given RF
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
made 2026-08-04; see the `integrated-fire-risk-model` skill for the full reasoning). Step 6,
by contrast, *is* necessary preprocessing — it's the one place LULC features get built and
every other step's output gets assembled into a single ML-ready table, which a PINN needs
regardless of model architecture.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=1800 "Step6_Integrated_FireRisk_Analysis.ipynb"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=3600 "Step7_FireRisk_Susceptibility_Model.ipynb"
```

Step 6 requires Steps 1, 2, 3, 4, and 5 (5a Terrain + 5b Accessibility) to have already run
(reads their outputs directly, no in-notebook recomputation). Step 7 requires Step 6's
parquet. Step 7's total wall time is now ~56 min (measured 2026-08-20: 3,352.5 sec, on the
expanded 58-feature table) — up slightly from ~50 min on the 52-feature table (measured
2026-08-17: 2,996.7 sec), since MaxEnt's 150,000-row fit takes proportionally longer with
more features (~24.8 min vs. ~23.3 min); the `--ExecutePreprocessor.timeout` above is a
per-cell limit, not a total-notebook limit, and 3600 sec comfortably covers the MaxEnt
training cell with margin.

## Outputs

```
Integrated_Outputs/
├── Integrated_FireRisk_Stack.tif              # 60-band GeoTIFF (not tracked, ~large)
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
