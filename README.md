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
> **Step 7 first retrained 2026-08-20** on the expanded 58-feature table (`feature_cols =
> [c for c in df.columns if c not in DROP_COLS]` picked up all 6 new terrain/accessibility
> columns automatically, no code change needed — verified directly, not assumed). This was
> the first Step 7 run that genuinely trained on all 15 of Biswas et al. (2025)'s predictor
> variables — since superseded by the two fixes below.
>
> **Data-leakage fix, 2026-08-21.** A literature-grounded audit found that
> `forest_frac_recent` (2020) and `forest_frac_current` (2022) — two of the model's
> top-3 Gini-importance features, combined ~0.40 importance with `forest_frac_baseline` —
> fall *inside* the same 2000-11-01–2022-12-15 window that the static, pooled `fire_ever`
> label spans. Published post-fire land-cover-change literature (ESA-CCI 300m + MODIS
> burned area) documents burned-forest pixels being reclassified to shrubland/agriculture
> in *subsequent* LULC epochs — i.e. these features could partly encode the outcome of
> fire, not a pre-fire risk condition. Step 6 now exports **only `forest_frac_baseline`
> (2001)** as the forest-fraction feature; `forest_frac_recent`, `forest_frac_current`,
> and `forest_loss_baseline_to_recent` were dropped from the parquet/GeoTIFF (60 → 57
> bands, 62 → 59 columns, 58 → 55 features). See the dedicated markdown cell in
> `Step6_Integrated_FireRisk_Analysis.ipynb`'s LULC section for the full reasoning. The
> CDR-PINN work (`Physics_Informed_FireRisk_Model/`) also pulled its `forest_frac`
> covariate from the now-removed `forest_frac_recent` column and needed a corresponding
> update (tracked separately in that repo).
>
> **RF hyperparameter tuning, 2026-08-21/22.** `hp_search_rf.py` (this folder) ran a
> genuine validation-set search over the Random Forest's two overfitting-control knobs,
> using `preprocessing.py`'s 65/15/20 train/val/test split (selection touches validation
> only; test is scored once, for the winner). The old literature-default config
> (`max_depth=20, min_samples_leaf=5`) scored validation AUC 0.9679; the winner
> (`max_depth=25, min_samples_leaf=3`) scored 0.9694 and confirmed on the untouched test
> split at ROC-AUC 0.9698 / AP 0.6961 (`Model_Outputs/rf_hp_search_result.json`).
> `n_estimators=200`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42` are
> unchanged. MaxEnt was deliberately left untuned in this pass — closed 2026-08-23 by
> `hp_search_maxent.py` (see below): the search found `beta_multiplier` barely matters
> for this feature set (validation AUC 0.9589-0.9592 across {0.5,1.0,1.5,2.5,4.0}), a
> genuine near-null tuning result; winner `beta_multiplier=4.0` is now used throughout.
>
> **Step 7 retrained again 2026-08-22**, this time on both fixes together: the corrected
> 55-feature (leak-fixed) table *and* the tuned RF hyperparameters. Every RF instantiation
> in the notebook (headline model, bit-exact re-run, 5-fold CV, spatial-block CV) now uses
> `max_depth=25, min_samples_leaf=3`. Clean execution, 0 cell errors. **Results below are
> current as of this run** — the STALE flag that previously sat here is resolved.
>
> **Specific-humidity completeness fix, 2026-08-22.** Step 4's FLDAS notebook added full
> spatial treatment (anomaly + monthly Mann-Kendall trend) for specific humidity
> (`Qair_f_tavg`), which previously only relative humidity received. Step 6 now loads the
> two resulting GeoTIFFs (`SpecificHumidity_anomaly_mean_on_NDVI_grid.tif`,
> `MannKendall_tau_SpecificHumidity_monthly_on_NDVI_grid.tif`) as `fldas_qair_anomaly` and
> `fldas_qair_mk_tau_monthly`, following the exact existing FLDAS dict pattern (same
> warn-and-skip-on-missing-file behavior). Purely additive — no existing feature was
> changed or removed — bringing the counts to 59 bands, 61 columns, 57 features (up from
> 57/59/55). **Downstream impact, since resolved**: Step 7
> (`Step7_FireRisk_Susceptibility_Model.ipynb`) picks up new columns automatically via its
> `feature_cols = [c for c in df.columns if c not in DROP_COLS]` pattern. At the time this
> note was first written, Step 7 had not yet been re-run against this expanded table; it
> has since been retrained on the full 57-feature table (RF ROC-AUC 0.9704, MaxEnt
> ROC-AUC 0.9598 with `beta_multiplier` validated-tuned 2026-08-23) — see "Current
> Headline Results" directly below and the Step 7 section's MaxEnt subsection for the
> up-to-date numbers. Any "still reflects the 55-feature run" language elsewhere in this
> document refers to a specific, explicitly-labeled historical table (kept for the
> before/after comparison it documents), not the current state.

## Current Headline Results (as of 2026-08-23 — final, validated numbers)

Both models below are trained on Step 6's 57-feature, leakage-fixed table
(4,161,009 pixels), both tuned via a genuine validation split rather than
literature defaults:

| Metric | Random Forest (`max_depth=25, min_samples_leaf=3`) | MaxEnt (`beta_multiplier=4.0`) |
|---|---|---|
| ROC-AUC (80/20 random split, held-out test) | **0.9704** | 0.9598 |
| Average Precision | **0.7011** | 0.6275 |
| Spatial-block CV AUC (2°×2° `GroupKFold`, n=3) | **0.9498 ± 0.0035** | 0.9465 ± 0.0054 |
| 5-fold `StratifiedKFold` CV AUC | **0.9698 ± 0.0002** | not run |

These are the numbers to cite as "current" anywhere in this document or a paper
draft. Any earlier RF/MaxEnt number elsewhere in this README (0.9674, 0.9683,
0.9687, or a 0.9701/0.6984 pair without spatial-block CV; an untuned MaxEnt
0.9594/0.9595) is a superseded intermediate result from this project's own
leak-fix/tuning history, kept in place only where the surrounding text is
documenting *that specific historical comparison* (e.g. "tuned vs. untuned"),
not presented as current. See `FULL_EXPERIMENT_LOG.md` (project root) Section B
for the complete, dated ledger of every run.

## Step 6 — Integrated Multi-Factor Fire-Risk Feature Alignment

### Why this step, and how

Steps 1–5 each produce a scientifically valid feature set on their own, but each
lives on a different native grid, resolution, and file format (fire points as
lon/lat CSV rows; NDVI/LST/FLDAS as monthly GeoTIFF stacks; terrain/accessibility
as static single-epoch rasters) — no model can train directly on five separate
files with five different pixel definitions. Step 6 exists purely to solve that
alignment problem: reproject/resample everything onto one shared grid (the NDVI
grid established in Step 2) and flatten it into one row-per-pixel table a
classical ML model or neural operator can actually consume. This is necessary
preprocessing regardless of which model architecture comes next — Step 7's Random
Forest/MaxEnt and Step 8's CDR-PINN both read this exact parquet file directly,
and neither performs its own alignment.

Combines every other step's per-pixel feature rasters — already on (or reprojected onto)
the **NDVI grid** (3641×3504, EPSG:4326, ~0.01°/1km, established in Step 2) — into one
aligned dataset:

| Source | Features |
|---|---|
| Step 1 (fire points) | fire count + binary fire-ever label, rasterized onto the grid |
| Step 2 (NDVI) | 9 features: QA mean, climatology, anomaly, trend, residual, Mann-Kendall τ, CVSI (`ndvi_cvsi_k8`, k=8 optimal lag), LISA cluster, breakpoint threshold |
| Step 3 (LST) | 5 features: Day/Night anomaly, DTR anomaly, monthly Mann-Kendall τ (Day/Night) |
| Step 4 (FLDAS) | 14 features: air temp/wind/precip/RH/specific humidity/soil moisture/net LW radiation — anomaly + monthly Mann-Kendall τ each |
| Step 4 (land cover) | 22 features: ESA CCI/C3S 2020 base-class fractional cover per pixel |
| Step 5a (Terrain) | 3 features: elevation, slope, aspect (SRTMGL3 90m DEM) |
| Step 5b (Accessibility) | 3 features: distance to roads, railways, waterways (Geofabrik OSM 2022) |
| This notebook | 1 feature: LULC forest fraction, 2001 baseline only (`forest_frac_baseline`) — the one input not aligned anywhere else |

**Total: 59 feature layers**, plus `lon`/`lat` = 61 columns in the flattened pixel table
(57 layers / 59 columns before the 2026-08-22 specific-humidity addition below).
This is the first run in which the trained feature table genuinely covers all 15 of
Biswas et al. (2025)'s real predictor variables (see the "Terrain & accessibility wired
in" note below) — previously only 9 of 15 were present, despite Step 5a/5b's work
existing separately since 2026-08-18.

**2026-08-22 specific-humidity completeness fix:** Step 4's FLDAS notebook added full
spatial treatment (anomaly + monthly Mann-Kendall trend) for specific humidity
(`Qair_f_tavg`), previously only given to relative humidity. The two new GeoTIFFs are
wired in here as `fldas_qair_anomaly` and `fldas_qair_mk_tau_monthly`, following the
identical dict pattern as every other FLDAS variable (same warn-and-skip-on-missing-file
behavior). Purely additive: no existing feature changed or was removed. This raises the
counts from 57 bands / 59 columns / 55 features to **59 bands / 61 columns / 57
features**.

**2026-08-21 data-leakage fix:** `forest_frac_recent` (2020), `forest_frac_current`
(2022), and `forest_loss_baseline_to_recent` were removed from the exported feature set
(60 → 57 bands, 62 → 59 columns). `fire_ever` is a single static label pooling every fire
from 2000-11-01 through 2022-12-15; the 2020/2022 LULC snapshots fall inside that same
window, and published post-fire land-cover-change literature documents burned forest
pixels commonly being reclassified to shrubland/agriculture in the *next* LULC epoch —
i.e. these features risked encoding the outcome of fire rather than a pre-fire condition.
This mattered because these three features were the model's top-3 Gini-importance
features pre-fix (combined ~0.40). Only `forest_frac_baseline` (2001) — the year closest
to a genuine pre-fire condition relative to the bulk of the study period — is kept, under
its existing name (no rename, to avoid a ripple into downstream references). The
underlying LULC-loading/reclassification code (`load_forest_fraction()`, the 2020/2022
reads) is untouched — only the recent/current/loss columns were dropped from the final
output. See the dedicated markdown cell in the notebook's LULC section for the full
reasoning.

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

1. `Integrated_FireRisk_Stack.tif` — 59-band GeoTIFF, one band per feature, identical
   pixel grid across bands.
2. `Integrated_FireRisk_Pixels.parquet` — same stack flattened to one row per valid
   in-India pixel (4,161,009 pixels × 61 columns), `fire_ever` as the label.
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

### Why this step, and how

With one aligned feature table in hand, Step 7 asks the actual scientific
question: can these features predict fire occurrence, and how well? Random
Forest is chosen as the primary baseline because it is a stronger, more modern
classical method than Biswas et al.'s own MaxEnt (native handling of confirmed
non-fire pixels as true negatives, a non-linear/non-additive decision boundary,
straightforward bit-exact reproducibility) — while a real, separately-trained
MaxEnt replication is kept alongside it specifically because it is the
literature-comparable baseline, the actual method Biswas et al. used. Both
models' hyperparameters are chosen via a genuine 65/15/20 train/validation/test
split (`hp_search_rf.py`, `hp_search_maxent.py`) rather than literature
defaults, because an untuned default configuration is not a fair representation
of either method's real ceiling on this data. Spatial-block cross-validation is
added on top of the standard random-split and 5-fold CV specifically because
random splits of spatially autocorrelated data systematically overstate
generalization (Roberts et al. 2017, cited in full below) — a model can score
well on a random held-out pixel sitting meters from a training pixel without
demonstrating it can generalize to genuinely new geography, the more
policy-relevant question for a national risk map. Step 7's outputs — the tuned
RF as the primary baseline, the tuned MaxEnt as the literature-comparable one,
and all three validation protocols — are what Step 8's CDR-PINN is benchmarked
against.

**Input:** Step 6's `Integrated_FireRisk_Pixels.parquet` — dynamically picks up every
feature column present (`feature_cols = [c for c in df.columns if c not in DROP_COLS]`),
so it automatically retrains on whatever's present without a code change. As of Step 7's
current run (2026-08-23), it trains on **57 features** (61 columns minus `lon`, `lat`,
`fire_count`, `fire_ever`), reflecting both Step 6's 2026-08-21 data-leakage fix (only
`forest_frac_baseline` kept; `forest_frac_recent`, `forest_frac_current`,
`forest_loss_baseline_to_recent` dropped) and the 2026-08-22 specific-humidity addition
(`fldas_qair_anomaly`, `fldas_qair_mk_tau_monthly`). Verified directly (not assumed):
`DROP_COLS` is still exactly `["lon", "lat", "fire_count", "fire_ever"]` and no cell
hardcodes a column count or name list. (An earlier version of this note, written
immediately after the specific-humidity fix landed in Step 6 but before Step 7 was
re-run against it, said Step 7 still reflected a 55-feature run — that has since been
resolved; see "Current Headline Results" above and the MaxEnt subsection below for the
57-feature numbers.)

Delivers:
1. A Random Forest fire-susceptibility classifier evaluated on a held-out test set
   (ROC-AUC, Average Precision, confusion matrix), with hyperparameters selected by a real
   validation-set search (see below), not literature defaults.
2. **Reproducibility evidence**: same fixed seed re-trained and compared bit-for-bit, plus
   5-fold cross-validation to show AUC stability across data subsets.
3. **Spatial-block cross-validation** (2°×2° `GroupKFold(n_splits=3)`) — a second,
   spatially-aware evaluation protocol, reported alongside the random-split numbers,
   directly comparable to CDR-PINN's own Track B1 spatial-generalization check.
4. **Computational cost accounting**: wall-clock time and memory for every stage.
5. A full-country fire susceptibility probability map (`Fire_Susceptibility_Probability.tif`).
6. A real, trained **MaxEnt baseline** (`elapid`), evaluated on the identical held-out test
   set, as a direct methodological comparison against Biswas et al. (2025) — the paper this
   pipeline extends, which itself uses MaxEnt for India forest-fire susceptibility mapping.

### Results (retrained 2026-08-22 — 55-feature leak-fixed table + tuned RF hyperparameters)

**RF hyperparameters are tuned, not literature defaults**: `max_depth=25,
min_samples_leaf=3`, selected by `hp_search_rf.py`'s validation-set search (see the
project-level note above). `n_estimators=200`, `class_weight="balanced"`, `n_jobs=-1`,
`random_state=42` unchanged.

*(Table below is the 2026-08-22 leak-fix + RF-tuning snapshot, 55 features. Since
superseded by specific humidity's addition, 57 features, RF 0.9704/0.7011 — see
the "Current Headline Results" section near the top of this README for the
up-to-date numbers; this table is kept for the specific 55→58-feature RF-tuning
comparison it documents.)*

| Metric | Then-current (55 features, tuned RF, 2026-08-22) | Prior (58 features, untuned RF, 2026-08-20) |
|---|---|---|
| ROC-AUC (held-out test) | **0.9701** | 0.9683 |
| Average Precision | **0.6984** (no-skill baseline = 0.0649) | 0.6796 |
| 5-fold CV mean AUC | **0.9698 ± 0.0002** (CV = 0.02% — stable across folds) | 0.9679 ± 0.0002 |
| Per-fold AUC | [0.9698, 0.9699, 0.9701, 0.9696, 0.9698] | [0.9679, 0.9679, 0.9682, 0.9677, 0.9679] |
| Spatial-block CV (2°×2° `GroupKFold`, n=3) | **0.9497 ± 0.0033** (folds: 0.9459, 0.9520, 0.9512) | not previously evaluated |
| Bit-exact reproducibility | max abs diff 7.77e-16 (float64 summation-order noise from `n_jobs=-1`, not a real divergence); identical within 1e-9 tolerance | max abs diff 7.77e-16 |
| Train / test split | 3,328,807 / 832,202 pixels (80/20, stratified, 6.49% fire rate both sides) | same |
| Training time | 217.5 sec (200 trees, max depth 25, min leaf 3, 24 cores) | 202.2 sec (max depth 20, min leaf 5) |

**Prior run's 58 → 55 features and 20 → 25 max depth are two independent changes landing
in the same retrain** — the leak-fix removed 3 features (`forest_frac_recent`,
`forest_frac_current`, `forest_loss_baseline_to_recent`) and the tuning changed
`max_depth`/`min_samples_leaf`. Both moved the headline metrics in the same (improving)
direction, so this table cannot cleanly attribute the +0.0018 ROC-AUC gain to one or the
other in isolation — the honest read is "removing 3 leaky features and tuning two
hyperparameters together improved test ROC-AUC by 0.0018 and AP by 0.0188," not a
controlled ablation of either change alone.

**Cross-check against `hp_search_rf.py`'s own numbers**: that script's winner
(`max_depth=25, min_samples_leaf=3`) scored test ROC-AUC 0.9698 / AP 0.6961 on its own
65/15/20 train/val/test split; this notebook's 80/20 split scores 0.9701 / 0.6984 — close
but not identical, expected given the different split ratios and therefore a different
random test set, not a discrepancy.

Top 5 features by Gini importance (2026-08-22 re-run, 55-feature leak-fixed table):
`forest_frac_baseline` (0.2058), `ndvi_mean` (0.0950), `ndvi_trend_2x12ma` (0.0934),
`ndvi_below_threshold` (0.0693), `ndvi_clim_june` (0.0585). With the leaky
`forest_frac_recent`/`forest_frac_current` features gone, `forest_frac_baseline` alone
now carries roughly the same combined importance the three forest-fraction features held
together before (~0.20 now vs. ~0.40 combined pre-fix) — the model still leans heavily on
forest fraction, just through the one feature that doesn't risk encoding fire's own
aftermath. Full ranking in `Model_Outputs/Feature_Importance.png`.

`hp_search_rf.py`'s own winner-model run (same hyperparameters, same leak-fixed table,
different 65% train split) — a close proxy for ranks beyond the top 5, not this exact
notebook run — puts `terrain_slope` at 7th (0.0456) and `terrain_elevation` at 9th
(0.0255), ahead of most FLDAS/land-cover features; `access_dist_roads` enters around
15th (0.0127). Terrain still doesn't crack the top 5, but slope remains the strongest of
the 6 terrain/accessibility features, consistent with Step 5a's own fire-coincidence
finding (fires sit at +115% mean slope vs. the national average). The remaining
terrain/accessibility features (`access_dist_railways`, `terrain_aspect`,
`access_dist_waterways`) rank lower still; exact ranks below ~15th were not independently
re-verified against this specific notebook run's 55-feature model (only the top 5 above
are quoted directly from it) — see `Model_Outputs/Feature_Importance.png` for the full,
authoritative ranking of this exact run.

### MaxEnt baseline (Biswas et al. 2025 comparison, added 2026-08-17)

This pipeline explicitly extends **Biswas, U., Mahato, S., & Joshi, P.K. (2025)**, the
paper cited at the bottom of this README, which uses **MaxEnt (Maximum Entropy)** for
forest-fire susceptibility mapping in India. Step 7 trains a real MaxEnt model
(`elapid.MaxentModel` v1.0.4 — a scikit-learn-compatible reimplementation of Phillips et
al.'s algorithm, `feature_types=['linear','hinge','product']`, cloglog transform) on this
project's own 55-feature, leak-fixed table (retrained 2026-08-22 alongside the Random
Forest, on the same corrected parquet) and evaluates it on the **identical held-out test
set** (832,202 pixels) as the Random Forest above — a direct, apples-to-apples comparison,
not a citation of Biswas et al.'s own reported numbers (which would be a weaker comparison
given their different feature set/resolution). MaxEnt's own hyperparameters are
deliberately untuned in this pass (a separate, still-open item) — only the Random Forest
was tuned, and both models were retrained on the same corrected feature table.

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

| Metric | Random Forest (headline, tuned) | MaxEnt (elapid, tuned) |
|---|---|---|
| ROC-AUC (held-out test, 832,202 px) | **0.9704** | 0.9598 |
| Average Precision | **0.7011** | 0.6275 |
| Spatial-block CV AUC (2°×2° `GroupKFold`, n=3) | **0.9498 ± 0.0035** | 0.9465 ± 0.0054 |
| Training rows | 3,328,807 (100% of train split) | 150,000 (4.51% of train split, stratified) |
| Training time | 216.0 sec | 1,232.2 sec (20.5 min) |
| Test-set inference time | 1.9 sec | 34.0 sec |

(Prior 58-feature/untuned-RF-era numbers, 2026-08-20: Random Forest ROC-AUC 0.9683 / AP
0.6796; MaxEnt ROC-AUC 0.9595 / AP 0.6237 — superseded by the 2026-08-22 leak-fix +
RF-tuning retrain, then the 2026-08-22 specific-humidity feature addition (57 features)
and 2026-08-23 validated MaxEnt `beta_multiplier` tuning above.)

**2026-08-23: MaxEnt hyperparameters validated-tuned** (`hp_search_maxent.py`, closing
the last "untuned" item in the project's rigor audit) — searched `beta_multiplier` over
{0.5, 1.0, 1.5, 2.5, 4.0} by validation AUC (65/15/20 split). The grid was essentially
flat (validation AUC 0.9589–0.9592 across the whole range), a genuine near-null tuning
result, not a large correction; winner `beta_multiplier=4.0` is used above.

Random Forest outperforms MaxEnt on this feature set by 0.0106 ROC-AUC (0.9704 vs. 0.9598)
and 0.0736 Average Precision on the random split, and by 0.0033 AUC (0.9498 vs. 0.9465) on
the spatial-block split — a modest but consistent RF advantage on both evaluation
protocols, plausible given RF trains on the full 3.3M-row training set with a more
flexible (non-linear, non-additive) decision boundary, while MaxEnt here is a
linear/hinge/product-feature exponential-family model trained on a 4.5%-of-training-set
stratified subsample. Both models show the
same qualitative pattern of AUC dropping under spatial-block evaluation versus the random
split (RF: 0.9704 → 0.9498, a 2.1% drop; MaxEnt: 0.9598 → 0.9465, a 1.4% drop) — expected,
since neighboring pixels are spatially autocorrelated and a random split alone overstates
generalization to genuinely new geography; both remain far above CDR-PINN's own Track B1
spatial-block AUC of 0.7510 on the same protocol. This is reported as a straightforward
measured result, not tuned in either model's favor. Outputs:
`ROC_PR_Curves_RF_vs_MaxEnt.png`, `RF_vs_MaxEnt_Comparison.csv`,
`MaxEnt_Feature_Importance.png` (permutation importance), `MaxEnt_Susceptibility_Probability.tif`,
`Fire_Susceptibility_Map_RF_vs_MaxEnt.png`.

### Comparison against Biswas et al. (2025) — validation-regime rigor, not just accuracy

Biswas et al. (2025) train a single MaxEnt model at their 0.25° (~27km, ~730 km²/pixel)
common working grid, evaluated with one random train/test split and no spatial
cross-validation. Per Roberts et al. (2017, *Ecography*, 40(8):913–929,
"Cross-validation strategies for data with temporal, spatial, hierarchical, or
phylogenetic structure," DOI: 10.1111/ecog.02881), that kind of random-split-only
protocol tends to overstate real-world generalization whenever the underlying data
is spatially autocorrelated — true of essentially every variable both this project
and Biswas et al. use (climate, vegetation, terrain). This project trains **both** a
Random Forest and a direct MaxEnt replication of their method, at ~1km native
resolution rather than 0.25°, and evaluates each with three independent protocols:
an 80/20 random split, 5-fold `StratifiedKFold` CV, and 2°×2° spatial-block
`GroupKFold` CV (matching CDR-PINN's own Track B1 exactly, so the same
spatial-generalization check applies consistently across every model in this
study) — a substantially more rigorous validation regime than the reference paper's
own.

The honest, disclosed numbers: this project's own MaxEnt replication (0.9598 AUC
random-split, 0.9465 spatial-block) already exceeds the informal ≥0.7 "useful
model" threshold commonly cited in the species-distribution-modeling literature
MaxEnt comes from. Random Forest is a modest, consistent improvement over that same
MaxEnt on identical data and identical splits (+0.0106 AUC / +0.0736 AP on the
random split, +0.0033 AUC on the spatial-block split) — a genuine RF-vs-MaxEnt
model-class comparison Biswas et al. themselves never report, since they only ever
trained MaxEnt. No numeric AUC is cited *from* Biswas et al.'s own paper anywhere in
this comparison (a deliberate choice, not an oversight — their MaxEnt was fit on a
different 15-variable table at a different resolution, making a cross-paper AUC
citation a weaker comparison than training MaxEnt fresh, here, on this project's own
data). Both classical models also comfortably outperform this project's own
CDR-PINN on the spatial-block protocol (0.7510) — disclosed as an honest,
unfavorable-to-the-PINN result, not smoothed over (see the
`integrated-fire-risk-model` skill and `Physics_Informed_FireRisk_Model/README.md`
for the full three-way comparison).

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
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=10800 "Step7_FireRisk_Susceptibility_Model.ipynb"
```

Step 6 requires Steps 1, 2, 3, 4, and 5 (5a Terrain + 5b Accessibility) to have already run
(reads their outputs directly, no in-notebook recomputation). Step 7 requires Step 6's
parquet. Step 7's total wall time is now **~137 min** (measured 2026-08-22: 8,241.2 sec, on
the 55-feature leak-fixed table with tuned RF hyperparameters) — up substantially from the
~56 min 2026-08-20 run, almost entirely because of the spatial-block CV cell added since
then: it refits MaxEnt from scratch on a 150k-row subsample in **each of 3 folds**
(1,863.3 + 1,132.6 + 1,674.5 sec ≈ 78 min alone) on top of 3 Random Forest refits (~5.8 min
total). The single-cell `--ExecutePreprocessor.timeout` must be **at least 10800 sec
(3 hr)** to comfortably cover that spatial-block CV cell — the previously-documented 3600
sec would now truncate it mid-fold.

## Outputs

```
Integrated_Outputs/
├── Integrated_FireRisk_Stack.tif              # 59-band GeoTIFF (not tracked, ~large)
├── Integrated_FireRisk_Pixels.parquet          # ML-ready table (not tracked, ~large)
├── Integrated_FireRisk_Pixels_sample200k.csv   # 200k-row sample (gitignored — 104MB, exceeds GitHub's 100MB limit; kept locally only, not tracked)
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
├── Model_Comparison_SpatialBlockCV.csv         # tracked -- per-fold RF/MaxEnt AUC/AP, 2deg x 2deg GroupKFold
├── rf_hp_search_result.json                    # tracked -- hp_search_rf.py's validation-set search results
├── maxent_hp_search_result.json                # tracked -- hp_search_maxent.py's validation-set search results
├── Computational_Cost_Dashboard.png             # tracked
└── Computational_Cost_Reproducibility_Report.json  # tracked
```

## Citation

- Biswas, U., Mahato, S., & Joshi, P.K. (2025). Spatial prediction of forest fires
  in India: a machine learning approach for improved risk assessment and early
  warning systems. *Environmental Science and Pollution Research*, 32(8), 4856–4878.
  DOI: 10.1007/s11356-025-35982-8. (Verified via Crossref, see `METHODOLOGY.md`.)
- Roberts, D.R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for
  data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*,
  40(8), 913–929. DOI: 10.1111/ecog.02881. (Motivates this step's spatial-block CV,
  see the "Comparison against Biswas et al." subsection above.)

## License

No license has been chosen yet for this repository's code.
