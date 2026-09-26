# SRF Presentation Reference — Feature Counts, Definitions & Impact

Reference material compiled for the SRF presentation, drawn from this repo and its sibling
step repos (Step 1–5b). Two versions of the feature set exist: **v1 (historical)** — the
numbers originally reported by each step's notebook — and **v2 (current/audited)** — the
corrected feature set after the 2026-09-25 end-to-end audit. Cite v2 numbers unless
explicitly discussing the pipeline's history.

## 1. Feature Count by Category

| Variables / Source | v1 (historical) | **v2 (current)** | What changed in v2 |
|---|---|---|---|
| NDVI (Step 2) | 9 | **6** | Dropped anomaly, 2×12-MA trend, residual, monthly Mann-Kendall τ, and the θ* breakpoint indicator — all flagged degenerate/statistically invalid in the audit. Kept QA mean, climatological June; added Seasonal Kendall τ + Sen's slope; kept CVSI (k*=8) and LISA. |
| LST (Step 3) | 5 | **6** | Anomaly-only features replaced by climatological levels (Day/Night/DTR = 3) + Seasonal Kendall τ now computed for all three, including DTR (3) = 6. v1 was 2×2 (Day/Night × anomaly+MK) + 1 DTR-anomaly-only = 5. |
| 7 Climatic variables — FLDAS (Step 4) | 14 | **14** | Unchanged in count. Anomaly swapped for climatological level; Seasonal Kendall τ replaces monthly Mann-Kendall. Same 7×2 structure (air temp, wind, precipitation, relative humidity, specific humidity, soil moisture, net LW radiation). |
| LULC, 2001 baseline map (Step 4) | 22 | **21** | One fewer land-cover fraction column in the v2 2001-map rebuild (class-set/encoding adjustment). |
| Elevation, Slope, Aspect (Step 5a) | 3 | **4** | Aspect re-encoded as sin/cos (2 columns) instead of one raw degree value, to correctly handle its circular (0°=360°) nature. |
| Distance to Roads/Railways/Waterways (Step 5b) | 3 | **3** | Unchanged. |
| Forest fraction, baseline (Step 6) | 1 | **1** | Unchanged — `forest_frac_baseline` (2001 LULC epoch only; `forest_frac_recent`/`forest_frac_current` were dropped 2026-08-21 as a data-leakage fix — see §3). |
| **Total predictor features** | **57** | **55** | — |

Not a model feature (kept separate): **Fire Points** is the target label (`fire_ever`,
binary; `fire_count` also exported), not a predictor. **Burned Area** (MCD64A1.061) is
validation-only — used to cross-check Step 1's fire-point extraction against independent
satellite burned-area records; it is never joined into `Integrated_FireRisk_Pixels.parquet`.

## 2. Why Each Feature Group Is Used, and Its Impact on the Study

**NDVI (6 v2 features)** — proxy for vegetation moisture/fuel condition; healthy canopy
(high NDVI) burns less readily, while browning/drying vegetation is a known fire-risk
precursor (Chuvieco et al., 2004). Biswas et al. (2025)'s single most important predictor
(22.3% importance / 28.4% contribution) — here decomposed into climatology, trend
(Seasonal Kendall + Sen's slope), CVSI antecedent-stress index, and LISA spatial clustering,
none of which appear in Biswas et al.'s treatment. CVSI's 8-month lag was chosen by mutual
information against real fire occurrence, not assumed from literature — a genuine novel
contribution.

**LST — Day/Night/DTR (6 v2 features)** — LST is a direct remote-sensing proxy for surface
heating and fuel dryness. Day and night are tracked separately because they carry different
physical information (solar heating vs. retained overnight heat/moisture); DTR (Diurnal
Temperature Range) is a distinct dryness/fuel-curing signal Biswas et al. do not compute at
all. Biswas et al.'s second-largest predictor group after NDVI (~19.7% combined importance:
LST night 10.1%/8.9%, LST day 9.6%/4.5%). Honest counterpoint: in the trained Random
Forest/MaxEnt models, these 6 features rank in the bottom half of all 55 by importance.

**7 Climatic variables — FLDAS (14 v2 features)** — air temperature, wind, precipitation,
relative + specific humidity, soil moisture, and net longwave radiation capture the
atmospheric/moisture conditions that govern fuel dryness independent of vegetation state
itself. Each variable gets a climatological level + Seasonal Kendall trend term (14 = 7×2),
paralleling the NDVI/LST treatment so all climate-adjacent variables share one consistent
statistical framework.

**LULC, 21 fractions (Step 4)** — Biswas et al. use ESA CCI/C3S land cover only once, as a
binary forest/non-forest mask to filter fire points. This project instead reclassifies the
same source into the full Level-1 LCCS legend and computes per-pixel fractional cover of
every class — a vegetation-composition feature set with no equivalent in their paper.

**Elevation, Slope, Aspect (4 v2 features)** — terrain shapes fire behavior through three
established mechanisms: slope accelerates upslope fire spread via fuel preheating
(Rothermel, 1972); aspect controls solar exposure/fuel dryness; elevation proxies
temperature/vegetation-zone gradients. Real fire points sit at 12.35° mean slope vs. 5.72°
nationally (+116%), directly corroborating Biswas et al.'s finding that slope is their
second-highest contribution variable (16.7%) after NDVI.

**Distance to Roads/Railways/Waterways (3 v2 features)** — standard proxies for
human-caused ignition likelihood (most Indian forest fires are human-started: agricultural
burning, discarded cigarettes, campfires, deliberate clearing). Waterways add a second
mechanism: riparian corridors carry denser fuel and more human activity. Fires cluster much
closer to roads (−38.9%) and waterways (−64.4%) than the national baseline; railways show
almost no effect, matching Biswas et al.'s own lowest-contribution ranking for that variable.

**Forest fraction, baseline only (1 v2 feature)** — the model's single highest-importance
feature (Gini importance ≈0.21, historically ~0.40 combined across three forest-fraction
columns before a 2026-08-21 leakage fix). Only the 2001 baseline is kept: 2020/2022 LULC
snapshots fall inside the same window as the pooled fire label, and published literature
documents burned-forest pixels being reclassified to shrubland/agriculture in later LULC
epochs — i.e. those features could partly encode the *outcome* of fire rather than a
pre-fire condition.

## 3. Key Corrections from the 2026-09-25 Audit (cite v2, not v1)

- Anomaly-mean features (NDVI, LST, climate) are **degenerate**: with a 2001–2020 baseline
  they mathematically equal the residue of the 26 out-of-baseline months. v2 replaces them
  with climatological levels.
- The original Mann-Kendall trend tests were run on an **MA-smoothed series**, which induces
  strong autocorrelation (lag-1 ≈ 0.975) and invalidates the test's independence assumption.
  v2 uses Seasonal Kendall (Hirsch et al., 1982) + Benjamini-Hochberg FDR correction on the
  raw series instead.
- Data-leakage fix (2026-08-21): `forest_frac_recent`/`forest_frac_current` (2020/2022) were
  dropped — they fall inside the same window as the pooled `fire_ever` label and risked
  encoding fire's aftermath rather than a pre-fire condition.
- Fire-point rasterization was corrected from round to floor pixel assignment (the round
  rule had displaced 74.9% of points by one pixel).
- India-boundary masking was added to Step 2 (NDVI), which previously had none — 67.20% of
  the raw grid was outside India and was silently included in every downstream NDVI feature
  before the fix.

## Sources

- [Terrain-Elevation-Slope-Aspect-India-SRTMGL3-90m-forest_fire_India-2000-2022-](https://github.com/rawan230/Terrain-Elevation-Slope-Aspect-India-SRTMGL3-90m-forest_fire_India-2000-2022-)
- [India_Distance_Analysis-Roads-Railways-Waterways-Forest-Fire-in-India-2000-2022-](https://github.com/rawan230/India_Distance_Analysis-Roads-Railways-Waterways-Forest-Fire-in-India-2000-2022-)
- [NDVI-WITH-FOREST-FIRE-POINTS-ANALYSIS-IN-INDIA](https://github.com/rawan230/NDVI-WITH-FOREST-FIRE-POINTS-ANALYSIS-IN-INDIA)
- [LST-Analysis-India-Nov-2000-to-Dec-2022-](https://github.com/rawan230/LST-Analysis-India-Nov-2000-to-Dec-2022-)
- [Land-Surface-Model-Variables-Analysis-FLDAS-](https://github.com/rawan230/Land-Surface-Model-Variables-Analysis-FLDAS-)
- [Forest-Fire-Points-Extraction-in-India-2000-2022-](https://github.com/rawan230/Forest-Fire-Points-Extraction-in-India-2000-2022-)
- [Integrated-Fire-Risk-Analysis-in-India-and-Impliment-Baseline-Random-Forest-Model-](https://github.com/rawan230/Integrated-Fire-Risk-Analysis-in-India-and-Impliment-Baseline-Random-Forest-Model-) (this repo)

Citation: Biswas, U., Mahato, S., & Joshi, P.K. (2025). Spatial prediction of forest fires
in India: a machine learning approach for improved risk assessment and early warning
systems. *Environmental Science and Pollution Research*, 32(8), 4856–4878.
DOI: 10.1007/s11356-025-35982-8.
