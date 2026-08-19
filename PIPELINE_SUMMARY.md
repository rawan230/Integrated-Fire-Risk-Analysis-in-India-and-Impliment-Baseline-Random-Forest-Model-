# Pipeline Technical Report — India Forest-Fire-Risk Preprocessing

Detailed mathematical reference for all 6 preprocessing steps: every feature's
formula (with every symbol explicitly defined), why it's engineered this way, **why
this specific method was chosen over other existing methods in the literature**,
its novelty classification, and its reference. For the audit trail (discrepancies
found/fixed, citation-confidence flags) see `METHODOLOGY.md`.

**Renumbered 2026-08-17**: FLDAS/land-cover is now Step 4 (was Step 6), integrated
alignment is now Step 5 (was Step 4), the susceptibility model is now Step 6 (was
Step 5) — training is genuinely the last step in the real execution order now, not
an oddly-numbered middle step. Only labels changed, not code or data flow (FLDAS
always had to run before assembly, regardless of its old number).

**Novelty tags**: **[STANDARD]** = established method applied as-is; **[ADAPTED]** =
established method modified for this project's data/domain; **[NOVEL]** =
project-original, no literature precedent.

## Resolution — Every Source, Compared Against Biswas et al. (2025)

**Corrected 2026-08-18**: the previous "~5km" figure below was explicitly flagged as
unverified guesswork from prior design notes. It has now been replaced with the real
number, extracted directly from the user's own copy of the paper: **"these datasets
were in raster format with a spatial resolution of 0.25° × 0.25°"** — the common grid
Biswas et al. resampled their forest-fire-occurrence conditioning factors onto for the
actual MaxEnt run. (Individual source datasets still have their own differing *native*
resolutions per their Table 2 — e.g. NDVI/LST at 0.05°, FLDAS/GPM at 0.1°, GLDAS at
0.25°, LULC at 300m — but 0.25° is the resolution their model actually consumes.)
0.25° ≈ 27.8 km north–south and ≈ 26 km east–west at India's mid-latitude (~20°N,
where longitude spacing shrinks by cos(20°) ≈ 0.94) — call it **~27 km per axis, ~730
km² per pixel**, not ~5km/~25km². This makes the resolution gap markedly larger than
previously stated.

| Source | Native spatial res. | Native temporal res. | Grid used in this project | vs. Biswas et al. (0.25° ≈ 27km) |
|---|---|---|---|---|
| MODIS NDVI (MOD13A3.061) | 1km | Monthly | 1km — **defines** the shared grid | **~27× finer per axis, ~730× finer per pixel area** |
| MODIS LST (MOD11A2.061) | 1km | 8-day composite → aggregated monthly | 1km (reprojected to NDVI grid) | ~27× finer per axis, ~730× finer per pixel area |
| ESA-CCI/C3S LULC (forest fraction + 22-class) | 300m | Annual | 1km (area-weighted fractional aggregation, §Step 4/5) | Native 300m is ~90× finer per axis than 0.25°; the aggregated 1km *output* is still ~27× finer, while the aggregation itself (unlike FLDAS below) genuinely averages real sub-pixel detail, not fabricated detail |
| FLDAS climatic (FLDAS_NOAH01_C_GL_M) | **~11km (0.1°)** | Monthly | 1km (bilinearly interpolated onto the shared grid) | Native ~11km is still **~2.5× finer** than Biswas et al.'s 0.25°/~27km common grid |
| MODIS active fire (FIRMS, Step 1 label) | 1km nominal | Daily detections, rasterized to the study's monthly cadence | 1km (exact affine lookup, no resampling) | ~27× finer per axis |

**Honest note on FLDAS**: unlike NDVI/LST/LULC — which are genuinely native at
(or finer than) 1km and are simply *reprojected/aggregated* onto the shared grid
without inventing information — FLDAS's raw data physically only carries ~11km-scale
information. The bilinear interpolation used in Step 4's grid-based pipeline
(and referenced in `Integrated_Analysis`'s stacked feature table) makes FLDAS
*appear* to sit on the same 1km grid as everything else, but does not create real
1km-resolution climatic detail — it's a smooth interpolation of a coarser field, not
new measurement. **This is precisely the reasoning behind the CDR-PINN's gridless
architecture decision** (`CDR_PINN_Diffusion_Design_v2.md` §3): querying FLDAS at
its own true ~11km resolution via a differentiable interpolator, rather than
pre-resampling it onto a shared 1km grid, avoids overstating this variable's actual
resolution. When this comparison goes into the paper, state NDVI/LST/LULC's ~730×
pixel-area advantage and FLDAS's genuine ~11km resolution (still ~6× finer per area
than Biswas et al.'s 0.25° grid) as two separate, honest claims — don't let FLDAS ride
on the same "~730× finer" framing the other sources earn legitimately.

### What resolution actually means, and why it matters this much

**Concretely**: "resolution" is the ground area one pixel represents. A 1km pixel
covers 1 km²; a 0.25° (~27km) pixel covers roughly 27×27 ≈ **730 km²** — this is why
linear and areal comparisons diverge (~27× finer *per axis* is ~730× finer *per
pixel*, i.e. ~730× more independent spatial observations over the same land area).
This distinction matters because "27× finer" sounds like a modest sharpening but is
actually a ~730-fold increase in the number of distinct data points describing the
same country.

**Made concrete with this project's own numbers**: Step 5 established national mean
forest fraction at ~10.2–10.7% of India's ~3.287 million km² land area — roughly
**335,000–350,000 km² of forest**. At 1km resolution that's on the order of
**335,000–350,000 individual forest pixels**; at Biswas et al.'s actual 0.25°
(~730 km²/pixel) working resolution it would be roughly **460–480 pixels** — a ~730×
difference in how many independent forest observations the model ever sees, not an
abstract ratio.

**Why this specifically matters for *fire* risk, not just "more data is better" in
general**:

1. **Fire processes are physically local.** Fuel continuity, canopy gaps, and
   forest-edge effects — the things that actually determine whether a fire spreads
   or self-extinguishes — operate at scales of tens of meters to a few kilometers,
   not tens of kilometers. A 5km pixel averaging across a mix of dense forest,
   cropland, and cleared land produces a value that describes *none* of those land
   covers accurately — a small, genuinely high-risk forest fragment gets diluted
   into a "medium" pixel alongside the cropland surrounding it. This is precisely
   the physical justification behind the CDR-PINN's spatially-varying diffusion
   coefficient `D(x,y)` (`CDR_PINN_Diffusion_Design_v2.md` §7) — modeling
   sub-regional heterogeneity in fuel continuity is meaningless if the underlying
   data resolution is already coarser than the heterogeneity itself.

2. **Label quality degrades at coarse resolution, not just feature quality.** This
   project's 541,545 fire points are individually geolocated. At 1km, a fire point
   labels roughly the 1 km² it actually occurred in. At 5km, many spatially-distinct
   fires collapse into the same pixel, and that pixel's "fire" label now describes a
   25 km² area where the fire may have burned a small fraction of it — the
   NDVI-fire relationship F9 fits (a fine-grained breakpoint in fire probability
   against NDVI) would be substantially washed out, because the fire label itself
   becomes spatially imprecise before the vegetation feature is even considered.

3. **Statistical power**: more independent spatial samples over the same land area
   means more effective training data for the same underlying phenomenon, and a
   sharper achievable decision boundary — part of why this project's models reach
   ROC-AUC ~0.967 is that the fire label and the vegetation/climate features are
   both resolved finely enough to actually correlate at the pixel level, rather
   than being pre-smoothed into a weaker relationship before modeling even starts.

4. **Operational relevance**: a fire-risk map is meant to inform real decisions —
   where to position watch resources, where to prioritize fuel management. Indian
   forest management units (ranges, beats) are frequently smaller than a single
   25 km² (5km) pixel, meaning a coarser map cannot resolve risk at the scale
   decisions actually get made at; a 1 km² pixel is much closer to that operational
   scale.

**Temporal resolution, briefly**: monthly aggregation (used throughout this
pipeline for climatology/anomaly/trend) smooths out short, sharp events — a 3-day
heatwave that critically dries fuel could be invisible in a monthly mean. This is a
real, acknowledged tradeoff, not resolved in this pipeline (the fire label itself is
available at daily granularity from FIRMS, but the climatic/vegetation drivers are
only meaningfully computable at monthly cadence given the source products'
own native temporal resolution — MOD13A3 is monthly-composited, FLDAS is
monthly-native). Worth naming explicitly as a limitation if the paper's discussion
section addresses what finer temporal resolution could add.

---

## Why feature engineering matters here

A raw satellite pixel (an NDVI value, a temperature reading) tells you almost
nothing about fire risk by itself — a moderate NDVI reading could mean healthy
grassland or drought-stressed forest. Fire risk is driven by **deviations from
normal**, **directional change over decades**, **accumulated stress**, and
**spatial context** — none of that is present in a raw pixel value. Every feature
below exists to extract one specific piece of that signal, and for every one there
were other ways it could have been engineered — the comparisons below explain why
the ones used here were chosen.

---

## Step 1 — Fire Point Extraction (ground truth)

**Method**: point-in-polygon test against the dissolved India state boundary Ω,
```
inside(p) = shapely.contains_xy(Ω, lon_p, lat_p)
```
**Definitions**: `p` = a candidate fire detection; `Ω` = the dissolved India state
boundary polygon (not the country boundary, which has ~60 degenerate sliver
polygons near the Palk Strait); `lon_p, lat_p` = the point's coordinates.

Forest classification via exact affine grid indexing:
```
row = round((lat_p − lat₀) / Δlat),   col = round((lon_p − lon₀) / Δlon)
```
**Definitions**: `(lat₀, lon₀)` = the LULC grid's own origin coordinate (its
first `[0,0]` cell's center); `(Δlat, Δlon)` = the grid's pixel step size in
degrees, read directly from the source NetCDF's own coordinate arrays; `row, col`
= the resulting integer pixel indices, clamped to `[0, grid_height−1]` /
`[0, grid_width−1]`.

**Why exact polygon clip vs. a bounding box**: a bbox clip is a single comparison
per point (`lon_min≤lon≤lon_max` etc.) — much cheaper, but for India's actual shape
it would include ~1.2M extra points from Sri Lanka, Nepal, Bangladesh, Myanmar, and
Pakistan (all inside the natural bounding rectangle of India's mainland). The exact
polygon test costs more per point but is the only way to get a correct label set —
not a tradeoff worth making at this ground-truth stage.

**Why affine-index lookup vs. a spatial join** (e.g. GeoPandas `sjoin`): a spatial
join tests each point against a spatial index of polygons/pixels — general-purpose,
but slow at scale (500k+ points × per-year rasters) and introduces a tolerance
parameter (how close counts as "inside"). Affine indexing is `O(1)` per point,
exact (no tolerance), and deterministic — appropriate specifically because both the
points and the raster share a known, regular coordinate system, which a generic
spatial join doesn't assume or exploit.

**Result**: 2,804,373 raw detections → 1,599,466 in-India (dedup'd) → **541,545
forest fires**. Forest = 13 ESA-CCI/C3S LCCS codes. **[ADAPTED]** — class scheme
follows Sannigrahi et al. (2018) as cited in Biswas, Mahato & Joshi (2025),
*Environ. Sci. Pollut. Res.*, 32(8):4856–4878, DOI:10.1007/s11356-025-35982-8
[cite-confirmed]; the exact-polygon clip and affine lookup are this project's own
implementation.

---

## Step 2 — NDVI Features (9 features)

**Grid**: 3641×3504 px, EPSG:4326, ~0.01°/1km. **Time axis**: `T=266` months —
derived as the count of calendar months from Nov 2000 to Dec 2022 inclusive
(Nov 2000 is month 1, Dec 2022 is month 266; `(2022−2000)×12 + (12−11) + 1 = 266`).
Source: MOD13A3.061 monthly 1km NDVI.

### F1 — QA-filtered NDVI mean
```
F1(i,j) = (1/n) Σ_t NDVI(t,i,j),   t over months where QA(t,i,j) ∈ {Good, Marginal}
```
**Definitions**: `i,j` = pixel row/column indices; `t` = month index, `t∈{1,...,T}`;
`NDVI(t,i,j)` = the raw NDVI value at that pixel-month; `QA(t,i,j)` = the MOD13A3
`pixel_reliability` quality flag for that pixel-month (categories: Good, Marginal,
Snow/Ice, Cloudy, Fill); `n` = count of pixel-months passing the QA filter for
pixel `(i,j)` (varies per pixel due to persistent cloud cover in some regions).

**Why QA-filtered mean vs. using all pixels unconditionally**: cloud- and
snow-contaminated NDVI readings are not random noise — they're systematically
biased (clouds read as artificially low NDVI, snow as near-zero), so including them
would not average out, it would drag the mean down in exactly the pixels where
cloud cover is heaviest (often high-elevation/monsoon-heavy regions), confounding
genuine low vegetation with data-quality artifacts. **Why simple mean vs. gap-filling/
smoothing** (e.g. harmonic regression, Savitzky-Golay filtering, common in NDVI
time-series literature for producing a smooth, gap-free curve): those methods
interpolate a continuous curve suited to phenological analysis (e.g. detecting
green-up dates); this project instead separates that job explicitly across F2–F5
(climatology, anomaly, trend, residual), so F1 only needs to answer "what's the
typical value here," where a direct filtered mean is simpler, more transparent, and
avoids a smoothing-bandwidth hyperparameter with no principled way to set it at
national/multi-decade scale. **[STANDARD]** NDVI: Rouse, Haas, Schell & Deering
(1974), NASA SP-351, 3010–3017 [cite-confirmed].

### F2 — Climatology
```
μ̄(i,j,m) = mean{ NDVI(y,m,i,j) : y ∈ [2001,2020] },   m = 1..12
```
**Definitions**: `y` = calendar year; `m` = calendar month (1=January, ..., 12=
December); `μ̄(i,j,m)` = the baseline/"normal" NDVI for pixel `(i,j)` in calendar
month `m`, averaged over the 20-year baseline window 2001–2020 (2000 excluded as a
partial year; 2021–2022 excluded to leave an independent out-of-baseline period).

**Why per-calendar-month empirical mean vs. a single grand mean**: a single
whole-series mean would treat a wet-season June and a dry-season January as
directly comparable, erasing the seasonal cycle that's the actual reference point
for detecting anomalies. **Why empirical mean vs. a smooth harmonic/Fourier-fit
climatology** (fitting e.g. a 1st/2nd-harmonic sinusoid to the annual cycle, common
in longer-record climate-normal literature): a harmonic fit assumes the seasonal
cycle is smoothly sinusoidal, which understates sharp monsoon-onset transitions
typical of Indian vegetation phenology; the empirical per-month mean makes no shape
assumption at all. **[STANDARD]** — climatological-normal methodology, standard in
remote-sensing anomaly detection generally.

### F3 — Anomaly
```
δ(t,i,j) = NDVI(t,i,j) − μ̄(i,j, month(t))
```
**Definitions**: `δ` = the anomaly value (can be negative or positive);
`month(t)` = the calendar month corresponding to time index `t`.

**Why raw departure vs. a standardized (z-scored) anomaly** (dividing by the
pixel's own standard deviation, the convention behind indices like SPI for
precipitation): precipitation varies over orders of magnitude between regions, so
standardizing is necessary before comparison; NDVI is already a bounded,
directly-comparable ratio (`[-1,1]`, empirically `[-0.2,1.0]` here) — dividing by σ
would instead *suppress* the raw magnitude signal in low-variance pixels (e.g.
persistently dense evergreen forest, where even a small absolute NDVI drop can be
ecologically meaningful) while inflating it in naturally noisy pixels. **[STANDARD]**.

### F4/F5 — Trend & Residual (classical 2×12-MA decomposition)
```
Trend(t) = Σ_{k=−6}^{6} w_k · x(t+k) / Σ_{k=−6}^{6} w_k
w = [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]          (13 taps, k=−6..6)
Detrended(t) = NDVI(t) − Trend(t)
Seasonal(m) = mean{ Detrended(t) : month(t) = m }         (over the full 266 months)
Residual(t) = Detrended(t) − Seasonal(month(t))
NDVI(t) = Trend(t) + Seasonal(month(t)) + Residual(t)     (classical additive model)
```
**Definitions**: `k` = the moving-average lag offset (−6 to +6 months around `t`);
`w_k` = the tap weight for offset `k` (half-weight, 0.5, at the two endpoints
`k=±6`, full weight 1 elsewhere — the standard centered-MA taper that avoids
double-counting when the window length equals the seasonal period, 12); F4 = the
time-mean of `Trend(t)` over all valid `t`; F5 = the time-mean of `Residual(t)`.
Undefined for the first/last 6 months (254/266 valid).

**Why classical 2×12-MA vs. STL** (Seasonal-Trend decomposition using Loess,
Cleveland et al. 1990 — the standard modern alternative): STL fits a locally-weighted
smoother with a tunable bandwidth, adapting to slowly-changing seasonal shape and
producing a smoother trend — but at the cost of a bandwidth hyperparameter with no
principled default at national/4M-pixel/266-month GPU scale, and a less transparent,
harder-to-audit trend estimate. The classical fixed-weight MA has no tunable
parameter, is exactly reproducible, and is fast enough to run on the full grid on
GPU — the simplicity is a deliberate tradeoff against STL's smoother but
harder-to-justify-at-scale output. **Why not simple linear detrending** (fit one
straight line per pixel): would force a strictly monotonic trend shape, missing
multi-year reversals (e.g. drought followed by recovery) that a moving-average
trend can follow. **[STANDARD]** decomposition family reference: Cleveland,
Cleveland, McRae & Terpenning (1990), *J. Official Statistics*, 6(1):3–73
[cite-verify — this project uses the simpler classical MA, not full STL/loess].

### F6 — Mann-Kendall trend τ (on the Trend component)
```
S = Σ_{lag=1}^{T−1} Σ_i sign(x_{i+lag} − x_i)
τ = S / [n(n−1)/2]
Var(S) = n(n−1)(2n+5)/18
z = (S∓1)/√Var(S),   p = 2·(1 − Φ(|z|))
```
**Definitions**: `x` = the Trend(t) series being tested (not raw NDVI); `sign(·)` =
`+1` if positive, `−1` if negative, `0` if zero; `S` = the Mann-Kendall statistic
(net count of increasing vs. decreasing pairs across every lag, not just lag-1);
`n` = count of valid (non-NaN) observations for that pixel (requires `n≥10`); `τ` =
Kendall's tau, in `[−1,1]`, the trend-direction/strength estimate; `Var(S)` = the
variance of `S` under the null hypothesis of no trend; `z` = the standardized test
statistic (with the `∓1` continuity correction); `Φ(·)` = the standard normal CDF;
`p` = the two-sided p-value.

**Why Mann-Kendall vs. ordinary least-squares (OLS) linear-regression slope**: OLS
assumes normally-distributed, homoscedastic residuals and a strictly linear trend
shape — assumptions routinely violated by ecological time series with outliers
(extreme drought years) and non-linear trajectories. Mann-Kendall is
**non-parametric**: it only asks whether later values tend to exceed earlier ones,
making no distributional assumption and remaining robust to outliers and moderate
non-linearity — the standard choice in environmental/climate trend literature for
exactly this reason. **Why normal-approximation p-value vs. an exact permutation
test**: the exact distribution of `S` is only tractable to compute directly for
small `n`; at `n` up to 266 the normal approximation is standard practice and
computationally required at this pixel count (4M+ pixels). **[STANDARD]**: Mann
(1945), *Econometrica*, 13(3):245–259; Kendall (1975), *Rank Correlation Methods*,
Griffin [both cite-confirmed]. (Not currently paired with Sen's slope estimator —
a natural, standard companion statistic that would give a trend *magnitude* in
NDVI-units/decade alongside τ's direction/strength; a defensible future extension,
not computed here.)

### F7 — CVSI: Cumulative (Pre-Fire) Vegetation Stress Index — **[NOVEL]**
```
CVSI(t,k) = Σ_{lag=1}^{k} max(−δ_{t−lag}, 0)
k* = argmax_k I(Y; CVSI_k)
```
**Definitions**: `k` = the trailing window length in months; `max(−δ,0)` = the
stress contribution of one month (zero if that month was wetter/greener than
normal, `−δ` i.e. positive if drier); `I(Y;CVSI_k)` = mutual information between
the real fire/no-fire label `Y` and quantile-binned (10 bins) `CVSI_k` values,
computed on a balanced case-control sample; `k*` = the lag maximizing that mutual
information, found by sweeping `k=1..12`.

**Why CVSI vs. the Standardized Precipitation Index (SPI) / SPEI family** (the
established drought-index alternatives): SPI/SPEI require precipitation (and for
SPEI, potential evapotranspiration) as direct inputs and are defined over fixed,
literature-conventional windows (1/3/6/12 months); CVSI is NDVI-native (uses the
vegetation response itself, not a separate meteorological proxy for it) and its
window length is *fit directly against this project's real fire labels* via mutual
information, rather than assumed from convention. **Why cumulative one-directional
stress vs. a plain rolling-sum of the anomaly** (`Σδ` over the window, without the
`max(·,0)` floor): a plain rolling sum lets wet months cancel out dry ones,
diluting the accumulated-drought signal the feature is specifically meant to
isolate — `max(−δ,0)` deliberately keeps CVSI monotonically non-decreasing with
drought severity and blind to wet-month "credit." **No literature precedent exists
for this exact index** — it is a genuine methodological contribution of this
project. Result: **k\*=8** (MI peaks 0.01246, confirmed interior optimum after
extending the original k=1..6 sweep to k=1..12). General MI reference: Cover &
Thomas (2006), *Elements of Information Theory* (2nd ed.), Wiley [cite-confirmed]
— not a citation for CVSI itself.

### F8 — LISA cluster map
```
I = (N/W) · [Σ_i Σ_j w_ij(x_i−x̄)(x_j−x̄)] / Σ_i(x_i−x̄)²                (global)
I_i = [(x_i − x̄) / m₂] · Σ_j w_ij(x_j − x̄),   m₂ = Σ_i(x_i−x̄)²/N        (local)
```
**Definitions**: `x_i` = the value (forest-cover indicator) at coarse cell `i`;
`x̄` = the grid mean; `N` = total coarse-cell count; `w_ij` = the spatial weight
between cells `i,j` (1 if queen-contiguous neighbors — sharing an edge or corner —
0 otherwise, row-standardized); `W` = `Σ_i Σ_j w_ij`, the sum of all weights; `I` =
global Moran's I (single national value, measures overall spatial autocorrelation);
`I_i` = local Moran's I for cell `i`; `m₂` = the sample variance of `x`. Local
results are categorized into quadrants — High-High (dense forest cluster),
Low-Low (sparse/degraded cluster), High-Low (forest fragment in sparse
surroundings), Low-High (isolated dense patch) — filtered to `p_sim<0.05` via 199
conditional permutations.

**Why Local Moran's I vs. Getis-Ord Gi\*** (the other standard local
spatial-statistics alternative): Gi\* identifies *hot/cold spots* — clusters of
consistently high or low values — but cannot distinguish a forest *fragment*
(a high-value cell surrounded by low-value neighbors) from a low-value cell in a
low-value neighborhood; Local Moran's I's four-quadrant output specifically
separates similarity clusters (HH/LL) from spatial *outliers* (HL/LH), and the
HL category — an isolated forest patch or edge — is exactly the
wildland-urban-interface-style fragmentation signal relevant to fire risk that
Gi\* would not surface on its own. **[STANDARD]**: Moran (1950), *Biometrika*,
37(1/2):17–23; Anselin (1995), *Geographical Analysis*, 27(2):93–115 [both
cite-confirmed].

### F9 — NDVI–fire breakpoint threshold
```
P(fire=1 | NDVI=x) = σ(a₁+b₁x)   if x ≤ θ
                    = σ(a₂+b₂x)   if x > θ
NLL(a₁,b₁,a₂,b₂,θ) = −Σ_n [y_n·log(p_n) + (1−y_n)·log(1−p_n)]
```
**Definitions**: `σ(·)` = the logistic sigmoid; `θ` = the breakpoint NDVI value
being estimated; `(a₁,b₁)`/`(a₂,b₂)` = the intercept/slope of the fire-probability
response curve below/above `θ` respectively; `y_n∈{0,1}` = the real fire label for
sample `n`; `p_n` = the model's predicted fire probability for sample `n`; fit by
minimizing NLL jointly over all five parameters via Nelder-Mead, 25-point
multi-start over the 10th–90th percentile of sampled NDVI, **θ bounded to the
physically valid NDVI range** (fixed 2026-08-10), on a balanced ≤100k+100k
case-control sample, both nationally and per biogeographic zone.

**Why a fitted piecewise-logistic vs. a ROC/Youden's-J optimal cutpoint** (the
common alternative for "find the best threshold"): Youden's J finds the single
cutpoint maximizing sensitivity+specificity *on top of* a model that's otherwise
fit as one smooth curve — it answers "where's the best decision boundary," not
"does the underlying relationship itself change shape here." This project's
two-regime fit lets the fire-probability response have a genuinely different slope
above vs. below `θ` (e.g. flat-then-rising, or rising-then-plateauing) — a real
structural claim about the data, not just a decision rule imposed on a
single-regime model. **Why not a single decision-tree split** (CART-style, optimize
information gain on one split): equivalent in spirit but doesn't produce a smooth
probability curve on either side — useful for classification but not for
characterizing *how* fire probability responds to NDVI away from the threshold
itself, which the logistic slopes `b₁,b₂` capture. **[ADAPTED]** — piecewise/
segmented regression is standard (general reference: Muggeo (2003), *Statistics in
Medicine*, 22(19):3055–3071 [cite-verify]); applying it as a two-regime logistic
against real satellite-derived fire labels, per biogeographic zone, is this
project's construction.

**Robustness fix (2026-08-10)**: Himalayan zone's θ\* was −0.613 (invalid). Root
cause: **MLE non-identifiability**, not sample size or weak signal (both
investigated and ruled out) — the unbounded optimizer let θ drift onto a flat,
non-identifiable NLL plateau once the winning solution emptied one regime entirely
(`n_below=0`). Fixed by bounding θ to the valid NDVI range and rejecting
degenerate (regime-collapse) solutions. Corrected: θ\*=−0.001, a genuine interior
optimum.

---

## Step 3 — LST Features (5 features)

**Product**: MOD11A2.061, 8-day composite, 1km. Climatology/anomaly/Mann-Kendall
use **identical formulas and definitions to Step 2** — not re-derived here.

```
DTR(t,i,j) = LST_Day(t,i,j) − LST_Night(t,i,j)
```
**Definitions**: `LST_Day`/`LST_Night` = the 8-day-composite land surface
temperature during satellite day-pass/night-pass respectively (Kelvin); `DTR` =
diurnal temperature range, with its own independently-computed anomaly and
Mann-Kendall trend (not derived by differencing the Day/Night τ values).

**Why engineer DTR as its own feature vs. leaving Day and Night LST as two separate
inputs**: a model *could* in principle learn the Day-minus-Night relationship on
its own from the two raw features — but Random Forest and similar tree-based
models split on **one feature at a time**, so an interaction/difference like DTR is
not automatically discoverable the way it would be for a model that considers
feature products/differences directly; explicitly engineering DTR hands the model
a physically-meaningful signal (atmospheric dryness/evaporative demand — large DTR
means dry air, hot days and cold nights) that would otherwise require the tree
ensemble to approximate via multiple splits on both raw features. **[ADAPTED]** —
LST anomaly/trend methodology is standard remote-sensing practice; using DTR
specifically as a fire-risk feature (rather than just Day/Night means) is this
project's choice, motivated generally by fire-weather literature linking DTR to
fuel-moisture deficit. **[not cited in-notebook]** MOD11 algorithm: Wan (2014),
*Remote Sensing of Environment*, 140:36–45, DOI:10.1016/j.rse.2013.08.027
[cite-confirmed].

**Results**: Day τ=−0.041, Night τ=+0.043, DTR τ=−0.104 (strongest of the three —
consistent with day-cooling and night-warming compounding in the same direction for
DTR specifically). Significance (p<0.05): Day 1,063,120 px, Night 234,318 px, DTR
2,545,287 px.

---

## Step 4 — FLDAS Climatic Variables + Land Cover (12 + 22 features)

> **Renumbered 2026-08-17**: was "Step 6" before — moved to Step 4 since it's an
> independent preprocessing step that always ran before assembly/training
> regardless of its old number. No content changed, only the label.

**Product**: FLDAS_NOAH01_C_GL_M.001, 0.1°/~11km, monthly. McNally et al. (2017),
*Scientific Data*, 4:170012, DOI:10.1038/sdata.2017.12 [cite-confirmed]. 6 variables
(air temp, wind, precip, RH, soil moisture, net LW radiation), each with
climatology/anomaly/Mann-Kendall τ+p identical in method to Steps 2/3.

**Why these six variables vs. the full FLDAS variable list** (FLDAS/Noah LSM
outputs dozens of fields): these six were chosen because together they approximate
the core inputs of established fire-weather indices — temperature, wind, humidity,
and precipitation all appear in both the Canadian Fire Weather Index (FWI) System
and the McArthur Forest Fire Danger Index — giving this project's climatic coverage
a defensible link to literature fire-danger formulations without directly
reimplementing either index's specific combination formula.

**Relative humidity** (derived, not native to FLDAS):
```
e_s(hPa) = 6.112 · exp[17.67·T_c / (T_c + 243.5)]
e(hPa)   = q·p / (0.622 + 0.378·q)
RH(%)    = 100 · e / e_s,   clipped to [0,100]
```
**Definitions**: `T_c` = air temperature in Celsius; `e_s` = saturation vapor
pressure at that temperature; `q` = specific humidity (kg water vapor / kg moist
air, from `Qair_f_tavg`); `p` = surface pressure in hPa (from `Psurf_f_tavg`); `e`
= actual vapor pressure.

**Why the Magnus/Tetens formula vs. more exact alternatives** (e.g. the Buck 1981
equation, or full numerical integration of the Clausius-Clapeyron relation):
Magnus/Tetens is a well-validated closed-form approximation, accurate to roughly
0.1–0.3% over typical terrestrial temperature ranges — Buck's equation is
marginally more accurate at temperature extremes (very cold/hot conditions) at the
cost of more terms; since this project uses RH primarily through its *anomaly* and
*trend* (relative changes, not absolute meteorological precision), the small
extra accuracy Buck's formula would offer over Magnus/Tetens is not decision-relevant
here. **[STANDARD]** formula, **[ADAPTED]** application (deriving RH specifically
to fill a fire-weather-relevant variable FLDAS doesn't output directly). Tetens
(1930), *Zeitschrift für Geophysik*, 6:297–309 [cite-verify].

**22-class land cover**:
```
base_code(i,j) = ⌊ raw_LCCS_code(i,j) / 10 ⌋ × 10
```
**Definitions**: `raw_LCCS_code` = the native ESA-CCI/C3S Level-2 land-cover class
code (38 possible values, e.g. 61/62 splitting broadleaved-evergreen into
closed/open canopy); `⌊·⌋` = integer floor division; `base_code` = the Level-1
parent class (22 possible values: 10, 20, 30, ..., 220). Reprojected 300m→~1km via
the same area-weighted `Resampling.average` as Step 5's forest fraction (below).

**Why collapse to Level-1 (22 classes) vs. keeping full Level-2 granularity (38
classes)**: Level-1 is the official ESA CCI legend tier (verified 2026-08-09 against
ESA CCI's own documentation — not a project guess, as an earlier in-notebook
comment had flagged it), and using 22 rather than 38 feature columns keeps the
final feature table's land-cover block at a size comparable to the other feature
groups (12 climatic, 9 NDVI, 5 LST) rather than letting land-cover sub-variants
dominate the column count, while the area-weighted fractional-cover approach still
preserves the sub-variant information *within* each parent class's fraction value
(a cell that's entirely closed-canopy vs. entirely open-canopy broadleaved-evergreen
still both register as 100% of class "50," but the fractional-cover values of
neighboring/mixed cells still carry real information at the 1km output resolution).
**[STANDARD]**.

**Results**: RH shows the most extensive significant trend (46.6% of valid pixels,
overwhelmingly increasing), soil moisture next (40.4%). **Corrected 2026-08-18**:
Biswas et al.'s actual predictor set (their Table 3) is **15 variables**, not 11 —
verified by direct extraction from the user's own copy of the paper. Burned area isn't
one of the 15 (it's a Table 2 dataset, not a Table 3 predictor), so it was never
actually the sole gap. The real gaps against their 15 are **distance to
roads/railways/waterways** (OSM 2022, 10.8% combined contribution in their model) and
**slope/aspect/elevation** (DEM-derived, 9.7% combined contribution) — six variables
this pipeline has not computed, not one. See `METHODOLOGY.md`'s Step 4 section for the
full corrected variable table.

**Closed 2026-08-18, numbered and split into two repos 2026-08-19**: all six built as
Step 5a (`Terrain_Elevation_Slope_Aspect_Analysis/`) and Step 5b
(`Distance_Roads_Railways_Waterways_Analysis/`), run alongside Step 4 and feeding
Step 6 (Integration, bumped from Step 5 to make room) — see each repo's own README for
full results and methodology. This pipeline now covers all 15 of Biswas et al.'s real
predictor variables. **Wired into Step 6's pixel table 2026-08-20** (60-band stack /
62-column parquet, up from 54/56 — see the Integration section below); **Step 7's model
was retrained on the expanded table the same day** (2026-08-20) — see the Step 6
(Susceptibility Model) section below for results.

---

## Step 5 — Integrated Alignment (60 features assembled)

> **Renumbered 2026-08-17**: was "Step 4" before — moved to Step 5, after FLDAS
> (now Step 4), since it depends on FLDAS's output and always ran after it. (This
> section's own heading is stale by one further renumbering pass — see the
> Integration notebook's actual current numbering, Step 6 as of 2026-08-19 — but is
> left as-is here since fixing the full renumbering trail is out of this task's scope;
> only the feature/column counts below have been corrected.)
>
> **Updated 2026-08-20**: Step 5a (Terrain) + Step 5b (Accessibility)'s 6 bands
> (elevation, slope, aspect, distance to roads/railways/waterways) are now wired in —
> 54 → 60 feature layers, 56 → 62 columns in the flattened pixel table. This closes
> the pipeline's last gap against Biswas et al.'s real 15-variable predictor set.

```
forest_frac(i,j) = (1/N_sub) Σ_{sub-pixel ∈ cell(i,j)} 1[LULC_sub ∈ ForestCodes]
```
**Definitions**: `N_sub` = the number of native 300m LULC sub-pixels falling inside
1km destination cell `(i,j)`; `1[·]` = the indicator function (1 if the sub-pixel's
land-cover code is in the 13-code forest set, 0 otherwise); computed via
`rasterio.warp.reproject(..., resampling=Resampling.average)`, which is exactly the
area-weighted mean of this binary mask. Three snapshots (baseline=2001, recent=2020,
current=2022); `forest_loss = forest_frac_baseline − forest_frac_recent`.

**Why area-weighted averaging vs. majority-class ("mode") resampling** (the more
common default when downsampling a categorical raster): mode resampling would
collapse a 1km cell that's 60% forest / 40% cropland to a single hard label
("forest"), discarding exactly the continuous fragmentation information that
Step 2's F8 (LISA) already establishes is meaningful for fire-edge risk —
area-weighted averaging of the *binary forest mask specifically* preserves this as
a continuous `[0,1]` fraction instead. **[ADAPTED]** — area-weighted resampling is
a standard GIS operation; applying it to preserve fractional forest cover (rather
than snapping to a majority class) is a deliberate choice here.

**Forest-class reconciliation (2026-08-10)**: now uses the same 13-code set as
Step 1 (previously 11 codes — an undocumented divergence, omitted mosaic-tree/shrub
and mosaic-herbaceous). National mean forest fraction rose ~7.8–8.0% → **10.2–10.7%**;
these three features became the **top 3** Step 6 predictors by Gini importance
after the fix.

**Result**: `Integrated_FireRisk_Pixels.parquet`, **4,161,009 pixels × 62 columns**
(60 features + lon + lat), following the 2026-08-20 addition of Step 5a/5b's 6
terrain/accessibility bands (`terrain_elevation`, `terrain_slope`, `terrain_aspect`,
`access_dist_roads`, `access_dist_railways`, `access_dist_waterways`).

---

## Step 6 — Susceptibility Model (Random Forest + MaxEnt)

> **Renumbered 2026-08-17**: was "Step 5" before — moved to Step 6, now the
> genuinely last step in the actual execution order.

```
Gini(node) = 1 − Σ_c p_c²
ROC-AUC = P(score(positive) > score(negative))
AP = Σ_n (R_n − R_{n−1}) · P_n
```
**Definitions**: `p_c` = the fraction of samples at a tree node belonging to class
`c` (fire / no-fire); a split is chosen to minimize the weighted Gini impurity of
the resulting child nodes; feature importance = total Gini decrease attributable to
a feature, summed across all splits/trees, normalized; `score(·)` = the model's
predicted fire probability for a sample; `R_n, P_n` = recall and precision at the
`n`-th threshold along the precision-recall curve (AP = the area under that curve,
via a step-weighted sum).

**Why Random Forest as the primary baseline vs. Gradient Boosting** (e.g. XGBoost,
also included in this project's separate Step 7 comparison ladder): RF trains
trees independently on bootstrap samples (bagging) — embarrassingly parallel,
lower variance from averaging, less sensitive to hyperparameter choices, and its
bit-exact reproducibility is straightforward to verify (this project explicitly
does, max diff 7.77e-16 across reruns). Gradient boosting trains trees sequentially
to correct prior residual errors — often higher raw accuracy (confirmed here:
Step 7's XGBoost slightly edges out RF, 0.9678 vs 0.9676 on a comparable split) but
more hyperparameter-sensitive and slower to verify exact reproducibility. RF is
used as the primary, most-scrutinized baseline for exactly the properties that make
it easiest to defend rigorously; XGBoost's comparison lives in the separate model
ladder (Step 7) rather than replacing RF here. **Why RF vs. MaxEnt** (Biswas et
al.'s own method, now also trained here as a direct comparison): MaxEnt is a
presence-background method — it was originally designed for species distribution
modeling where only presence records (and no confirmed absences) are typically
available, and it models presence probability relative to a background sample.
This project has confirmed non-fire pixels (true negatives, not just unlabeled
background), which a fully discriminative method like RF can use directly and
MaxEnt's classical formulation does not exploit the same way — RF is the more
natural fit for this project's actual label structure, while MaxEnt is included
specifically *because* it's the established comparison point in the literature this
project extends. **[STANDARD]**: Breiman (2001), *Machine Learning*, 45(1):5–32;
Davis & Goadrich (2006), *ICML '06*, 233–240 [both cite-confirmed].

**Current result (retrained 2026-08-20 on the 58-feature table, after terrain +
accessibility were wired into Step 6)**: Random Forest ROC-AUC 0.9683, AP 0.6796,
5-fold CV 0.9679±0.0002 (up from 0.9674 / 0.6761 / 0.9670±0.0002 on the prior
52-feature table). MaxEnt (`elapid`, trained on a 150,000-row stratified subsample of
the training portion — recalibrated down from an original 450k-row target after a
direct timing probe showed super-linear fit-time scaling on this data/library
combination — evaluated on the full 832,202-row test set): ROC-AUC 0.9595, AP 0.6237
(up from 0.9576 / 0.6111). Random Forest still outperforms MaxEnt by 0.0088 ROC-AUC /
0.0559 AP. Top 5 features by Gini importance unchanged in kind (forest-fraction + NDVI
variables dominate); of the 6 new terrain/accessibility features, `terrain_slope` is
the strongest, ranking 6th overall (just outside the top 5) — a real, testable signal
directly consistent with Step 5a's fire-coincidence finding (fires sit at +115% mean
slope vs. the national average). See `Integrated_Analysis/README.md` for the full
results table, per-fold AUCs, and complete feature-importance ranking.

---

## Cross-cutting: what makes this pipeline defensible

1. **Identical anomaly/trend methodology across Steps 2/3/4** — one implementation
   to verify, not six.
2. **Every "why this method" comparison above names a specific, real alternative**
   (STL, SPI/SPEI, Getis-Ord Gi\*, Youden's J, gradient boosting, Buck's equation,
   Level-2 granularity) rather than asserting the chosen method without contrast.
3. **Novel contributions are isolated**: CVSI (F7) is the one feature with zero
   literature precedent; everything else is standard or a disclosed, motivated
   adaptation.
4. **Every fitted (not assumed) parameter is fit against this project's own
   541,545 real fire labels** — CVSI's lag, F9's breakpoint — not borrowed from a
   different geography's literature values.
5. **Known limitations are stated, not hidden**: burned-area coverage gap, two
   forest-class definitions reconciled, an MLE non-identifiability bug found and
   fixed with root-cause evidence, a lag-sweep boundary resolved by extending the
   sweep rather than asserted.
