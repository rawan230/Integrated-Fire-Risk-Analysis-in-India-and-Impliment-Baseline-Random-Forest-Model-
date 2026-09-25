"""
Assemble the audit feature tables on the historical pixel universe (the 4,161,009 rows of
Integrated_FireRisk_Pixels.parquet), joining the independent recalculations R1-R7.

Outputs (results/recalculated/):
  FEATURE_TABLE_v2.parquet   -- v1 features (prefix v1_), v2 features (prefix v2_),
                                Biswas-15 level set (prefix b15_), labels, populations, splits
  FEATURE_SETS.json          -- the exact column lists of every feature set used downstream
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import P, RES, provenance, dump, ndvi_grid  # noqa: E402

REC = os.path.join(RES, "recalculated")
g = ndvi_grid(); H, W = g["shape"]
pq = pd.read_parquet(P["parquet"])
n0 = len(pq)
rr = np.floor((pq.lat.values - g["transform"].f) / g["transform"].e).astype(np.int64)
cc = np.floor((pq.lon.values - g["transform"].c) / g["transform"].a).astype(np.int64)
gi = rr * W + cc
T = pd.DataFrame({"lon": pq.lon.values, "lat": pq.lat.values, "grid_index": gi})

# ---------------- v1 (historical) features, exactly as trained
v1_cols = [c for c in pq.columns if c not in ("lon", "lat", "fire_count", "fire_ever")]
for c in v1_cols:
    T[f"v1_{c}"] = pq[c].values.astype(np.float32)
T["y_v1_hist_shifted"] = (pq.fire_ever.values > 0).astype(np.int8)

# ---------------- labels (R1, containing-pixel rasterisation)
lab = np.load(os.path.join(REC, "R1_fire_labels", "fire_count_variants_ndvi_grid.npz"))
for k in lab.files:
    T[f"y_{k}"] = (lab[k].ravel()[gi] > 0).astype(np.int8)
T["fire_count_corrected"] = lab["all"].ravel()[gi].astype(np.int32)

# ---------------- NDVI (R4) joined on grid index
nd = pd.read_parquet(os.path.join(REC, "R4_ndvi", "ndvi_features_india_pixels.parquet"))
nd = nd.set_index("grid_index")
j = nd.reindex(gi)
T["in_R4"] = j["v2_F1_mean"].notna().values
T["v2_split"] = j["v2_split"].fillna(-1).astype(np.int8).values
nd_map = {"v2_ndvi_mean": "v2_F1_mean", "v2_ndvi_clim_june": "v2_F2_clim_june", "v2_ndvi_sk_tau": "v2_sk_tau",
          "v2_ndvi_sen_slope": "v2_sen_per_yr", "v2_ndvi_cvsi": "v2_cvsi_kstar_trainonly", "v2_ndvi_lisa": "v2_lisa_cluster"}
for new, old in nd_map.items():
    T[new] = j[old].values.astype(np.float32)
T["chk_label_R4_vs_R1"] = (j["fire_ever_corrected"].values == T["y_all"].values)

# ---------------- FLDAS (R2), LST (R5), LULC (R3), terrain (R6), access (R7) -- all row-aligned to pq
fl = pd.read_parquet(os.path.join(REC, "R2_fldas", "fldas_v2_features_pixels.parquet"))
assert np.allclose(fl.lon.values, pq.lon.values)
for v in ("tair", "qair", "rh", "wind", "precip", "lwnet", "soilm"):
    T[f"v2_{v}_level"] = fl[f"{v}_level_clim_annual"].values.astype(np.float32)
    T[f"v2_{v}_sk_tau"] = fl[f"{v}_sk_tau"].values.astype(np.float32)
ls = pd.read_parquet(os.path.join(REC, "R5_lst", "lst_features_pixels.parquet"))
assert np.allclose(ls.lon.values, pq.lon.values)
for b in ("day", "night", "dtr"):
    T[f"v2_lst_{b}_level"] = ls[f"{b}_level_clim_annual"].values.astype(np.float32)
    T[f"v2_lst_{b}_sk_tau"] = ls[f"{b}_sk_tau"].values.astype(np.float32)
for b in ("day", "night"):
    T[f"sens_lst_{b}_level_005deg"] = ls[f"{b}_level_clim_annual_005deg"].values.astype(np.float32)
lc = pd.read_parquet(os.path.join(REC, "R3_lulc", "lc22_fractions_2001_pixels.parquet"))
for c in lc.columns:
    if c != "lc2001_140":
        T[f"v2_{c}"] = lc[c].values.astype(np.float32)
ff = pd.read_parquet(os.path.join(REC, "R3_lulc", "forest_frac_pixels.parquet"))
T["v2_forest_frac_2001"] = ff.forest_frac_2001.values.astype(np.float32)
T["forest_frac_2020"] = ff.forest_frac_2020.values.astype(np.float32)
te = pd.read_parquet(os.path.join(REC, "R6_terrain", "terrain_pixels.parquet"))
T["v2_elevation"] = te.elevation.values.astype(np.float32)
T["v2_slope"] = te.slope_horn.values.astype(np.float32)
asp = np.radians(te.aspect.values)
T["v2_aspect_sin"] = np.sin(asp).astype(np.float32); T["v2_aspect_cos"] = np.cos(asp).astype(np.float32)
T["sens_slope_zt"] = te.slope_zt.values.astype(np.float32)
T["sens_slope_from_025deg_dem"] = te.slope_from_025deg_dem.values.astype(np.float32)
T["b15_aspect_deg"] = te.aspect.values.astype(np.float32)
_acp = os.path.join(REC, "R7_access", "access_pixels.parquet")
ACCESS_SOURCE = "R7 recalculation" if os.path.exists(_acp) else "historical parquet columns (R7 pending; same method -- verified separately in R7)"
for k in ("roads", "railways", "waterways"):
    T[f"v2_dist_{k}"] = (pd.read_parquet(_acp)[k].values if os.path.exists(_acp) else pq[f"access_dist_{k}"].values).astype(np.float32)

# ---------------- populations
T["pop_all"] = T["in_R4"] & (T["v2_split"] >= 0)
T["pop_forest"] = T["pop_all"] & (T["v2_forest_frac_2001"] > 0)

# ---------------- spatial partitions, identical geometry to the CDR-PINO tracks
LON_MIN, LAT_MIN = 68.20, 6.75
T["block_2deg"] = (np.floor((T.lon - LON_MIN) / 2.0).astype(int) * 1000 + np.floor((T.lat - LAT_MIN) / 2.0).astype(int)).astype(np.int32)
d = np.load(os.path.join(P["parquet"].split("Integrated_Analysis")[0], "Physics_Informed_FireRisk_Model", "CDR_PINN_Data", "cdr_pinn_monthly_stacks.npz"))
valid12 = ~np.isnan(d["ndvi_f1"])
lat_deg = np.linspace(37.09, LAT_MIN, 256); lon_deg = np.linspace(LON_MIN, 97.40, 256)
LG, AG = np.meshgrid(lon_deg, lat_deg)
blk12 = np.floor((LG - LON_MIN) / 2.0).astype(int) * 1000 + np.floor((AG - LAT_MIN) / 2.0).astype(int)
vb = np.unique(blk12[valid12])
perm = np.random.RandomState(42).permutation(vb)
fold_of = {int(b): k for k, f in enumerate(np.array_split(perm, 3)) for b in f}
T["b1_fold"] = T.block_2deg.map(fold_of).fillna(-1).astype(np.int8)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(np.stack([LG[valid12], AG[valid12]], 1))
T["b2_region"] = km.predict(np.stack([T.lon.values, T.lat.values], 1)).astype(np.int8)
# validation carve for spatial folds: whole blocks (same rule as the unified CDR runner)
def carve(train_blocks, seed=42, frac=0.1875):
    rng = np.random.RandomState(seed + 1000)
    return set(rng.choice(np.array(sorted(train_blocks)), size=max(1, int(round(len(train_blocks) * frac))), replace=False).tolist())
meta = {"b1_val_blocks": {}, "b2_val_blocks": {}}
for k in range(3):
    trb = np.unique(T.block_2deg[(T.b1_fold != k) & (T.b1_fold >= 0)])
    meta["b1_val_blocks"][k] = sorted(int(x) for x in carve(trb))
for r in range(6):
    trb = np.unique(T.block_2deg[T.b2_region != r])
    meta["b2_val_blocks"][r] = sorted(int(x) for x in carve(trb))
# B3-analogue labels for static models: burned in held-out years vs burned in fit years
TEST_YEARS, VAL_YEARS = [2000, 2008, 2009, 2015], None
fp = pd.read_parquet(os.path.join(REC, "R1_fire_labels", "forest_fire_points_recalc.parquet"), columns=["latitude", "longitude", "year"])
r1 = np.floor((fp.latitude.values - g["transform"].f) / g["transform"].e).astype(np.int64)
c1 = np.floor((fp.longitude.values - g["transform"].c) / g["transform"].a).astype(np.int64)
gi1 = r1 * W + c1
for name, sel in (("y_testyears", np.isin(fp.year.values, TEST_YEARS)), ("y_nontestyears", ~np.isin(fp.year.values, TEST_YEARS))):
    s = pd.Index(np.unique(gi1[sel]))
    T[name] = pd.Index(gi).isin(s).astype(np.int8)

# ---------------- feature sets
v2 = [c for c in T.columns if c.startswith("v2_") and c != "v2_split"]
biswas15 = {"NDVI": "v2_ndvi_mean", "Air temperature": "v2_tair_level", "Specific humidity": "v2_qair_level",
            "LST night": "v2_lst_night_level", "LST day": "v2_lst_day_level", "Distance to roads": "v2_dist_roads",
            "Slope": "v2_slope", "Distance to railways": "v2_dist_railways", "Soil moisture": "v2_soilm_level",
            "Precipitation": "v2_precip_level", "Near-surface wind speed": "v2_wind_level", "Elevation": "v2_elevation",
            "Net longwave radiation flux": "v2_lwnet_level", "Aspect": "b15_aspect_deg", "Distance to waterways": "v2_dist_waterways"}
sets = {
    "v1_57": [f"v1_{c}" for c in v1_cols],
    "v2_57": v2,
    "biswas15": list(biswas15.values()),
    "biswas15_map": biswas15,
    "cdr7_static_1km": ["v2_ndvi_mean", "v2_forest_frac_2001", "v2_slope", "v2_dist_roads", "v2_elevation"],
}
sets["v1_minus_labelderived"] = [c for c in sets["v1_57"] if c not in ("v1_ndvi_below_threshold", "v1_ndvi_cvsi_k8")]
sets["v2_minus_landcover"] = [c for c in v2 if not c.startswith("v2_lc2001_") and c != "v2_forest_frac_2001"]
sets["v2_minus_trends"] = [c for c in v2 if not c.endswith("_sk_tau") and c != "v2_ndvi_sen_slope"]
sets["v2_minus_terrain"] = [c for c in v2 if c not in ("v2_elevation", "v2_slope", "v2_aspect_sin", "v2_aspect_cos")]
groups = {
    "vegetation": [c for c in v2 if c.startswith("v2_ndvi_")],
    "climate_levels": [f"v2_{v}_level" for v in ("tair", "qair", "rh", "wind", "precip", "lwnet", "soilm")],
    "climate_trends": [f"v2_{v}_sk_tau" for v in ("tair", "qair", "rh", "wind", "precip", "lwnet", "soilm")],
    "lst": [c for c in v2 if c.startswith("v2_lst_")],
    "terrain": ["v2_elevation", "v2_slope", "v2_aspect_sin", "v2_aspect_cos"],
    "human": ["v2_dist_roads", "v2_dist_railways", "v2_dist_waterways"],
    "landcover": [c for c in v2 if c.startswith("v2_lc2001_")] + ["v2_forest_frac_2001"],
}
assert sorted(sum(groups.values(), [])) == sorted(v2), set(v2) ^ set(sum(groups.values(), []))
sets["v2_groups"] = groups
out = os.path.join(REC, "FEATURE_TABLE_v2.parquet")
T.to_parquet(out, index=False)
with open(os.path.join(REC, "FEATURE_SETS.json"), "w") as f:
    json.dump(sets, f, indent=1)
rep = dict(provenance=provenance(__file__), access_source=ACCESS_SOURCE, n_rows=n0, n_pop_all=int(T.pop_all.sum()), n_pop_forest=int(T.pop_forest.sum()),
           n_v1_features=len(sets["v1_57"]), n_v2_features=len(v2), label_R4_R1_agree=float(T.chk_label_R4_vs_R1[T.in_R4].mean()),
           prevalence={k: dict(all=float(T.loc[T.pop_all, k].mean()), forest=float(T.loc[T.pop_forest, k].mean()))
                       for k in [c for c in T.columns if c.startswith("y_")]},
           v2_nan_pct={c: float(100 * T.loc[T.pop_all, c].isna().mean()) for c in v2},
           b1_fold_pixels={int(k): int(v) for k, v in T.loc[T.pop_all, "b1_fold"].value_counts().items()},
           b2_region_pixels={int(k): int(v) for k, v in T.loc[T.pop_all, "b2_region"].value_counts().items()},
           split_counts={int(k): int(v) for k, v in T.loc[T.pop_all, "v2_split"].value_counts().items()},
           partitions_meta=meta, kmeans_centroids=km.cluster_centers_.tolist())
dump(rep, os.path.join(REC, "FEATURE_TABLE_v2_report.json"))
print({k: v for k, v in rep.items() if k not in ("v2_nan_pct", "partitions_meta", "provenance")})
