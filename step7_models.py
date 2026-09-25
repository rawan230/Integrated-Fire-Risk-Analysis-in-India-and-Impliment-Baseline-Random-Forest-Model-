"""
Step 7 -- classical fire-susceptibility models under the corrected ("v2") evaluation protocol
of the 2026-09-24/25 end-to-end audit of record (results/FULL_METHODOLOGY_AUDIT.md;
Integrated_Analysis/AUDIT_2026-09-25.md). This module is a port of the audit's verified
reference code (results/code/models_classical.py, analyze_classical.py, bridge_12km.py,
biswas_style_baseline.py, make_final_map.py, build_v2_table.py partition logic) onto
Step 6's regenerated v2 pixel table. All outputs go under Integrated_Analysis/Model_Outputs/.

Run order (each subcommand is independent once its inputs exist; `analyze` last):

  python step7_models.py trackA     # random 65/15/20; 8 models; importance + PDP for 4 models
  python step7_models.py B1         # 3-fold 2-degree-block CV, CDR-PINO fold geometry
  python step7_models.py B2         # leave-one-region-out, CDR-PINO KMeans regions (6)
  python step7_models.py B3         # static temporal analogue (held-out years) + persistence null
  python step7_models.py maxent_n   # MaxEnt training-sample-size sensitivity (+ fit times)
  python step7_models.py bridge     # 12 km CDR-PINO population, identical partitions
  python step7_models.py biswas     # Biswas-style 0.25-degree REIMPLEMENTATION
  python step7_models.py map        # RF v2 relative-score GeoTIFF + 5-class forest-quantile map
  python step7_models.py analyze    # metric tables, paired DeLong / block bootstrap, figures

Optional arguments: --only RF_v2,MaxEnt_v2 (restrict models for trackA/B1/B2),
--parquet PATH, --out DIR, --partitions PATH, --fire-points PATH, --cdr-stacks PATH.

Every fit appends one JSON row to Model_Outputs/metrics/experiment_registry.jsonl and saves
its test predictions to Model_Outputs/predictions/{exp}__{tag}[__f{fold}|__r{region}].npz
(bridge: Model_Outputs/bridge_predictions/bridge12_{track}__{tag}.npz, same naming as the
audit) so that paired DeLong / block-bootstrap comparisons use identical test pixels.
Re-running a stage appends new registry rows; `analyze` keeps the LAST row per key.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import preprocessing as pp  # noqa: E402
from step7_eval import metrics, boot_ci, delong_paired, block_boot_diff  # noqa: E402

# --------------------------------------------------------------------------- configuration
N_JOBS = 20
RF_HP = dict(n_estimators=200, max_depth=25, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", random_state=42)
MX_HIST = dict(feature_types=["linear", "hinge", "product"], beta_multiplier=4.0)          # Step 7 validated (2026-08-23)
MX_BISWAS = dict(feature_types=["linear", "quadratic", "hinge", "product"], beta_multiplier=1.0)  # MaxEnt 3.4 'auto' (>=80 presences), default reg.
MX_SUBSAMPLE = 150000

DEFAULTS = dict(
    parquet=pp.PARQUET_PATH,
    out=os.path.join(HERE, "Model_Outputs"),
    partitions=os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "CDR_PINN_Data", "unified", "partitions.npz"),
    cdr_stacks=os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "CDR_PINN_Data", "cdr_pinn_monthly_stacks.npz"),
    fire_points=os.path.join(ROOT, "Forest fire Extraction in INDIA(2000-2022)", "Forest_Fire_Outputs", "all_forest_fires_2000_2022.csv"),
    ndvi_raw=os.path.join(ROOT, "NDVI_DATA_INDIA_", "NDVI TIF File_INDIA"),
    stack_tif=os.path.join(HERE, "Integrated_Outputs", "Integrated_FireRisk_Stack.tif"),
)

# 12 km CDR-PINO grid and 2-degree block geometry (identical to the unified CDR runner)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 68.20, 97.40, 6.75, 37.09
SPLIT_SEED = 42
VAL_BLOCK_FRAC = 0.1875
FALLBACK_B3_TEST_YEARS = [2000, 2008, 2009, 2015]

# Biswas et al. (2025) Table 3 predictors -> v2 column names. Aspect is reconstructed in DEGREES
# from aspect_sin/aspect_cos (column "aspect_deg", never part of the v2 feature list).
BISWAS15 = {"NDVI": "ndvi_mean", "Air temperature": "tair_level", "Specific humidity": "qair_level",
            "LST night": "lst_night_level", "LST day": "lst_day_level", "Distance to roads": "dist_roads",
            "Slope": "slope", "Distance to railways": "dist_railways", "Soil moisture": "soilm_level",
            "Precipitation": "precip_level", "Near-surface wind speed": "wind_level", "Elevation": "elevation",
            "Net longwave radiation flux": "lwnet_level", "Aspect": "aspect_deg", "Distance to waterways": "dist_waterways"}
# CDR-PINO's 5 static covariates at 1 km (tag kept as "RF_cdr7static" for 1:1 comparison with the audit tables)
CDR_STATIC = ["ndvi_mean", "forest_frac_baseline", "slope", "dist_roads", "elevation"]
CLIM = ("tair", "qair", "rh", "wind", "precip", "lwnet", "soilm")
TERRAIN = ["elevation", "slope", "aspect_sin", "aspect_cos"]

MODELS = {
    # tag: (model kind, feature-set key)       -- label is always fire_ever (corrected label)
    "RF_v2": ("RF", "v2"),
    "RF_biswas15": ("RF", "biswas15"),
    "RF_cdr7static": ("RF", "cdr7static"),
    "RF_v2_minus_landcover": ("RF", "v2_minus_landcover"),
    "RF_v2_minus_trends": ("RF", "v2_minus_trends"),
    "RF_v2_minus_terrain": ("RF", "v2_minus_terrain"),
    "MaxEnt_v2": ("MX_HIST", "v2"),
    "MaxEnt_biswas15_LQHP": ("MX_BISWAS", "biswas15"),
}
TRACKA_MODELS = list(MODELS)
IMPORTANCE_MODELS = ("RF_v2", "MaxEnt_v2", "RF_biswas15", "MaxEnt_biswas15_LQHP")
B1_MODELS = ("RF_v2", "RF_biswas15", "RF_cdr7static", "MaxEnt_v2", "MaxEnt_biswas15_LQHP")
B2_MODELS = ("RF_v2", "RF_biswas15", "RF_cdr7static", "MaxEnt_v2", "MaxEnt_biswas15_LQHP")
B3_MODELS = ("RF_v2", "RF_biswas15", "MaxEnt_v2")

PAIRS_A = [("RF_v2", "RF_biswas15", "added predictors / feature engineering"),
           ("RF_v2", "RF_v2_minus_landcover", "land-cover group"),
           ("RF_v2", "RF_v2_minus_trends", "trend features (Seasonal Kendall / Sen)"),
           ("RF_v2", "RF_v2_minus_terrain", "terrain group"),
           ("RF_v2", "RF_cdr7static", "CDR-PINO's 5 static covariates vs v2"),
           ("RF_biswas15", "RF_cdr7static", "Biswas-15 vs CDR-PINO's 5 static covariates"),
           ("RF_v2", "MaxEnt_v2", "model family RF vs MaxEnt (v2)"),
           ("RF_biswas15", "MaxEnt_biswas15_LQHP", "model family RF vs MaxEnt (Biswas-15)"),
           ("MaxEnt_v2", "MaxEnt_biswas15_LQHP", "added predictors under MaxEnt")]
PAIRS_B = [("RF_v2", "RF_biswas15"), ("RF_v2", "MaxEnt_v2"), ("RF_v2", "RF_cdr7static"), ("RF_biswas15", "MaxEnt_biswas15_LQHP")]


class Cfg:
    """Paths, filled by configure()."""


C = Cfg()


def configure(args=None):
    a = vars(args) if args is not None else {}
    for k, v in DEFAULTS.items():
        setattr(C, k, a.get(k) or v)
    C.metrics = os.path.join(C.out, "metrics")
    C.pred = os.path.join(C.out, "predictions")
    C.bridge_pred = os.path.join(C.out, "bridge_predictions")
    C.importance = os.path.join(C.out, "importance")
    C.figures = os.path.join(C.out, "figures")
    C.maps = os.path.join(C.out, "maps")
    C.biswas = os.path.join(C.out, "biswas_reference")
    for d in (C.metrics, C.pred, C.bridge_pred, C.importance, C.figures, C.maps, C.biswas):
        os.makedirs(d, exist_ok=True)
    C.registry = os.path.join(C.metrics, "experiment_registry.jsonl")
    return C


# --------------------------------------------------------------------------- provenance
_SHA_CACHE = {}


def _sha256(path, n=1 << 22):
    if path not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for c in iter(lambda: f.read(n), b""):
                h.update(c)
        _SHA_CACHE[path] = h.hexdigest()
    return _SHA_CACHE[path]


def git_commit(path=HERE):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def provenance():
    return dict(timestamp=dt.datetime.now().isoformat(timespec="seconds"), script="Integrated_Analysis/step7_models.py",
                script_sha256=_sha256(os.path.abspath(__file__)), step7_repo_commit=git_commit(HERE),
                parquet=C.parquet, parquet_sha256=_sha256(C.parquet), python=sys.version.split()[0],
                protocol="v2 (audit of record 2026-09-24/25, results/FULL_METHODOLOGY_AUDIT.md)")


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log(row):
    row = dict(row)
    row["date"] = dt.datetime.now().isoformat(timespec="seconds")
    row["step7_repo_commit"] = git_commit(HERE)
    with open(C.registry, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_default) + "\n")


# --------------------------------------------------------------------------- data
class Data:
    """The v2 pixel table plus derived arrays. Rows = India pixels with finite NDVI mean."""

    def __init__(self, path):
        t0 = time.time()
        T = pd.read_parquet(path)
        missing = [c for c in pp.DROP_COLS + [pp.FOREST_COL, "aspect_sin", "aspect_cos"] + CDR_STATIC if c not in T.columns]
        if missing:
            raise SystemExit(f"{path} is not the v2 table (missing {missing}). Regenerate Step 6 first.")
        gi = T["grid_index"].to_numpy()
        if not (np.diff(gi) > 0).all():
            raise SystemExit("v2 table rows must be ROW-MAJOR by grid_index (strictly increasing); the split depends on row order.")
        self.feats = pp.feature_columns(T)
        if len(self.feats) != pp.N_EXPECTED_FEATURES:
            print(f"WARNING: {len(self.feats)} features, protocol expects {pp.N_EXPECTED_FEATURES}", flush=True)
        # Biswas aspect in degrees, reconstructed from sin/cos (not a model feature of the v2 set)
        T["aspect_deg"] = np.mod(np.degrees(np.arctan2(T["aspect_sin"].to_numpy(np.float64), T["aspect_cos"].to_numpy(np.float64))), 360.0).astype(np.float32)
        self.T = T
        self.n = len(T)
        self.y = pp.label_of(T)
        self.forest = pp.forest_mask(T)
        self.split = pp.v2_split(self.y)
        self.sets, self.groups = feature_sets(self.feats)
        print(f"[data] {self.n:,} rows, {len(self.feats)} features, prevalence all={self.y.mean():.4f} "
              f"forest={self.y[self.forest].mean():.4f} (n_forest={int(self.forest.sum()):,}); load {time.time() - t0:.0f}s", flush=True)

    def X(self, cols):
        return self.T[cols].to_numpy(np.float32)

    def masks(self):
        return self.split == 0, self.split == 1, self.split == 2


def feature_sets(feats):
    lc = [c for c in feats if c.startswith("lc2001_")]
    groups = {
        "vegetation": [c for c in feats if c.startswith("ndvi_")],
        "climate_levels": [f"{v}_level" for v in CLIM],
        "climate_trends": [f"{v}_sk_tau" for v in CLIM],
        "lst": [c for c in feats if c.startswith("lst_")],
        "terrain": list(TERRAIN),
        "human": ["dist_roads", "dist_railways", "dist_waterways"],
        "landcover": lc + [pp.FOREST_COL],
    }
    grouped = sum(groups.values(), [])
    if sorted(grouped) != sorted(feats):
        raise SystemExit(f"feature groups do not partition the v2 feature list: {set(grouped) ^ set(feats)}")
    sets = {
        "v2": list(feats),
        "biswas15": list(BISWAS15.values()),
        "cdr7static": list(CDR_STATIC),
        "v2_minus_landcover": [c for c in feats if c not in groups["landcover"]],
        "v2_minus_trends": [c for c in feats if not c.endswith("_sk_tau") and c != "ndvi_sen_slope"],
        "v2_minus_terrain": [c for c in feats if c not in TERRAIN],
    }
    return sets, groups


# --------------------------------------------------------------------------- partitions
def load_ndvi_grid():
    """Common 1 km grid: raw MOD13A3 GeoTIFF (as in the audit), else Step 6's stack."""
    import glob
    import rasterio
    fs = sorted(glob.glob(os.path.join(C.ndvi_raw, "*_monthly_NDVI_doy*.tif")))
    src = fs[0] if fs else C.stack_tif
    with rasterio.open(src) as s:
        return dict(transform=s.transform, crs=s.crs, shape=s.shape, source=src)


def rebuild_partitions(stacks_path):
    """Re-create the unified CDR runner's partitions.npz (cdr_unified_runner.py) from the
    monthly stacks: valid cells, 2-degree block ids, B1 fold permutation, B2 KMeans regions,
    Track-A 65/15/20 cell split (cdr_pinn/preprocessing.build_masks_3way), B3 held-out years."""
    from sklearn.cluster import KMeans
    d = np.load(stacks_path)
    ndvi_f1 = d["ndvi_f1"]
    H, W = ndvi_f1.shape
    valid = ~np.isnan(ndvi_f1)
    lat_deg = np.linspace(LAT_MAX, LAT_MIN, H)
    lon_deg = np.linspace(LON_MIN, LON_MAX, W)
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg)
    block_id = np.floor((lon_grid - LON_MIN) / 2.0).astype(int) * 1000 + np.floor((lat_grid - LAT_MIN) / 2.0).astype(int)
    vb = np.unique(block_id[valid])
    perm = np.random.RandomState(SPLIT_SEED).permutation(vb)
    folds = [set(f.tolist()) for f in np.array_split(perm, 3)]
    km = KMeans(n_clusters=6, random_state=SPLIT_SEED, n_init=10).fit(np.stack([lon_grid[valid], lat_grid[valid]], 1))
    region = np.full(lon_grid.shape, -1, dtype=int)
    region[valid] = km.labels_
    rng = np.random.RandomState(SPLIT_SEED)
    vidx = np.argwhere(valid)
    p = rng.permutation(len(vidx))
    n_test, n_val = int(len(vidx) * 0.20), int(len(vidx) * 0.15)
    masks = []
    for sel in (p[n_test + n_val:], p[n_test:n_test + n_val], p[:n_test]):
        m = np.zeros_like(valid)
        m[vidx[sel, 0], vidx[sel, 1]] = True
        masks.append(m)
    years = np.array([int(str(m)[:4]) for m in d["months"]])[:-1]
    uy = np.unique(years)
    ty = set(np.random.RandomState(SPLIT_SEED).choice(uy, size=max(1, int(len(uy) * 0.2)), replace=False).tolist())
    train_years = sorted(set(uy.tolist()) - ty)
    vy = set(np.random.RandomState(SPLIT_SEED + 1000).choice(train_years, size=max(1, int(round(len(train_years) * VAL_BLOCK_FRAC))), replace=False).tolist())
    return dict(valid=valid, block_id=block_id, b1_fold_of_block=np.array([[b, k] for k, f in enumerate(folds) for b in f]),
                b2_region=region, b2_centroids=km.cluster_centers_, lon_grid=lon_grid, lat_grid=lat_grid,
                trackA_train=masks[0], trackA_val=masks[1], trackA_test=masks[2],
                b3_test_years=np.array(sorted(ty)), b3_val_years=np.array(sorted(vy)))


def load_partitions():
    if os.path.exists(C.partitions):
        z = np.load(C.partitions)
        P = {k: z[k] for k in z.files}
        P["_source"] = C.partitions
    else:
        print(f"[partitions] {C.partitions} not found -> rebuilding from {C.cdr_stacks}", flush=True)
        P = rebuild_partitions(C.cdr_stacks)
        np.savez_compressed(os.path.join(C.metrics, "partitions_rebuilt.npz"), **P)
        P["_source"] = "rebuilt from cdr_pinn_monthly_stacks.npz (Model_Outputs/metrics/partitions_rebuilt.npz)"
    return P


def carve_blocks(train_blocks, seed=SPLIT_SEED, frac=VAL_BLOCK_FRAC):
    """Block-carved validation (same rule as the unified CDR runner / build_v2_table.py)."""
    rng = np.random.RandomState(seed + 1000)
    arr = np.array(sorted(train_blocks))
    return sorted(int(x) for x in rng.choice(arr, size=max(1, int(round(len(arr) * frac))), replace=False))


def pixel_partitions(lon, lat, P):
    """Map 1 km pixels onto the CDR-PINO spatial partitions exactly as build_v2_table.py does."""
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    blk = (np.floor((lon - LON_MIN) / 2.0).astype(int) * 1000 + np.floor((lat - LAT_MIN) / 2.0).astype(int)).astype(np.int32)
    fold_of = {int(b): int(k) for b, k in P["b1_fold_of_block"]}
    fold = pd.Series(blk).map(fold_of).fillna(-1).astype(np.int8).to_numpy()
    cen = np.asarray(P["b2_centroids"], np.float64)
    best = np.full(len(lon), np.inf)
    region = np.zeros(len(lon), np.int8)
    for j, (cx, cy) in enumerate(cen):                     # nearest KMeans centroid == KMeans.predict
        dj = (lon - cx) ** 2 + (lat - cy) ** 2
        better = dj < best
        region[better] = j
        best = np.where(better, dj, best)
    b1_val = {k: carve_blocks(np.unique(blk[(fold != k) & (fold >= 0)])) for k in range(3)}
    b2_val = {r: carve_blocks(np.unique(blk[region != r])) for r in range(6)}
    test_years = [int(v) for v in P.get("b3_test_years", FALLBACK_B3_TEST_YEARS)]
    return dict(block=blk, fold=fold, region=region, b1_val_blocks=b1_val, b2_val_blocks=b2_val, b3_test_years=test_years)


def save_partition_report(D, Q, P):
    rep = dict(provenance=provenance(), partitions_source=P["_source"], n_rows=D.n,
               split_counts={pp.SPLIT_NAMES[k]: int((D.split == k).sum()) for k in (0, 1, 2)},
               prevalence=dict(all=float(D.y.mean()), forest=float(D.y[D.forest].mean()), n_forest=int(D.forest.sum())),
               b1_fold_pixels={int(k): int((Q["fold"] == k).sum()) for k in (-1, 0, 1, 2)},
               b2_region_pixels={int(r): int((Q["region"] == r).sum()) for r in range(6)},
               b1_val_blocks=Q["b1_val_blocks"], b2_val_blocks=Q["b2_val_blocks"], b3_test_years=Q["b3_test_years"],
               feature_sets=D.sets, feature_groups=D.groups)
    dump(rep, os.path.join(C.metrics, "PARTITIONS_1km.json"))


# --------------------------------------------------------------------------- models
def fit_rf(Xtr, ytr, hp=None):
    m = RandomForestClassifier(n_jobs=N_JOBS, **(hp or RF_HP))
    t = time.time()
    m.fit(Xtr, ytr)
    return m, time.time() - t


def fit_mx(Xtr, ytr, cfg, n_sub=MX_SUBSAMPLE, seed=42, n_cpus=N_JOBS):
    import elapid
    if n_sub is not None and len(ytr) > n_sub:
        idx, _ = train_test_split(np.arange(len(ytr)), train_size=n_sub, stratify=ytr, random_state=seed)
        Xtr, ytr = Xtr[idx], ytr[idx]
    m = elapid.MaxentModel(n_cpus=n_cpus, random_state=seed, **cfg)
    t = time.time()
    m.fit(Xtr, ytr)
    return m, time.time() - t


def predict(m, X):
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    return m.predict(X)


def evaluate(tag, exp, y_te, p_te, y_va, p_va, pops, extra, save_blocks=None, pred_dir=None):
    """pops: dict name -> boolean mask over the test rows. Threshold metrics use the max-F1
    threshold chosen on VALIDATION (per population); none when there is no validation split."""
    res = {}
    for pn, pm in pops.items():
        yv_mask = extra.get("val_pops", {}).get(pn)
        if y_va is None:
            mv = metrics(y_te[pm], p_te[pm])
        else:
            mv = metrics(y_te[pm], p_te[pm], y_va[yv_mask] if yv_mask is not None else y_va,
                         p_va[yv_mask] if yv_mask is not None else p_va)
        if len(np.unique(y_te[pm])) == 2 and extra.get("ci", True):
            mv.update(boot_ci(y_te[pm], p_te[pm], n=100))
        res[pn] = mv
    suffix = (f"__f{extra['fold']}" if "fold" in extra else "") + (f"__r{extra['region']}" if "region" in extra else "")
    np.savez_compressed(os.path.join(pred_dir or C.pred, f"{exp}__{tag}{suffix}.npz"), y=y_te.astype(np.int8), p=p_te.astype(np.float32),
                        **{f"pop_{k}": v for k, v in pops.items()}, **({"blocks": save_blocks} if save_blocks is not None else {}))
    row = dict(experiment=exp, tag=tag, **{k: v for k, v in extra.items() if k != "val_pops"}, metrics=res)
    log(row)
    print(f"[{exp}] {tag}{suffix}: " + "  ".join(
        f"{pn}: AUC={r.get('roc_auc', float('nan')):.4f} AP={r.get('ap', float('nan')):.4f} prev={r['prevalence']:.4f}"
        for pn, r in res.items()), flush=True)
    return res


def run_model(name, D, tr, va, te, exp, extra=None, blocks=None):
    kind, fkey = MODELS[name]
    cols = D.sets[fkey]
    X = D.X(cols)
    (Xtr, Xva, Xte), med = pp.impute(X[tr], X[va], X[te])
    del X
    if kind == "RF":
        m, tt = fit_rf(Xtr, D.y[tr])
        hp, ntr = RF_HP, int(tr.sum())
    else:
        hp = MX_HIST if kind == "MX_HIST" else MX_BISWAS
        m, tt = fit_mx(Xtr, D.y[tr], hp)
        ntr = min(MX_SUBSAMPLE, int(tr.sum()))
    t1 = time.time()
    p_te, p_va = predict(m, Xte), predict(m, Xva)
    inf = time.time() - t1
    pops = {"all": np.ones(int(te.sum()), bool), "forest": D.forest[te]}
    vpops = {"all": np.ones(int(va.sum()), bool), "forest": D.forest[va]}
    r = evaluate(name, exp, D.y[te], p_te, D.y[va], p_va, pops,
                 dict(model=kind, features=fkey, n_features=len(cols), label="fire_ever", hyperparameters=hp, train_rows=ntr,
                      n_val=int(va.sum()), n_test=int(te.sum()), fit_sec=tt, infer_sec=inf, val_pops=vpops, **(extra or {})),
                 save_blocks=blocks)
    return m, cols, med, r


# --------------------------------------------------------------------------- experiments
def exp_trackA(D, only=None):
    tr, va, te = D.masks()
    for n in (only or TRACKA_MODELS):
        m, cols, med, _ = run_model(n, D, tr, va, te, "trackA", extra=dict(split="v2 random 65/15/20 stratified rs42"))
        if n in IMPORTANCE_MODELS:
            importance(m, n, cols, med, D, te)
        del m


def importance(m, name, cols, med, D, te, n_sample=200000, reps=5):
    """Permutation importance (drop in test ROC-AUC) on a fixed stratified test subsample,
    per feature and per predictor GROUP (group = permute all its columns jointly), plus
    partial dependence (mean model response over 5,000 rows on a 2-98th percentile grid).
    Imputation uses the TRAINING medians of the fitted model."""
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    y = D.y[te]
    X = pp.apply_median(D.X(cols)[te], med)
    if len(y) > n_sample:   # full data: identical to the audit (stratified 200k test subsample, rs 7)
        idx, _ = train_test_split(np.arange(len(y)), train_size=n_sample, stratify=y, random_state=7)
    else:                   # tiny/smoke-test data only
        idx = np.arange(len(y))
    Xs, ys = X[idx], y[idx]
    del X
    base = roc_auc_score(ys, predict(m, Xs))
    rng = np.random.RandomState(0)
    out = {"model": name, "baseline_auc": base, "n_sample": int(len(ys)), "reps": reps, "feature": {}, "group": {}}
    for j, c in enumerate(cols):
        d = []
        for _ in range(reps):
            Xp = Xs.copy()
            Xp[:, j] = Xp[rng.permutation(len(ys)), j]
            d.append(base - roc_auc_score(ys, predict(m, Xp)))
        out["feature"][c] = (float(np.mean(d)), float(np.std(d)))
    groups = D.groups if name.endswith("v2") else {k: [v] for k, v in BISWAS15.items()}
    for gname, gcols in groups.items():
        js = [cols.index(c) for c in gcols if c in cols]
        if not js:
            continue
        d = []
        for _ in range(reps):
            Xp = Xs.copy()
            perm = rng.permutation(len(ys))
            Xp[:, js] = Xp[perm][:, js]
            d.append(base - roc_auc_score(ys, predict(m, Xp)))
        out["group"][gname] = (float(np.mean(d)), float(np.std(d)))
    pdp = {}
    sub = Xs[:5000]
    for j, c in enumerate(cols):
        grid = np.nanpercentile(Xs[:, j], np.linspace(2, 98, 20))
        vals = []
        for gv in grid:
            Xp = sub.copy()
            Xp[:, j] = gv
            vals.append(float(np.mean(predict(m, Xp))))
        pdp[c] = dict(grid=grid.tolist(), mean_response=vals)
    out["pdp"] = pdp
    out["runtime_sec"] = time.time() - t0
    out["provenance"] = provenance()
    dump(out, os.path.join(C.importance, f"importance__trackA__{name}.json"))
    print(f"[importance] {name}: top-5 " + ", ".join(f"{k}={v[0]:.4f}" for k, v in sorted(out['feature'].items(), key=lambda kv: -kv[1][0])[:5]), flush=True)


def _spatial_setup(D):
    P = load_partitions()
    Q = pixel_partitions(D.T["lon"].to_numpy(), D.T["lat"].to_numpy(), P)
    save_partition_report(D, Q, P)
    return P, Q


def exp_B1(D, only=None):
    _, Q = _spatial_setup(D)
    fold, blk = Q["fold"], Q["block"]
    for n in (only or B1_MODELS):
        for k in range(3):
            te = fold == k
            vb = np.array(Q["b1_val_blocks"][k])
            va = (fold >= 0) & (fold != k) & np.isin(blk, vb)
            tr = (fold >= 0) & (fold != k) & ~np.isin(blk, vb)
            run_model(n, D, tr, va, te, "B1", blocks=blk[te],
                      extra=dict(split=f"B1 fold {k} (CDR-PINO 2deg block permutation, block-carved validation)", fold=k))


def exp_B2(D, only=None):
    _, Q = _spatial_setup(D)
    reg, blk = Q["region"], Q["block"]
    for n in (only or B2_MODELS):
        for r in range(6):
            te = reg == r
            vb = np.array(Q["b2_val_blocks"][r])
            va = (reg != r) & np.isin(blk, vb)
            tr = (reg != r) & ~np.isin(blk, vb)
            run_model(n, D, tr, va, te, "B2", blocks=blk[te],
                      extra=dict(split=f"B2 region {r} (CDR-PINO KMeans regions)", region=r))


def year_labels(D, test_years):
    """Pixel burned in held-out years / in the remaining years, from Step 1's fire points with
    containing-pixel (floor) rasterisation on the common grid -- as build_v2_table.py."""
    g = load_ndvi_grid()
    H, W = g["shape"]
    a, _, c, _, e, f = tuple(g["transform"])[:6]
    fp = pd.read_csv(C.fire_points, usecols=["latitude", "longitude", "year"])
    r = np.floor((fp.latitude.to_numpy() - f) / e).astype(np.int64)
    cc = np.floor((fp.longitude.to_numpy() - c) / a).astype(np.int64)
    ok = (r >= 0) & (r < H) & (cc >= 0) & (cc < W)
    gi = r[ok] * W + cc[ok]
    yrs = fp.year.to_numpy()[ok]
    gidx = D.T["grid_index"].to_numpy()
    y_test = np.isin(gidx, np.unique(gi[np.isin(yrs, test_years)])).astype(np.int8)
    y_non = np.isin(gidx, np.unique(gi[~np.isin(yrs, test_years)])).astype(np.int8)
    agree = float(((y_test | y_non) == D.y).mean())
    if agree < 0.999:
        print(f"WARNING: fire points ({C.fire_points}) reproduce fire_ever on only {100 * agree:.3f}% of pixels", flush=True)
    return y_test, y_non, agree


def exp_B3(D):
    """Static-model temporal analogue: fit 'burned in non-test years', score 'burned in test years'
    (CDR-PINO B3 held-out years), same pixels. Plus the persistence null model."""
    P, Q = _spatial_setup(D)
    ty = Q["b3_test_years"]
    yte_all, ytr_all, agree = year_labels(D, ty)
    tr = D.split != 1
    te = np.ones(D.n, bool)
    pops = {"all": np.ones(int(te.sum()), bool), "forest": D.forest[te]}
    evaluate("persistence_null_everburned_trainyears", "B3static", yte_all[te], ytr_all[te].astype(float), None, None, pops,
             dict(model="null", label="y_testyears", test_years=ty, label_agreement_with_fire_ever=agree,
                  note="score = pixel burned in any non-test year"))
    for n in B3_MODELS:
        kind, fkey = MODELS[n]
        cols = D.sets[fkey]
        X = D.X(cols)
        (Xtr, Xte), _ = pp.impute(X[tr], X[te])
        del X
        m, tt = fit_rf(Xtr, ytr_all[tr]) if kind == "RF" else fit_mx(Xtr, ytr_all[tr], MX_HIST if kind == "MX_HIST" else MX_BISWAS)
        p = predict(m, Xte)
        evaluate(n, "B3static", yte_all[te], p, None, None, pops,
                 dict(model=kind, features=fkey, label_train="y_nontestyears", label_test="y_testyears", test_years=ty, fit_sec=tt,
                      note="same pixels in train and test; temporal-only generalisation"))
        del m


def exp_maxent_n(D):
    tr, va, te = D.masks()
    cols = D.sets["v2"]
    X = D.X(cols)
    (Xtr, Xva, Xte), _ = pp.impute(X[tr], X[va], X[te])
    del X
    ytr = D.y[tr]
    plan = [(n, s) for n in (50000, 100000, 150000) for s in (42, 43, 44)] + [(300000, 42), (500000, 42)]
    for n, seed in plan:
        m, tt = fit_mx(Xtr, ytr, MX_HIST, n_sub=n, seed=seed)
        p_va, p_te = predict(m, Xva), predict(m, Xte)
        evaluate(f"MaxEnt_v2_n{n}_seed{seed}", "maxent_n", D.y[te], p_te, D.y[va], p_va,
                 {"all": np.ones(int(te.sum()), bool), "forest": D.forest[te]},
                 dict(model="MaxEnt(elapid)", features="v2", hyperparameters=MX_HIST, train_rows=n, subsample_seed=seed, fit_sec=tt,
                      val_pops={"all": np.ones(int(va.sum()), bool), "forest": D.forest[va]}, ci=False))


# --------------------------------------------------------------------------- bridge (12 km)
def exp_bridge(D):
    """Classical models on CDR-PINO's OWN population (256x256 grid, valid cells), with its
    covariates and the IDENTICAL partitions (Track-A cell split, B1 folds, B2 regions, B3 years),
    separating model/physics effects from predictor-set and population effects (port of
    results/code/bridge_12km.py).
      cdr7  : 5 static CDR covariates + mean/std over months of ndvi_anomaly and dryness
      v2agg : the 55 v2 features averaged from 1 km into each 12 km cell
    Label fire_ever_frac > 0. B3: monthly indicator at t+1 with climatological/seasonal nulls."""
    from rasterio.transform import from_bounds
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    d = np.load(C.cdr_stacks)
    P = load_partitions()
    valid = P["valid"]
    H, W = valid.shape
    NT = d["fire_indicator"].shape[0]
    years = np.array([int(str(m)[:4]) for m in d["months"]])
    months_ = np.array([int(str(m)[5:7]) for m in d["months"]])
    y_ever = (np.nan_to_num(d["fire_ever_frac"]) > 0).astype(int)
    na = np.nan_to_num(d["ndvi_anomaly"])
    dr = np.nan_to_num(d["dryness_proxy"])
    static = {"ndvi_f1": d["ndvi_f1"], "forest_frac": d["forest_frac"], "slope": d["slope"], "dist_roads": d["dist_roads"],
              "elevation": d["elevation"], "ndvi_anom_mean": na.mean(0), "ndvi_anom_std": na.std(0),
              "dryness_mean": dr.mean(0), "dryness_std": dr.std(0)}
    F_cdr7 = np.stack(list(static.values()), -1)
    # v2 features aggregated to 12 km
    feats = D.sets["v2"]
    tr12 = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)
    r12 = np.floor((D.T.lat.to_numpy() - tr12.f) / tr12.e).astype(int)
    c12 = np.floor((D.T.lon.to_numpy() - tr12.c) / tr12.a).astype(int)
    ok = (r12 >= 0) & (r12 < H) & (c12 >= 0) & (c12 < W)
    cell = r12[ok] * W + c12[ok]
    agg = pd.DataFrame(D.T.loc[ok, feats].to_numpy(), columns=feats).groupby(cell).mean()
    F_v2 = np.full((H * W, len(feats)), np.nan, np.float32)
    F_v2[agg.index.to_numpy()] = agg.to_numpy()
    F_v2 = F_v2.reshape(H, W, -1)
    forest = np.nan_to_num(d["forest_frac"]) > 0
    block = P["block_id"]
    fold_of = {int(b): int(k) for b, k in P["b1_fold_of_block"]}
    fold = np.vectorize(lambda b: fold_of.get(int(b), -1))(block)
    region = P["b2_region"]

    def carve(train_mask):
        blocks = np.unique(block[train_mask])
        rng = np.random.RandomState(SPLIT_SEED + 1000)
        vb = rng.choice(blocks, size=max(1, int(round(len(blocks) * VAL_BLOCK_FRAC))), replace=False)
        val = train_mask & np.isin(block, vb)
        return train_mask & ~val, val

    def fit_all(Xtr, ytr):
        out = {}
        m = RandomForestClassifier(n_jobs=N_JOBS, **RF_HP)
        m.fit(Xtr, ytr)
        out["RF"] = m
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(Xtr, ytr)
        out["LogReg"] = lr
        out["MaxEnt"], _ = fit_mx(Xtr, ytr, MX_HIST, n_sub=None)
        return out

    def run_spatial(track, tr, va, te, tag_extra):
        for fname, F in (("cdr7", F_cdr7), ("v2agg", F_v2)):
            X = F.reshape(H * W, -1).astype(np.float32)
            trm, vam, tem = tr.ravel(), va.ravel(), te.ravel()
            (Xtr, Xva, Xte), _ = pp.impute(X[trm], X[vam], X[tem])
            models = fit_all(Xtr, y_ever.ravel()[trm])
            for mn, m in models.items():
                evaluate(f"{mn}_{fname}{tag_extra}", f"bridge12_{track}", y_ever.ravel()[tem], predict(m, Xte),
                         y_ever.ravel()[vam], predict(m, Xva),
                         {"all": np.ones(int(tem.sum()), bool), "forest": forest.ravel()[tem]},
                         dict(model=mn, features=fname, population="CDR-PINO 256x256 valid cells", label="fire_ever_frac>0",
                              partitions=P["_source"], val_pops={"all": np.ones(int(vam.sum()), bool), "forest": forest.ravel()[vam]}),
                         save_blocks=block.ravel()[tem], pred_dir=C.bridge_pred)

    run_spatial("A", valid & P["trackA_train"], valid & P["trackA_val"], valid & P["trackA_test"], "")
    for k in range(3):
        te = valid & (fold == k)
        tr, va = carve(valid & (fold >= 0) & (fold != k))
        run_spatial("B1", tr, va, te, f"_f{k}")
    for r in range(6):
        te = region == r
        tr, va = carve(valid & ~te)
        run_spatial("B2", tr, va, te, f"_r{r}")

    # B3 monthly (identical held-out years; features at month t predict fire at t+1)
    test_years = set(np.asarray(P["b3_test_years"]).tolist())
    val_years = set(np.asarray(P["b3_val_years"]).tolist())
    yrs_pred = years[:-1]
    fit_m = np.array([y not in test_years and y not in val_years for y in yrs_pred])
    val_m = np.array([y in val_years for y in yrs_pred])
    test_m = np.array([y in test_years for y in yrs_pred])
    V = valid.ravel()
    lab = d["fire_indicator"][1:].reshape(NT - 1, -1)[:, V]
    freq_fit = lab[fit_m].mean(0)
    mon_next = months_[1:]
    seas = np.stack([lab[fit_m & (mon_next == m)].mean(0) if (fit_m & (mon_next == m)).any() else freq_fit for m in range(1, 13)])

    def rows(mask, with_month):
        st = F_cdr7.reshape(H * W, -1)[V]
        ti = np.where(mask)[0]
        Xs = [np.repeat(st[None], len(ti), 0),
              na[:-1].reshape(NT - 1, -1)[:, V][ti][..., None], dr[:-1].reshape(NT - 1, -1)[:, V][ti][..., None]]
        if with_month:
            mm = months_[:-1][ti]
            Xs += [np.broadcast_to(np.sin(2 * np.pi * mm / 12)[:, None, None], (len(ti), V.sum(), 1)),
                   np.broadcast_to(np.cos(2 * np.pi * mm / 12)[:, None, None], (len(ti), V.sum(), 1))]
        X = np.concatenate(Xs, -1).reshape(len(ti) * V.sum(), -1).astype(np.float32)
        return np.nan_to_num(X), lab[ti].ravel()

    fr = forest.ravel()[V]
    for name, score in (("null_climatological_frequency", np.tile(freq_fit, test_m.sum())),
                        ("null_seasonal_frequency", np.concatenate([seas[mon_next[t] - 1] for t in np.where(test_m)[0]]))):
        yte = lab[test_m].ravel()
        evaluate(name, "bridge12_B3", yte, score, None, None, {"all": np.ones(len(yte), bool), "forest": np.tile(fr, test_m.sum())},
                 dict(model="null", population="CDR-PINO cells x held-out months", label="monthly fire indicator (t+1)",
                      note="fit-year statistics only; no covariates"), pred_dir=C.bridge_pred)
    for wm in (False, True):
        Xtr, ytr = rows(fit_m, wm)
        Xva, yva = rows(val_m, wm)
        Xte, yte = rows(test_m, wm)
        rf = RandomForestClassifier(n_jobs=N_JOBS, **RF_HP)
        rf.fit(Xtr, ytr)
        evaluate(f"RF_monthly_cdr7{'_month' if wm else ''}", "bridge12_B3", yte, rf.predict_proba(Xte)[:, 1], yva, rf.predict_proba(Xva)[:, 1],
                 {"all": np.ones(len(yte), bool), "forest": np.tile(fr, test_m.sum())},
                 dict(model="RF", features="cdr7 static + monthly ndvi_anomaly_t, dryness_t" + (" + month sin/cos" if wm else ""),
                      label="monthly fire indicator (t+1)", train_rows=len(ytr),
                      val_pops={"all": np.ones(len(yva), bool), "forest": np.tile(fr, val_m.sum())}), pred_dir=C.bridge_pred)
        del rf
    print("[bridge] done", round(time.time() - t0), "s", flush=True)


# --------------------------------------------------------------------------- Biswas-style
BISWAS_LABEL = "BISWAS-STYLE REIMPLEMENTATION with this project's data -- reimplementation, NOT the original Biswas et al. (2025) result"


def exp_biswas(D):
    """Biswas-style baseline (port of results/code/biswas_style_baseline.py): 15 level predictors
    at 0.25 deg (mean of 1 km pixels, cells with >= 50% valid pixels, grid origin 68.0E/37.5N),
    presences = 0.25 deg cells with >= 1 2020 forest fire, background = remaining cells (<= 10,000),
    MaxEnt L+Q+H+P beta 1, 75/25 presence split x 10 seeds; permutation importance (%, drop in
    training AUC), jackknife (only/without) and Fig. 11 correlations. Percent contribution is a
    path-dependent MaxEnt-optimiser quantity not produced by elapid and is NOT computed."""
    import elapid
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    rep = dict(provenance=provenance(), label=BISWAS_LABEL)
    B = BISWAS15
    T = D.T[["lon", "lat"] + list(B.values())].copy()
    r = np.floor((37.5 - T.lat.to_numpy()) / 0.25).astype(int)
    c = np.floor((T.lon.to_numpy() - 68.0) / 0.25).astype(int)
    T["cell"] = r * 1000 + c
    agg = T.groupby("cell")[list(B.values())].mean()
    npx = T.groupby("cell").size()
    full = np.floor(0.25 / (1 / 120)) ** 2
    agg = agg[npx.reindex(agg.index) >= 0.5 * full]
    fp = pd.read_csv(C.fire_points, usecols=["latitude", "longitude", "year"])

    def cells_of(df):
        return (np.floor((37.5 - df.latitude.to_numpy()) / 0.25).astype(int) * 1000 + np.floor((df.longitude.to_numpy() - 68.0) / 0.25).astype(int))

    pres_2020 = pd.Index(np.unique(cells_of(fp[fp.year == 2020]))).intersection(agg.index)
    pres_0120 = pd.Index(np.unique(cells_of(fp[(fp.year >= 2001) & (fp.year <= 2020)]))).intersection(agg.index)
    rep["grid"] = dict(n_cells_valid=int(len(agg)), n_presence_cells_2020=int(len(pres_2020)),
                       n_presence_cells_2001_2020=int(len(pres_0120)), biswas_presence_total=1830 + 609,
                       n_2020_forest_fire_points=int((fp.year == 2020).sum()),
                       ten_percent_of_2020_points=int(round(0.1 * (fp.year == 2020).sum())), fire_points=C.fire_points)
    print(rep["grid"], flush=True)
    X_all = agg.fillna(agg.median())
    names = list(B.keys())
    MX = dict(feature_types=["linear", "quadratic", "hinge", "product"], beta_multiplier=1.0, max_iter=10000, n_cpus=8)

    def run(pres, tag, seeds=range(10)):
        bg = X_all.index.difference(pres)
        out = []
        for s in seeds:
            rng = np.random.RandomState(s)
            pp_ = np.array(pres)
            rng.shuffle(pp_)
            ntr = int(round(0.75 * len(pp_)))
            ptr, pte = pp_[:ntr], pp_[ntr:]
            bgs = bg if len(bg) <= 10000 else pd.Index(rng.choice(bg, 10000, replace=False))
            Xtr = np.r_[X_all.loc[ptr].values, X_all.loc[bgs].values]
            ytr = np.r_[np.ones(len(ptr)), np.zeros(len(bgs))]
            m = elapid.MaxentModel(random_state=s, **MX)
            m.fit(Xtr, ytr)
            p_bg = m.predict(X_all.loc[bgs].values)
            auc_tr = roc_auc_score(ytr, m.predict(Xtr))
            yte = np.r_[np.ones(len(pte)), np.zeros(len(bgs))]
            auc_te = roc_auc_score(yte, np.r_[m.predict(X_all.loc[pte].values), p_bg])
            res = dict(seed=s, n_train_presence=len(ptr), n_test_presence=len(pte), n_background=len(bgs), train_auc=auc_tr, test_auc=auc_te)
            if s == 0:
                rng2 = np.random.RandomState(123)
                drops = {}
                for j, nm in enumerate(names):
                    dd = []
                    for _ in range(5):
                        Xp = Xtr.copy()
                        Xp[:, j] = Xp[rng2.permutation(len(Xp)), j]
                        dd.append(max(auc_tr - roc_auc_score(ytr, m.predict(Xp)), 0.0))
                    drops[nm] = float(np.mean(dd))
                tot = sum(drops.values())
                res["permutation_importance_pct"] = {k: 100 * v / tot for k, v in drops.items()} if tot > 0 else drops
                jk = {}
                Xte_all = np.r_[X_all.loc[pte].values, X_all.loc[bgs].values]
                for j, nm in enumerate(names):
                    keep = [i for i in range(len(names)) if i != j]
                    mo = elapid.MaxentModel(random_state=s, **MX)
                    mo.fit(Xtr[:, [j]], ytr)
                    mw = elapid.MaxentModel(random_state=s, **MX)
                    mw.fit(Xtr[:, keep], ytr)
                    jk[nm] = dict(only_test_auc=float(roc_auc_score(yte, mo.predict(Xte_all[:, [j]]))),
                                  without_test_auc=float(roc_auc_score(yte, mw.predict(Xte_all[:, keep]))))
                res["jackknife"] = jk
                mp = pd.DataFrame({"cell": X_all.index, "prob": m.predict(X_all.values)})
                mp["note"] = "relative MaxEnt output, " + BISWAS_LABEL
                mp.to_csv(os.path.join(C.biswas, f"map_{tag}.csv"), index=False)
            out.append(res)
            print(tag, s, round(auc_tr, 4), round(auc_te, 4), flush=True)
        summ = dict(train_auc_mean=float(np.mean([o["train_auc"] for o in out])), train_auc_sd=float(np.std([o["train_auc"] for o in out])),
                    test_auc_mean=float(np.mean([o["test_auc"] for o in out])), test_auc_sd=float(np.std([o["test_auc"] for o in out])),
                    test_auc_min=float(np.min([o["test_auc"] for o in out])), test_auc_max=float(np.max([o["test_auc"] for o in out])))
        return dict(label=BISWAS_LABEL, runs=out, summary=summ)

    cnt = pd.Series(cells_of(fp[(fp.year >= 2001) & (fp.year <= 2020)])).value_counts()
    dens = cnt.reindex(agg.index).fillna(0)
    rep["fig11_correlations_025deg"] = {nm: float(np.corrcoef(dens.values, X_all[col].values)[0, 1]) for nm, col in B.items()}
    rep["fig11_biswas_reported"] = {"NDVI": 0.43, "Slope": 0.40, "Near-surface wind speed": -0.32,
                                    "Net longwave radiation flux": 0.28, "Soil moisture": 0.27}
    print("Fig11 correlations", {k: round(v, 3) for k, v in rep["fig11_correlations_025deg"].items()}, flush=True)
    rep["replica_2020_presences_025deg"] = run(pres_2020, "2020")
    rep["variant_2001_2020_presences_025deg"] = run(pres_0120, "2001_2020", seeds=range(3))
    rep["biswas_reported"] = dict(train_auc=0.894, test_auc=0.879, n_train_presence=1830, n_test_presence=609, points_total=11360,
                                  permutation_importance_pct={"NDVI": 22.3, "Air temperature": 13.1, "Specific humidity": 13.0,
                                                              "LST night": 10.1, "LST day": 9.6, "Distance to roads": 5.7, "Slope": 5.6,
                                                              "Distance to railways": 4.6, "Soil moisture": 3.8, "Precipitation": 3.6,
                                                              "Near-surface wind speed": 2.4, "Elevation": 2.4, "Net longwave radiation flux": 1.8,
                                                              "Aspect": 1.7, "Distance to waterways": 0.5},
                                  percent_contribution_pct={"NDVI": 28.4, "Air temperature": 3.8, "Specific humidity": 15.0,
                                                            "LST night": 8.9, "LST day": 4.5, "Distance to roads": 2.6, "Slope": 16.7,
                                                            "Distance to railways": 4.9, "Soil moisture": 0.9, "Precipitation": 1.7,
                                                            "Near-surface wind speed": 4.3, "Elevation": 2.0, "Net longwave radiation flux": 0.6,
                                                            "Aspect": 3.8, "Distance to waterways": 1.7},
                                  source="Biswas et al. 2025 Table 3 (p.4870) and p.4865")
    rep["percent_contribution_this_study"] = "NOT COMPUTED: path-dependent MaxEnt-optimiser quantity, not produced by elapid"
    rep["comparability_note"] = ("Presence-vs-background AUC at 0.25 deg; NOT comparable with this project's pixel-level "
                                 "fire/no-fire AUCs. Aspect reconstructed in degrees from aspect_sin/aspect_cos.")
    rep["runtime_sec"] = time.time() - t0
    dump(rep, os.path.join(C.biswas, "BISWAS_STYLE_BASELINE.json"))
    print(rep["replica_2020_presences_025deg"]["summary"], flush=True)


# --------------------------------------------------------------------------- final map
def exp_map(D):
    """RF v2 refit on the v2 TRAINING partition (identical hyperparameters and seed to Track A),
    applied to every valid pixel. Values are CLASS-BALANCED RF SCORES (relative susceptibility),
    NOT calibrated probabilities. 5-class map: quantile breaks over FOREST pixels (20% each)."""
    import rasterio
    tr, _, te = D.masks()
    cols = D.sets["v2"]
    X = D.X(cols)
    (Xtr, Xall), med = pp.impute(X[tr], X)
    del X
    m, tt = fit_rf(Xtr, D.y[tr])
    del Xtr
    t1 = time.time()
    p = m.predict_proba(Xall)[:, 1]
    inf = time.time() - t1
    # Reproducibility evidence: refit must reproduce Track A's saved test predictions
    repro = {}
    fa = os.path.join(C.pred, "trackA__RF_v2.npz")
    if os.path.exists(fa):
        za = np.load(fa)
        dmax = float(np.abs(za["p"].astype(np.float64) - p[te].astype(np.float32).astype(np.float64)).max())
        repro = dict(compared_with=fa, max_abs_diff_test_scores=dmax, identical=bool(dmax == 0.0))
        print(f"[map] refit vs Track-A RF_v2 test scores: max |diff| = {dmax:.3g}", flush=True)
    g = load_ndvi_grid()
    H, W = g["shape"]
    gidx = D.T["grid_index"].to_numpy()
    arr = np.full(H * W, np.nan, np.float32)
    arr[gidx] = p
    q = np.quantile(p[D.forest], [0.2, 0.4, 0.6, 0.8])
    cls = np.full(H * W, np.nan, np.float32)
    cls[gidx] = np.digitize(p, q) + 1
    prov = provenance()
    tags = dict(MODEL="RandomForest v2 (%d features), %s" % (len(cols), json.dumps(RF_HP)),
                LABEL="MODIS C6.1 forest-fire occurrence 2000-11-01..2022-12-15, containing-pixel rasterisation (fire_ever)",
                VALUES="class-balanced RF score in [0,1]; RELATIVE susceptibility score, NOT a calibrated probability",
                PERIOD="2000-11-01 to 2022-12-15 (static, whole-period susceptibility)", CRS="EPSG:4326, 1/120 deg (common NDVI grid)",
                TRAINING="v2 training partition (65%% of pixels, stratified random split rs42), training-row median imputation",
                CREATED=prov["timestamp"], STEP7_COMMIT=prov["step7_repo_commit"], PARQUET_SHA256=prov["parquet_sha256"],
                PROTOCOL="v2, audit of record 2026-09-24/25 (results/FULL_METHODOLOGY_AUDIT.md)")
    outs = (("Susceptibility_RF_v2_score.tif", arr, {}),
            ("Susceptibility_RF_v2_5class_quantile.tif", cls,
             dict(CLASSES="1=Very low..5=Very high; quantile breaks (20/40/60/80%) of the score over FOREST pixels "
                          "(forest_frac_baseline>0), applied to all pixels: " + ", ".join(f"{v:.4f}" for v in q))))
    for name, a, extra in outs:
        with rasterio.open(os.path.join(C.maps, name), "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                           crs=g["crs"], transform=g["transform"], nodata=np.nan, compress="deflate", tiled=True) as dst:
            dst.write(a.reshape(H, W), 1)
            dst.update_tags(**tags, **extra)
    rep = dict(provenance=prov, n_pixels=int(D.n), score_mean_all=float(p.mean()), score_mean_forest=float(p[D.forest].mean()),
               prevalence_all=float(D.y.mean()), prevalence_forest=float(D.y[D.forest].mean()), forest_quantile_breaks=q.tolist(),
               class_pixel_counts={int(k): int((np.digitize(p, q) + 1 == k).sum()) for k in range(1, 6)},
               fit_sec=tt, full_grid_inference_sec=inf, reproducibility=repro, grid_source=g["source"],
               note="Relative susceptibility score (class-balanced RF), not a calibrated probability; see ECE in metrics.")
    dump(rep, os.path.join(C.maps, "Susceptibility_RF_v2_map_report.json"))
    print(f"[map] {rep['n_pixels']:,} px, mean score {rep['score_mean_all']:.4f} (prevalence {rep['prevalence_all']:.4f}), breaks {q.round(4).tolist()}", flush=True)


# --------------------------------------------------------------------------- analysis
def exp_analyze():
    """Compile the registry into metric tables, paired comparisons and figures (port of
    results/code/analyze_classical.py + the classical part of compile_deliverables.py)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows = [json.loads(l) for l in open(C.registry, encoding="utf-8")]
    flat = []
    for r in rows:
        for pop, m in r["metrics"].items():
            t = m.get("at_val_maxF1", {})
            flat.append(dict(experiment=r["experiment"], tag=r["tag"], population=pop, model=r.get("model"), features=r.get("features"),
                             label=r.get("label", r.get("label_test")), split=r.get("split"), fold=r.get("fold"), region=r.get("region"),
                             n=m["n"], n_pos=m["n_pos"], prevalence=m["prevalence"], roc_auc=m.get("roc_auc"), ap=m.get("ap"),
                             auc_ci95=m.get("auc_ci95"), ap_ci95=m.get("ap_ci95"), brier=m.get("brier"), ece=m.get("ece"), logloss=m.get("logloss"),
                             thr_valmaxF1=t.get("threshold"), precision=t.get("precision"), recall_sensitivity=t.get("recall_sensitivity"),
                             specificity=t.get("specificity"), f1=t.get("f1"), train_rows=r.get("train_rows"), fit_sec=r.get("fit_sec"),
                             infer_sec=r.get("infer_sec"), date=r["date"]))
    M = pd.DataFrame(flat)
    M = (M.assign(_f=M.fold.fillna(-1), _r=M.region.fillna(-1))
          .drop_duplicates(["experiment", "tag", "population", "_f", "_r"], keep="last").drop(columns=["_f", "_r"]))
    M.to_csv(os.path.join(C.metrics, "CLASSICAL_METRICS.csv"), index=False)

    # per track summary (mean over folds/regions), FINAL_METRICS-style
    summ = []
    for (exp, tag, pop), s in M.groupby(["experiment", "tag", "population"]):
        summ.append(dict(track=exp, model=tag, population=pop, n_units=len(s), n_test=int(s.n.sum()), prevalence=s.prevalence.mean(),
                         roc_auc=s.roc_auc.mean(), roc_auc_sd_units=s.roc_auc.std(ddof=0) if len(s) > 1 else np.nan,
                         ap=s.ap.mean(), brier=s.brier.mean(), ece=s.ece.mean(), precision=s.precision.mean(),
                         recall_sensitivity=s.recall_sensitivity.mean(), specificity=s.specificity.mean(), f1=s.f1.mean(),
                         threshold_rule="max-F1 on validation" if s.precision.notna().any() else "none (no validation split)",
                         grid="1 km" if not exp.startswith("bridge12") else "12 km (CDR-PINO cells)"))
    FM = pd.DataFrame(summ)
    FM.to_csv(os.path.join(C.metrics, "STEP7_METRICS_BY_TRACK.csv"), index=False)

    # B1/B2 fold summaries
    pool = []
    for exp in ("B1", "B2"):
        for tag in sorted(set(M[M.experiment == exp].tag)):
            sub = M[(M.experiment == exp) & (M.tag == tag)]
            for pop in ("all", "forest"):
                s = sub[sub.population == pop]
                pool.append(dict(experiment=exp, tag=tag, population=pop, n_folds=len(s), auc_mean_over_folds=s.roc_auc.mean(),
                                 auc_sd_over_folds=s.roc_auc.std(ddof=0), auc_min=s.roc_auc.min(), auc_max=s.roc_auc.max(),
                                 ap_mean_over_folds=s.ap.mean(), prevalence_mean=s.prevalence.mean()))
    pd.DataFrame(pool).to_csv(os.path.join(C.metrics, "B1_B2_SUMMARY.csv"), index=False)

    # Track A paired DeLong on identical test pixels
    out = []
    for a, b, why in PAIRS_A:
        fa, fb = os.path.join(C.pred, f"trackA__{a}.npz"), os.path.join(C.pred, f"trackA__{b}.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        za, zb = np.load(fa), np.load(fb)
        assert np.array_equal(za["y"], zb["y"]), (a, b)
        for pop in ("all", "forest"):
            m = za[f"pop_{pop}"]
            r = delong_paired(za["y"][m], za["p"][m], zb["p"][m])
            out.append(dict(model_A=a, model_B=b, question=why, population=pop, n=int(m.sum()), prevalence=float(za["y"][m].mean()),
                            auc_A=r["auc1"], auc_B=r["auc2"], diff=r["diff"], se=r["se"], ci95_lo=r["ci95"][0], ci95_hi=r["ci95"][1],
                            z=r["z"], p=r["p"], ap_A=float(average_precision_score(za["y"][m], za["p"][m])),
                            ap_B=float(average_precision_score(za["y"][m], zb["p"][m]))))
    PA = pd.DataFrame(out)
    PA.to_csv(os.path.join(C.metrics, "PAIRED_trackA.csv"), index=False)

    # B1/B2 pooled folds, block bootstrap over 2-degree blocks
    outB = []
    for exp, nfold, key in (("B1", 3, "f"), ("B2", 6, "r")):
        for a, b in PAIRS_B:
            fa = [os.path.join(C.pred, f"{exp}__{a}__{key}{k}.npz") for k in range(nfold)]
            fb = [os.path.join(C.pred, f"{exp}__{b}__{key}{k}.npz") for k in range(nfold)]
            if not all(os.path.exists(f) for f in fa + fb):
                continue
            Za = [np.load(f) for f in fa]
            Zb = [np.load(f) for f in fb]
            for pop in ("all", "forest"):
                y = np.concatenate([z["y"][z[f"pop_{pop}"]] for z in Za])
                pa = np.concatenate([z["p"][z[f"pop_{pop}"]] for z in Za])
                pb = np.concatenate([z["p"][z[f"pop_{pop}"]] for z in Zb])
                bl = np.concatenate([z["blocks"][z[f"pop_{pop}"]] for z in Za])
                assert np.array_equal(y, np.concatenate([z["y"][z[f"pop_{pop}"]] for z in Zb]))
                sub = np.random.RandomState(0).choice(len(y), min(400000, len(y)), replace=False)
                r = block_boot_diff(y[sub], pa[sub], pb[sub], bl[sub], n=300)
                r["diff"] = float(roc_auc_score(y, pa) - roc_auc_score(y, pb))  # point estimate on all pixels; CI from block bootstrap on a 400k subsample
                per_unit = [float(roc_auc_score(za["y"][za[f"pop_{pop}"]], za["p"][za[f"pop_{pop}"]]) - roc_auc_score(zb["y"][zb[f"pop_{pop}"]], zb["p"][zb[f"pop_{pop}"]]))
                            if len(np.unique(za["y"][za[f"pop_{pop}"]])) == 2 else np.nan for za, zb in zip(Za, Zb)]
                outB.append(dict(experiment=exp, model_A=a, model_B=b, population=pop, n_pixels=len(y), n_blocks=r["n_blocks"],
                                 pooled_auc_A=float(roc_auc_score(y, pa)), pooled_auc_B=float(roc_auc_score(y, pb)),
                                 pooled_diff=r["diff"], blockboot_ci95_lo=r["ci95"][0], blockboot_ci95_hi=r["ci95"][1], blockboot_p=r["p_two_sided"],
                                 per_unit_auc_diff=per_unit, n_units_A_better=int(np.nansum(np.array(per_unit) > 0))))
    pd.DataFrame(outB).to_csv(os.path.join(C.metrics, "PAIRED_B1_B2.csv"), index=False)

    # headline summary for the notebook / README
    def pick(track, model, pop):
        s = FM[(FM.track == track) & (FM.model == model) & (FM.population == pop)]
        return None if not len(s) else {k: (None if pd.isna(s[k].iloc[0]) else float(s[k].iloc[0]))
                                        for k in ("roc_auc", "ap", "prevalence", "ece", "roc_auc_sd_units")}
    head = {f"{tr}|{mo}": {pop: pick(tr, mo, pop) for pop in ("all", "forest")}
            for tr, mo in [("trackA", t) for t in TRACKA_MODELS] + [("B1", t) for t in B1_MODELS] + [("B2", t) for t in B2_MODELS]
            + [("B3static", t) for t in B3_MODELS + ("persistence_null_everburned_trainyears",)]}
    bis = os.path.join(C.biswas, "BISWAS_STYLE_BASELINE.json")
    summary = dict(provenance=provenance(), headline=head,
                   biswas_style=(json.load(open(bis))["replica_2020_presences_025deg"]["summary"] if os.path.exists(bis) else None),
                   biswas_style_label=BISWAS_LABEL,
                   note="Forest pixels (forest_frac_baseline>0) are the PRIMARY population. RF scores are class-balanced, not calibrated.")
    dump(summary, os.path.join(C.metrics, "STEP7_SUMMARY.json"))
    make_figures(M, PA)
    if len(PA):
        print(PA[["model_A", "model_B", "population", "auc_A", "auc_B", "diff", "ci95_lo", "ci95_hi", "p"]].to_string(), flush=True)


def make_figures(M, PA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    # 1) ROC / PR, Track A, both populations
    show = [t for t in ("RF_v2", "RF_biswas15", "MaxEnt_v2", "MaxEnt_biswas15_LQHP", "RF_cdr7static") if os.path.exists(os.path.join(C.pred, f"trackA__{t}.npz"))]
    if show:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        for i, pop in enumerate(("all", "forest")):
            for t in show:
                z = np.load(os.path.join(C.pred, f"trackA__{t}.npz"))
                m = z[f"pop_{pop}"]
                y, p = z["y"][m], z["p"][m]
                fpr, tpr, _ = roc_curve(y, p)
                pr, rc, _ = precision_recall_curve(y, p)
                row = M[(M.experiment == "trackA") & (M.tag == t) & (M.population == pop)]
                auc = row.roc_auc.iloc[0] if len(row) else np.nan
                apv = row.ap.iloc[0] if len(row) else np.nan
                ax[i, 0].plot(fpr, tpr, lw=1.2, label=f"{t} (AUC {auc:.3f})")
                ax[i, 1].plot(rc, pr, lw=1.2, label=f"{t} (AP {apv:.3f})")
            prev = float(y.mean())
            ax[i, 0].plot([0, 1], [0, 1], "k:", lw=0.8)
            ax[i, 1].axhline(prev, color="k", ls=":", lw=0.8, label=f"no skill ({prev:.3f})")
            ax[i, 0].set(title=f"ROC, Track A, {pop} pixels", xlabel="False positive rate", ylabel="True positive rate")
            ax[i, 1].set(title=f"Precision-recall, Track A, {pop} pixels", xlabel="Recall", ylabel="Precision")
            ax[i, 0].legend(fontsize=8)
            ax[i, 1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(C.figures, "TrackA_ROC_PR_all_vs_forest.png"), dpi=150)
        plt.close(fig)
        # 2) reliability diagrams (calibration)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
        for i, pop in enumerate(("all", "forest")):
            for t in show:
                z = np.load(os.path.join(C.pred, f"trackA__{t}.npz"))
                m = z[f"pop_{pop}"]
                y, p = z["y"][m].astype(float), z["p"][m].astype(float)
                edges = np.linspace(0, 1, 11)
                idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
                xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
                ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
                row = M[(M.experiment == "trackA") & (M.tag == t) & (M.population == pop)]
                ax[i].plot(xs, ys, "o-", ms=3, label=f"{t} (ECE {row.ece.iloc[0]:.3f})" if len(row) else t)
            ax[i].plot([0, 1], [0, 1], "k:", lw=0.8)
            ax[i].set(title=f"Reliability, Track A, {pop} pixels", xlabel="Mean score", ylabel="Observed fire frequency")
            ax[i].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(C.figures, "TrackA_reliability.png"), dpi=150)
        plt.close(fig)
    # 3) group permutation importance
    imps = [(n, os.path.join(C.importance, f"importance__trackA__{n}.json")) for n in IMPORTANCE_MODELS]
    imps = [(n, f) for n, f in imps if os.path.exists(f)]
    if imps:
        fig, ax = plt.subplots(1, len(imps), figsize=(5 * len(imps), 5), squeeze=False)
        for a, (n, f) in zip(ax[0], imps):
            g = json.load(open(f))["group"]
            items = sorted(g.items(), key=lambda kv: kv[1][0])
            a.barh([k for k, _ in items], [v[0] for _, v in items], xerr=[v[1] for _, v in items], color="steelblue")
            a.set(title=n, xlabel="Drop in test ROC-AUC (group permutation)")
        fig.tight_layout()
        fig.savefig(os.path.join(C.figures, "TrackA_group_permutation_importance.png"), dpi=150)
        plt.close(fig)
    # 4) MaxEnt sample-size sensitivity
    mx = M[(M.experiment == "maxent_n")].copy()
    if len(mx):
        mx["rows"] = mx.tag.str.extract(r"_n(\d+)_").astype(float)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for pop in ("all", "forest"):
            s = mx[mx.population == pop].groupby("rows").roc_auc.agg(["mean", "std"]).reset_index()
            ax[0].errorbar(s["rows"], s["mean"], yerr=s["std"].fillna(0), marker="o", label=pop)
        t = mx[mx.population == "all"].groupby("rows").fit_sec.mean()
        ax[1].loglog(t.index, t.values, "o-")
        ax[0].set(xlabel="MaxEnt training rows", ylabel="Test ROC-AUC", title="MaxEnt sample-size sensitivity")
        ax[0].legend()
        ax[1].set(xlabel="MaxEnt training rows", ylabel="Fit time (s)", title="Fit time")
        fig.tight_layout()
        fig.savefig(os.path.join(C.figures, "MaxEnt_sample_size_sensitivity.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- CLI
STAGES = ("trackA", "B1", "B2", "B3", "maxent_n", "bridge", "biswas", "map", "analyze")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--only", default=None, help="comma-separated model tags (trackA/B1/B2)")
    for k in ("parquet", "out", "partitions", "fire_points", "cdr_stacks"):
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args(argv)
    configure(args)
    global N_JOBS
    if args.n_jobs:
        N_JOBS = args.n_jobs
    only = args.only.split(",") if args.only else None
    if only:
        bad = [o for o in only if o not in MODELS]
        if bad:
            ap.error(f"unknown model tags {bad}; choose from {list(MODELS)}")
    t0 = time.time()
    if args.stage == "analyze":
        exp_analyze()
    else:
        D = Data(C.parquet)
        {"trackA": lambda: exp_trackA(D, only), "B1": lambda: exp_B1(D, only), "B2": lambda: exp_B2(D, only),
         "B3": lambda: exp_B3(D), "maxent_n": lambda: exp_maxent_n(D), "bridge": lambda: exp_bridge(D),
         "biswas": lambda: exp_biswas(D), "map": lambda: exp_map(D)}[args.stage]()
    print(f"[{args.stage}] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
