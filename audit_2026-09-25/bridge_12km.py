"""
Bridge experiment: classical models on CDR-PINO's OWN population (256x256 grid, 22,542 valid
cells), with CDR-PINO's covariates and the IDENTICAL partitions (Track A pixel split, B1 block
folds, B2 KMeans regions, B3 held-out years), so model/physics effects can be separated from
predictor-set and population effects.

Feature sets (12 km):
  cdr7   : the 5 static CDR covariates + mean/std over months of the 2 monthly covariates
           (ndvi_anomaly, dryness) -- a static summary of exactly what CDR-PINO sees
  v2agg  : the 57 v2 features averaged from 1 km into each 12 km cell
Label: fire_ever_frac > 0 (the CDR-PINO evaluation label). B3: monthly indicator, t+1.
Null models for B3: training-period fire frequency of the cell (climatological persistence)
and same-calendar-month frequency (seasonal persistence).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import RES, ROOT  # noqa: E402
from models_classical import evaluate, impute, fit_mx, predict, MX_HIST, RF_HP, N_JOBS  # noqa: E402

OUT = os.path.join(RES, "baseline")
NPZ = os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "CDR_PINN_Data", "cdr_pinn_monthly_stacks.npz")
PART = os.path.join(RES, "cdr_pino", "unified", "partitions.npz")
t0 = time.time()
d = np.load(NPZ); pt = np.load(PART)
valid = pt["valid"]; H, W = valid.shape
T = d["fire_indicator"].shape[0]
years = np.array([int(m[:4]) for m in d["months"]]); months_ = np.array([int(m[5:7]) for m in d["months"]])
y_ever = (np.nan_to_num(d["fire_ever_frac"]) > 0).astype(int)
na = np.nan_to_num(d["ndvi_anomaly"]); dr = np.nan_to_num(d["dryness_proxy"])
static = {"ndvi_f1": d["ndvi_f1"], "forest_frac": d["forest_frac"], "slope": d["slope"], "dist_roads": d["dist_roads"],
          "elevation": d["elevation"], "ndvi_anom_mean": na.mean(0), "ndvi_anom_std": na.std(0),
          "dryness_mean": dr.mean(0), "dryness_std": dr.std(0)}
F_cdr7 = np.stack([np.nan_to_num(v, nan=np.nan) for v in static.values()], -1)  # (H,W,9)

# v2 features aggregated to 12 km
Tv = pd.read_parquet(os.path.join(RES, "recalculated", "FEATURE_TABLE_v2.parquet"))
Sv = json.load(open(os.path.join(RES, "recalculated", "FEATURE_SETS.json")))
from rasterio.transform import from_bounds
tr12 = from_bounds(68.20, 6.75, 97.40, 37.09, W, H)
r12 = np.floor((Tv.lat.values - tr12.f) / tr12.e).astype(int); c12 = np.floor((Tv.lon.values - tr12.c) / tr12.a).astype(int)
ok = (r12 >= 0) & (r12 < H) & (c12 >= 0) & (c12 < W) & Tv.pop_all.values
cell = r12[ok] * W + c12[ok]
agg = pd.DataFrame(Tv.loc[ok, Sv["v2_57"]].values, columns=Sv["v2_57"]).groupby(cell).mean()
F_v2 = np.full((H * W, len(Sv["v2_57"])), np.nan, np.float32); F_v2[agg.index.values] = agg.values
F_v2 = F_v2.reshape(H, W, -1)
del Tv
forest = np.nan_to_num(d["forest_frac"]) > 0
block = pt["block_id"]
fold_of = {int(b): int(k) for b, k in pt["b1_fold_of_block"]}
fold = np.vectorize(lambda b: fold_of.get(int(b), -1))(block)
region = pt["b2_region"]


def carve(train_mask, frac=0.1875, seed=42):
    blocks = np.unique(block[train_mask])
    rng = np.random.RandomState(seed + 1000)
    vb = rng.choice(blocks, size=max(1, int(round(len(blocks) * frac))), replace=False)
    val = train_mask & np.isin(block, vb)
    return train_mask & ~val, val


def fit_all(Xtr, ytr):
    out = {}
    m = RandomForestClassifier(n_jobs=N_JOBS, **RF_HP); m.fit(Xtr, ytr); out["RF"] = m
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")); lr.fit(Xtr, ytr); out["LogReg"] = lr
    mx, _ = fit_mx(Xtr, ytr, MX_HIST, n_sub=None); out["MaxEnt"] = mx
    return out


def run_spatial(track, tr, va, te, tag_extra):
    for fname, F in (("cdr7", F_cdr7), ("v2agg", F_v2)):
        X = F.reshape(H * W, -1).astype(np.float32)
        trm, vam, tem = tr.ravel(), va.ravel(), te.ravel()
        Xtr, Xva, Xte = impute(X[trm], X[vam], X[tem])
        models = fit_all(Xtr, y_ever.ravel()[trm])
        for mn, m in models.items():
            evaluate(f"{mn}_{fname}{tag_extra}", f"bridge12_{track}", y_ever.ravel()[tem], predict(m, Xte),
                     y_ever.ravel()[vam], predict(m, Xva),
                     {"all": np.ones(tem.sum(), bool), "forest": forest.ravel()[tem]},
                     dict(model=mn, features=fname, population="CDR-PINO 256x256 valid cells", label="fire_ever_frac>0",
                          val_pops={"all": np.ones(vam.sum(), bool), "forest": forest.ravel()[vam]}),
                     save_blocks=block.ravel()[tem])


# Track A (identical pixel split)
run_spatial("A", valid & pt["trackA_train"], valid & pt["trackA_val"], valid & pt["trackA_test"], "")
# B1 (identical folds, block-carved validation)
for k in range(3):
    te = valid & (fold == k); tr_all = valid & (fold >= 0) & (fold != k)
    tr, va = carve(tr_all)
    run_spatial("B1", tr, va, te, f"_f{k}")
# B2 (identical regions)
for r in range(6):
    te = region == r; tr_all = valid & ~te
    tr, va = carve(tr_all)
    run_spatial("B2", tr, va, te, f"_r{r}")

# B3 monthly (identical held-out years; features at month t predict fire at t+1)
test_years = set(pt["b3_test_years"].tolist()); val_years = set(pt["b3_val_years"].tolist())
yrs_pred = years[:-1]
fit_m = np.array([y not in test_years and y not in val_years for y in yrs_pred])
val_m = np.array([y in val_years for y in yrs_pred]); test_m = np.array([y in test_years for y in yrs_pred])
V = valid.ravel()
lab = d["fire_indicator"][1:].reshape(T - 1, -1)[:, V]            # (T-1, n_cells)
freq_fit = lab[fit_m].mean(0)                                        # climatological persistence
mon_next = months_[1:]
seas = np.stack([lab[fit_m & (mon_next == m)].mean(0) if (fit_m & (mon_next == m)).any() else freq_fit for m in range(1, 13)])


def rows(mask, with_month):
    st = F_cdr7.reshape(H * W, -1)[V]
    ti = np.where(mask)[0]
    Xs = [np.repeat(st[None], len(ti), 0),
          na[:-1].reshape(T - 1, -1)[:, V][ti][..., None], dr[:-1].reshape(T - 1, -1)[:, V][ti][..., None]]
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
                  note="fit-year statistics only; no covariates"))
for wm in (False, True):
    Xtr, ytr = rows(fit_m, wm); Xva, yva = rows(val_m, wm); Xte, yte = rows(test_m, wm)
    rf = RandomForestClassifier(n_jobs=N_JOBS, **RF_HP); rf.fit(Xtr, ytr)
    evaluate(f"RF_monthly_cdr7{'_month' if wm else ''}", "bridge12_B3", yte, rf.predict_proba(Xte)[:, 1], yva, rf.predict_proba(Xva)[:, 1],
             {"all": np.ones(len(yte), bool), "forest": np.tile(fr, test_m.sum())},
             dict(model="RF", features="cdr7 static + monthly ndvi_anomaly_t, dryness_t" + (" + month sin/cos" if wm else ""),
                  label="monthly fire indicator (t+1)", train_rows=len(ytr),
                  val_pops={"all": np.ones(len(yva), bool), "forest": np.tile(fr, val_m.sum())}))
print("bridge done", round(time.time() - t0), "s")
