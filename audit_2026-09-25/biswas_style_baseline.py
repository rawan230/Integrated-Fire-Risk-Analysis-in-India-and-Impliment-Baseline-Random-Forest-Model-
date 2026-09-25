"""
BISWAS-STYLE BASELINE (a re-implementation with THIS project's verified data -- NOT the
original Biswas et al. 2025 result, whose data are not available here).

Reference protocol (Biswas et al. 2025, ESPR 32:4856-4878, pp. 4861-4865, 4870-4871):
  predictors  : 15 variables (Table 3), all rasterised to 0.25 x 0.25 deg
  model       : MaxEnt 3.4.4, max 10,000 iterations; presence/background
  presences   : 2020 forest-fire records ("ten percent of the total number of forest fires
                in 2020 ... 11,360 points ... 1830 presence records used for training, and
                609 for testing"); 70% train stated in text, 1830/2439 = 75.0% in numbers
  evaluation  : train / test AUC 0.894 / 0.879 (presence vs background)
  importance  : permutation importance (%), percent contribution (%), jackknife
Choices made here where the paper is underspecified (all disclosed): grid origin 68.0E/37.5N;
predictor value = mean of the 1 km v2 level features inside each 0.25 deg cell (cells with
>= 50% valid India pixels); presences = unique 0.25 deg cells containing >= 1 2020 forest
fire (tested below against their 2,439); background = all remaining India cells or 10,000
random cells if more; MaxEnt 3.4 'auto' feature classes for >= 80 presences (L,Q,H,P) and
regularisation multiplier 1; 75/25 presence split repeated over 10 seeds.
Percent contribution is path-dependent in MaxEnt's own optimiser and is not produced by
elapid -> NOT computed (reported as such).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import RES, dump, provenance  # noqa: E402

OUT = os.path.join(RES, "biswas_reference")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
rep = dict(provenance=provenance(__file__), label="BISWAS-STYLE REIMPLEMENTATION (not the original Biswas et al. result)")
T = pd.read_parquet(os.path.join(RES, "recalculated", "FEATURE_TABLE_v2.parquet"))
S = json.load(open(os.path.join(RES, "recalculated", "FEATURE_SETS.json")))
B = S["biswas15_map"]
T = T[T.pop_all]
r = np.floor((37.5 - T.lat.values) / 0.25).astype(int); c = np.floor((T.lon.values - 68.0) / 0.25).astype(int)
T["cell"] = r * 1000 + c
agg = T.groupby("cell")[list(B.values())].mean()
npx = T.groupby("cell").size()
full = np.floor(0.25 / (1 / 120)) ** 2                      # 1 km pixels per full 0.25 deg cell (900)
agg = agg[npx.reindex(agg.index) >= 0.5 * full]
fp = pd.read_parquet(os.path.join(RES, "recalculated", "R1_fire_labels", "forest_fire_points_recalc.parquet"),
                     columns=["latitude", "longitude", "year"])
def cells_of(df):
    return (np.floor((37.5 - df.latitude.values) / 0.25).astype(int) * 1000 + np.floor((df.longitude.values - 68.0) / 0.25).astype(int))
pres_2020 = pd.Index(np.unique(cells_of(fp[fp.year == 2020]))).intersection(agg.index)
pres_0120 = pd.Index(np.unique(cells_of(fp[(fp.year >= 2001) & (fp.year <= 2020)]))).intersection(agg.index)
rep["grid"] = dict(n_cells_valid=int(len(agg)), n_presence_cells_2020=int(len(pres_2020)),
                   n_presence_cells_2001_2020=int(len(pres_0120)), biswas_presence_total=1830 + 609,
                   n_2020_forest_fire_points=int((fp.year == 2020).sum()),
                   ten_percent_of_2020_points=int(round(0.1 * (fp.year == 2020).sum())))
print(rep["grid"], flush=True)
import elapid
X_all = agg.fillna(agg.median())
cols = list(B.values()); names = list(B.keys())


def run(pres, tag, seeds=range(10)):
    bg = X_all.index.difference(pres)
    out = []
    for s in seeds:
        rng = np.random.RandomState(s)
        pp = np.array(pres); rng.shuffle(pp)
        ntr = int(round(0.75 * len(pp)))
        ptr, pte = pp[:ntr], pp[ntr:]
        bgs = bg if len(bg) <= 10000 else pd.Index(rng.choice(bg, 10000, replace=False))
        Xtr = np.r_[X_all.loc[ptr].values, X_all.loc[bgs].values]; ytr = np.r_[np.ones(len(ptr)), np.zeros(len(bgs))]
        m = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"], beta_multiplier=1.0,
                               max_iter=10000, n_cpus=8, random_state=s)
        m.fit(Xtr, ytr)
        p_bg = m.predict(X_all.loc[bgs].values)
        auc_tr = roc_auc_score(ytr, m.predict(Xtr))
        yte = np.r_[np.ones(len(pte)), np.zeros(len(bgs))]
        auc_te = roc_auc_score(yte, np.r_[m.predict(X_all.loc[pte].values), p_bg])
        res = dict(seed=s, n_train_presence=len(ptr), n_test_presence=len(pte), n_background=len(bgs), train_auc=auc_tr, test_auc=auc_te)
        if s == 0:
            # MaxEnt-style permutation importance: shuffle among training points, drop in TRAINING AUC, normalised to 100%
            rng2 = np.random.RandomState(123); drops = {}
            for j, nm in enumerate(names):
                dd = []
                for _ in range(5):
                    Xp = Xtr.copy(); Xp[:, j] = Xp[rng2.permutation(len(Xp)), j]
                    dd.append(max(auc_tr - roc_auc_score(ytr, m.predict(Xp)), 0.0))
                drops[nm] = float(np.mean(dd))
            tot = sum(drops.values())
            res["permutation_importance_pct"] = {k: 100 * v / tot for k, v in drops.items()} if tot > 0 else drops
            # jackknife: test AUC with only / without each variable
            jk = {}
            for j, nm in enumerate(names):
                keep = [i for i in range(len(names)) if i != j]
                mo = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"], beta_multiplier=1.0, max_iter=10000, n_cpus=8, random_state=s)
                mo.fit(Xtr[:, [j]], ytr)
                mw = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"], beta_multiplier=1.0, max_iter=10000, n_cpus=8, random_state=s)
                mw.fit(Xtr[:, keep], ytr)
                Xte_all = np.r_[X_all.loc[pte].values, X_all.loc[bgs].values]
                jk[nm] = dict(only_test_auc=float(roc_auc_score(yte, mo.predict(Xte_all[:, [j]]))),
                              without_test_auc=float(roc_auc_score(yte, mw.predict(Xte_all[:, keep]))))
            res["jackknife"] = jk
            # map of relative probability for all cells
            pd.DataFrame({"cell": X_all.index, "prob": m.predict(X_all.values)}).to_csv(os.path.join(OUT, f"map_{tag}.csv"), index=False)
        out.append(res)
        print(tag, s, round(auc_tr, 4), round(auc_te, 4), flush=True)
    summ = dict(train_auc_mean=float(np.mean([o["train_auc"] for o in out])), train_auc_sd=float(np.std([o["train_auc"] for o in out])),
                test_auc_mean=float(np.mean([o["test_auc"] for o in out])), test_auc_sd=float(np.std([o["test_auc"] for o in out])),
                test_auc_min=float(np.min([o["test_auc"] for o in out])), test_auc_max=float(np.max([o["test_auc"] for o in out])))
    return dict(runs=out, summary=summ)


# Biswas Fig. 11: Pearson correlation between forest-fire point DENSITY (2001-2020, per 0.25 deg cell)
# and each predictor. Reported: NDVI 0.43, slope 0.40, wind -0.32, net LW 0.28, soil moisture 0.27.
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
                              source="Biswas et al. 2025 Table 3 (p.4870) and p.4865; verified from the PDF by text extraction")
rep["percent_contribution_this_study"] = "NOT COMPUTED: path-dependent MaxEnt-optimiser quantity, not produced by elapid"
rep["runtime_sec"] = time.time() - t0
dump(rep, os.path.join(OUT, "BISWAS_STYLE_BASELINE.json"))
print(rep["replica_2020_presences_025deg"]["summary"])
