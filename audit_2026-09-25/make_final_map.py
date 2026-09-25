"""
v2 final susceptibility map: RF_v2 (55 features, corrected label) refit on the v2 training
partition (identical hyperparameters and seed to the Track-A run), applied to every valid pixel.
Output values are CLASS-BALANCED RF SCORES (relative susceptibility), not calibrated
probabilities -- class_weight='balanced' shifts scores upward (see calibration in FINAL_METRICS).
A 5-class map uses quantile breaks computed over forest pixels (20% of forest pixels each), stated in tags.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import RES, ndvi_grid, git_commit, ROOT  # noqa: E402
from models_classical import RF_HP, fit_rf, impute  # noqa: E402

T = pd.read_parquet(os.path.join(RES, "recalculated", "FEATURE_TABLE_v2.parquet"))
S = json.load(open(os.path.join(RES, "recalculated", "FEATURE_SETS.json")))
cols = S["v2_57"]
pop = T.pop_all.values; tr = pop & (T.v2_split.values == 0)
X = T[cols].to_numpy(np.float32)
Xtr, Xall = impute(X[tr], X[pop])
m, _ = fit_rf(Xtr, T.y_all.values[tr])
p = m.predict_proba(Xall)[:, 1]
g = ndvi_grid(); H, W = g["shape"]
arr = np.full(H * W, np.nan, np.float32); arr[T.grid_index.values[pop]] = p
forest_pop = T.v2_forest_frac_2001.values[pop] > 0
q = np.quantile(p[forest_pop], [0.2, 0.4, 0.6, 0.8])   # breaks from FOREST pixels (primary population); ~40% of all pixels score exactly 0
cls = np.full(H * W, np.nan, np.float32); cls[T.grid_index.values[pop]] = np.digitize(p, q) + 1
tags = dict(MODEL="RandomForest v2 (55 features), " + json.dumps(RF_HP), LABEL="MODIS C6.1 forest-fire occurrence 2000-11-01..2022-12-15, containing-pixel rasterisation",
            VALUES="class-balanced RF score in [0,1]; RELATIVE susceptibility, NOT a calibrated probability",
            PERIOD="2000-11-01 to 2022-12-15 (static, whole-period susceptibility)", CRS="EPSG:4326, 1/120 deg",
            CREATED=dt.datetime.now().isoformat(timespec="seconds"), ROOT_COMMIT=git_commit(ROOT), AUDIT="results/final (2026-09-24/25 audit)")
os.makedirs(os.path.join(RES, "final", "maps"), exist_ok=True)
for name, a, extra in (("Susceptibility_RF_v2_score.tif", arr, {}),
                       ("Susceptibility_RF_v2_5class_quantile.tif", cls, dict(CLASSES="1=Very low..5=Very high; quantile breaks (20/40/60/80%) of the score over FOREST pixels (forest_frac_2001>0), applied to all pixels: " + ", ".join(f"{v:.4f}" for v in q)))):
    with rasterio.open(os.path.join(RES, "final", "maps", name), "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs=g["crs"], transform=g["transform"], nodata=np.nan, compress="deflate", tiled=True) as d:
        d.write(a.reshape(H, W), 1); d.update_tags(**tags, **extra)
print(dict(n=int(pop.sum()), score_mean=float(p.mean()), breaks=q.tolist()))
