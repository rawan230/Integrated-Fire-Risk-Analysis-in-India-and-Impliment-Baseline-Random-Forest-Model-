"""
Compile the classical-model registry into metric tables and paired comparisons.

Outputs:
  results/baseline/CLASSICAL_METRICS.csv        one row per (experiment, tag, population)
  results/baseline/PAIRED_trackA.csv            DeLong on identical test pixels
  results/baseline/PAIRED_B1_B2.csv             pooled-fold block bootstrap (2 deg blocks)
Label-variant runs are NOT paired against the primary label (different targets).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import RES  # noqa: E402
from eval_utils import delong_paired, block_boot_diff  # noqa: E402

REG = os.path.join(RES, "audit", "experiment_registry.jsonl")
PB = os.path.join(RES, "baseline", "predictions")
rows = [json.loads(l) for l in open(REG, encoding="utf-8")]
flat = []
for r in rows:
    for pop, m in r["metrics"].items():
        t = m.get("at_val_maxF1", {})
        flat.append(dict(experiment=r["experiment"], tag=r["tag"], population=pop, model=r.get("model"), features=r.get("features"),
                         label=r.get("label", r.get("label_test")), split=r.get("split"), fold=r.get("fold"), region=r.get("region"),
                         n=m["n"], n_pos=m["n_pos"], prevalence=m["prevalence"], roc_auc=m.get("roc_auc"), ap=m.get("ap"),
                         auc_ci95=m.get("auc_ci95"), ap_ci95=m.get("ap_ci95"), brier=m.get("brier"), ece=m.get("ece"), logloss=m.get("logloss"),
                         thr_valmaxF1=t.get("threshold"), precision=t.get("precision"), recall_sensitivity=t.get("recall_sensitivity"),
                         specificity=t.get("specificity"), f1=t.get("f1"), train_rows=r.get("train_rows"), fit_sec=r.get("fit_sec"), date=r["date"]))
M = pd.DataFrame(flat)
M = M.assign(_f=M.fold.fillna(-1), _r=M.region.fillna(-1)).drop_duplicates(["experiment", "tag", "population", "_f", "_r"], keep="last").drop(columns=["_f", "_r"])
M.to_csv(os.path.join(RES, "baseline", "CLASSICAL_METRICS.csv"), index=False)

# pooled B1/B2 summaries (mean over folds, pooled AUC over concatenated test sets)
from sklearn.metrics import roc_auc_score, average_precision_score
pool = []
for exp in ("B1", "B2"):
    tags = sorted(set(M[M.experiment == exp].tag))
    for tag in tags:
        f = os.path.join(PB, f"{exp}__{tag}.npz")
        sub = M[(M.experiment == exp) & (M.tag == tag)]
        for pop in ("all", "forest"):
            s = sub[sub.population == pop]
            pool.append(dict(experiment=exp, tag=tag, population=pop, n_folds=len(s), auc_mean_over_folds=s.roc_auc.mean(),
                             auc_sd_over_folds=s.roc_auc.std(ddof=0), auc_min=s.roc_auc.min(), auc_max=s.roc_auc.max(),
                             ap_mean_over_folds=s.ap.mean(), prevalence_mean=s.prevalence.mean()))
pd.DataFrame(pool).to_csv(os.path.join(RES, "baseline", "B1_B2_SUMMARY.csv"), index=False)


def load(exp, tag):
    z = np.load(os.path.join(PB, f"{exp}__{tag}.npz"))
    return z


pairsA = [("RF_v2", "RF_biswas15", "added predictors / feature engineering"),
          ("RF_v2", "RF_v1_corrlabel", "v2 vs v1 feature definitions (same label)"),
          ("RF_v1_corrlabel", "RF_v1_corrlabel_minus_labelderived", "label-derived NDVI features"),
          ("RF_v2", "RF_v2_minus_landcover", "land-cover group"),
          ("RF_v2", "RF_v2_minus_trends", "trend features (Seasonal Kendall / Sen)"),
          ("RF_v2", "RF_v2_minus_terrain", "terrain group"),
          ("RF_v2", "RF_cdr7static", "CDR-PINO's 5 static covariates vs v2"),
          ("RF_biswas15", "RF_cdr7static", "Biswas-15 vs CDR-PINO's 5 static covariates"),
          ("RF_v2", "MaxEnt_v2", "model family RF vs MaxEnt (v2)"),
          ("RF_biswas15", "MaxEnt_biswas15_LQHP", "model family RF vs MaxEnt (Biswas-15)"),
          ("MaxEnt_v2", "MaxEnt_biswas15_LQHP", "added predictors under MaxEnt"),
          ("RF_v2", "RF_v2_lst005", "LST at 0.05 deg (MOD11C3 resolution proxy)"),
          ("RF_v2", "RF_v2_slopezt", "slope algorithm Horn vs Zevenbergen-Thorne"),
          ("RF_v2", "RF_v2_slope025", "slope from 0.25 deg DEM (Biswas-resolution proxy)")]
out = []
for a, b, why in pairsA:
    fa, fb = os.path.join(PB, f"trackA__{a}.npz"), os.path.join(PB, f"trackA__{b}.npz")
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
pd.DataFrame(out).to_csv(os.path.join(RES, "baseline", "PAIRED_trackA.csv"), index=False)

pairsB = [("RF_v2", "RF_biswas15"), ("RF_v2", "MaxEnt_v2"), ("RF_v2", "RF_cdr7static"), ("RF_biswas15", "MaxEnt_biswas15_LQHP"),
          ("RF_v2", "RF_v1_corrlabel")]
outB = []
for exp, nfold, key in (("B1", 3, "f"), ("B2", 6, "r")):
    for a, b in pairsB:
        fa = [os.path.join(PB, f"{exp}__{a}__{key}{k}.npz") for k in range(nfold)]
        fb = [os.path.join(PB, f"{exp}__{b}__{key}{k}.npz") for k in range(nfold)]
        if not all(os.path.exists(f) for f in fa + fb):
            continue
        Za = [np.load(f) for f in fa]; Zb = [np.load(f) for f in fb]
        for pop in ("all", "forest"):
            y = np.concatenate([z["y"][z[f"pop_{pop}"]] for z in Za]); pa = np.concatenate([z["p"][z[f"pop_{pop}"]] for z in Za])
            pb = np.concatenate([z["p"][z[f"pop_{pop}"]] for z in Zb]); bl = np.concatenate([z["blocks"][z[f"pop_{pop}"]] for z in Za])
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
pd.DataFrame(outB).to_csv(os.path.join(RES, "baseline", "PAIRED_B1_B2.csv"), index=False)
print(pd.DataFrame(out)[["model_A", "model_B", "population", "auc_A", "auc_B", "diff", "ci95_lo", "ci95_hi", "p"]].to_string())
