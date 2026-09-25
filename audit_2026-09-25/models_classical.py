"""
Classical-model experiments (Random Forest, MaxEnt) for the 2026-09-24 audit, on the 1 km
pixel universe. Every fit appends one row to results/audit/experiment_registry.jsonl and
saves its test predictions (for DeLong / block-bootstrap paired comparisons).

  python models_classical.py repro      -- exact historical Step 7 (v1, shifted label, 80/20)
  python models_classical.py trackA     -- random 65/15/20 (v2 split), both populations
  python models_classical.py B1         -- 3-fold 2deg-block CV, CDR-PINO fold geometry
  python models_classical.py B1hist     -- historical GroupKFold(3) reproduction (v1)
  python models_classical.py B2         -- leave-one-region-out, CDR-PINO KMeans regions
  python models_classical.py B3         -- static temporal analogue + persistence null
  python models_classical.py maxent_n   -- MaxEnt training-sample-size sensitivity
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import RES, ROOT  # noqa: E402
from eval_utils import metrics, boot_ci  # noqa: E402

REC = os.path.join(RES, "recalculated")
OUT = os.path.join(RES, "baseline")
PRED = os.path.join(OUT, "predictions")
os.makedirs(PRED, exist_ok=True)
REG = os.path.join(RES, "audit", "experiment_registry.jsonl")
os.makedirs(os.path.dirname(REG), exist_ok=True)
N_JOBS = 20
RF_HP = dict(n_estimators=200, max_depth=25, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", random_state=42)
MX_HIST = dict(feature_types=["linear", "hinge", "product"], beta_multiplier=4.0)        # Step 7 validated
MX_BISWAS = dict(feature_types=["linear", "quadratic", "hinge", "product"], beta_multiplier=1.0)  # MaxEnt 3.4.x 'auto' (>=80 presences), default regularisation


def commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "n/a"


def log(row):
    row = dict(row); row["date"] = dt.datetime.now().isoformat(timespec="seconds"); row["root_commit"] = commit()
    with open(REG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def load():
    T = pd.read_parquet(os.path.join(REC, "FEATURE_TABLE_v2.parquet"))
    S = json.load(open(os.path.join(REC, "FEATURE_SETS.json")))
    return T, S


def impute(Xtr, *others):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda X: np.where(np.isfinite(X), X, med)
    return [fill(Xtr)] + [fill(X) for X in others]


def fit_rf(Xtr, ytr, hp=None):
    m = RandomForestClassifier(n_jobs=N_JOBS, **(hp or RF_HP))
    t = time.time(); m.fit(Xtr, ytr); return m, time.time() - t


def fit_mx(Xtr, ytr, cfg, n_sub=150000, seed=42):
    import elapid
    if n_sub is not None and len(ytr) > n_sub:
        idx, _ = train_test_split(np.arange(len(ytr)), train_size=n_sub, stratify=ytr, random_state=seed)
        Xtr, ytr = Xtr[idx], ytr[idx]
    m = elapid.MaxentModel(n_cpus=N_JOBS, random_state=seed, **cfg)
    t = time.time(); m.fit(Xtr, ytr); return m, time.time() - t


def predict(m, X):
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    return m.predict(X)


def evaluate(tag, exp, y_te, p_te, y_va, p_va, pops, extra, save_blocks=None):
    """pops: dict name -> boolean mask over the test rows."""
    res = {}
    for pn, pm in pops.items():
        yv_mask = extra.get("val_pops", {}).get(pn)
        if y_va is None:
            mv = metrics(y_te[pm], p_te[pm])
        else:
            mv = metrics(y_te[pm], p_te[pm], y_va[yv_mask] if yv_mask is not None else y_va, p_va[yv_mask] if yv_mask is not None else p_va)
        if len(np.unique(y_te[pm])) == 2 and extra.get("ci", True):
            mv.update(boot_ci(y_te[pm], p_te[pm], n=100))
        res[pn] = mv
    suffix = (f"__f{extra['fold']}" if "fold" in extra else "") + (f"__r{extra['region']}" if "region" in extra else "")
    np.savez_compressed(os.path.join(PRED, f"{exp}__{tag}{suffix}.npz"), y=y_te.astype(np.int8), p=p_te.astype(np.float32),
                        **{f"pop_{k}": v for k, v in pops.items()}, **({"blocks": save_blocks} if save_blocks is not None else {}))
    row = dict(experiment=exp, tag=tag, **{k: v for k, v in extra.items() if k != "val_pops"}, metrics=res)
    log(row)
    print(f"[{exp}] {tag}: " + "  ".join(f"{pn}: AUC={r.get('roc_auc', float('nan')):.4f} AP={r.get('ap', float('nan')):.4f} prev={r['prevalence']:.4f}" for pn, r in res.items()), flush=True)
    return res


# --------------------------------------------------------------------------- experiments
def exp_repro(T, S):
    """Exact historical Step 7: v1 features, historical (shifted) label, median fill on ALL rows pre-split, 80/20."""
    cols = S["v1_57"]
    X = T[cols].to_numpy(np.float32); y = T["y_v1_hist_shifted"].to_numpy()
    med = np.nanmedian(X, axis=0); X = np.where(np.isfinite(X), X, med)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    m, tt = fit_rf(Xtr, ytr)
    p = predict(m, Xte)
    evaluate("RF_v1_histlabel_80_20", "repro", yte, p, None, None, {"all": np.ones(len(yte), bool)},
             dict(model="RF", features="v1_57", n_features=len(cols), label="y_v1_hist_shifted", split="80/20 stratified rs42",
                  hyperparameters=RF_HP, train_rows=len(ytr), fit_sec=tt, historical=dict(roc_auc=0.9704471015873686, ap=0.7011235367097762), ci=False))
    mx, tt = fit_mx(Xtr, ytr, MX_HIST)
    p = predict(mx, Xte)
    evaluate("MaxEnt_v1_histlabel_80_20_150k", "repro", yte, p, None, None, {"all": np.ones(len(yte), bool)},
             dict(model="MaxEnt(elapid)", features="v1_57", label="y_v1_hist_shifted", split="80/20 stratified rs42",
                  hyperparameters=MX_HIST, train_rows=150000, fit_sec=tt, historical=dict(roc_auc=0.9598071502830996, ap=0.6275017379150017), ci=False))


def split_masks(T):
    pop = T["pop_all"].to_numpy()
    sp = T["v2_split"].to_numpy()
    return pop & (sp == 0), pop & (sp == 1), pop & (sp == 2)


MODELS = {
    # name: (model, feature set key, label column)
    "RF_v2": ("RF", "v2_57", "y_all"),
    "RF_v1_corrlabel": ("RF", "v1_57", "y_all"),
    "RF_v1_corrlabel_minus_labelderived": ("RF", "v1_minus_labelderived", "y_all"),
    "RF_v1_histlabel": ("RF", "v1_57", "y_v1_hist_shifted"),
    "RF_biswas15": ("RF", "biswas15", "y_all"),
    "RF_cdr7static": ("RF", "cdr7_static_1km", "y_all"),
    "RF_v2_minus_landcover": ("RF", "v2_minus_landcover", "y_all"),
    "RF_v2_minus_trends": ("RF", "v2_minus_trends", "y_all"),
    "RF_v2_minus_terrain": ("RF", "v2_minus_terrain", "y_all"),
    "RF_v2_label_conf30": ("RF", "v2_57", "y_conf_ge30"),
    "RF_v2_label_type0": ("RF", "v2_57", "y_type0"),
    "RF_v2_label_conf30_type0": ("RF", "v2_57", "y_conf_ge30_type0"),
    "MaxEnt_v2": ("MX_HIST", "v2_57", "y_all"),
    "MaxEnt_biswas15_LQHP": ("MX_BISWAS", "biswas15", "y_all"),
    "MaxEnt_v1_corrlabel": ("MX_HIST", "v1_57", "y_all"),
}


def feats_of(S, key, T):
    if key == "v2_57_lst005":
        return [c.replace("v2_lst_day_level", "sens_lst_day_level_005deg").replace("v2_lst_night_level", "sens_lst_night_level_005deg") for c in S["v2_57"]]
    if key == "v2_57_slopezt":
        return [("sens_slope_zt" if c == "v2_slope" else c) for c in S["v2_57"]]
    if key == "v2_57_slope025":
        return [("sens_slope_from_025deg_dem" if c == "v2_slope" else c) for c in S["v2_57"]]
    return S[key]


def run_model(name, T, S, tr, va, te, exp, extra=None, subset=None, blocks=None):
    kind, fkey, ycol = MODELS[name] if name in MODELS else subset
    cols = feats_of(S, fkey, T)
    X = T[cols].to_numpy(np.float32); y = T[ycol].to_numpy()
    Xtr, Xva, Xte = impute(X[tr], X[va], X[te])
    if kind == "RF":
        m, tt = fit_rf(Xtr, y[tr]); hp = RF_HP; ntr = int(tr.sum())
    else:
        cfg = MX_HIST if kind == "MX_HIST" else MX_BISWAS
        m, tt = fit_mx(Xtr, y[tr], cfg); hp = cfg; ntr = 150000
    t1 = time.time(); p_te = predict(m, Xte); p_va = predict(m, Xva); inf = time.time() - t1
    forest = T["v2_forest_frac_2001"].to_numpy() > 0
    pops = {"all": np.ones(te.sum(), bool), "forest": forest[te]}
    vpops = {"all": np.ones(va.sum(), bool), "forest": forest[va]}
    r = evaluate(name, exp, y[te], p_te, y[va], p_va, pops,
                 dict(model=kind, features=fkey, n_features=len(cols), label=ycol, hyperparameters=hp, train_rows=ntr,
                      n_val=int(va.sum()), n_test=int(te.sum()), fit_sec=tt, infer_sec=inf, val_pops=vpops, **(extra or {})),
                 save_blocks=blocks)
    return m, cols, r


def exp_trackA(T, S, which=None):
    tr, va, te = split_masks(T)
    names = which or list(MODELS) + ["RF_v2_lst005", "RF_v2_slopezt", "RF_v2_slope025"]
    for n in names:
        sub = {"RF_v2_lst005": ("RF", "v2_57_lst005", "y_all"), "RF_v2_slopezt": ("RF", "v2_57_slopezt", "y_all"),
               "RF_v2_slope025": ("RF", "v2_57_slope025", "y_all")}.get(n)
        m, cols, _ = run_model(n, T, S, tr, va, te, "trackA", extra=dict(split="v2 random 65/15/20 stratified rs42"), subset=sub)
        if n in ("RF_v2", "MaxEnt_v2", "RF_biswas15", "MaxEnt_biswas15_LQHP"):
            importance(m, n, cols, T, S, te)


def importance(m, name, cols, T, S, te, n_sample=200000, reps=5):
    """Permutation importance (drop in test ROC-AUC) on a fixed stratified test subsample, per
    feature and per predictor GROUP (group permutation = permute all group columns jointly)."""
    from sklearn.metrics import roc_auc_score
    y = T["y_all"].to_numpy()[te]
    X = T[cols].to_numpy(np.float32)[te]
    X = impute(X)[0]
    idx, _ = train_test_split(np.arange(len(y)), train_size=min(n_sample, len(y) - 1), stratify=y, random_state=7)
    Xs, ys = X[idx], y[idx]
    base = roc_auc_score(ys, predict(m, Xs))
    rng = np.random.RandomState(0)
    out = {"baseline_auc": base, "feature": {}, "group": {}}
    for j, c in enumerate(cols):
        d = []
        for _ in range(reps):
            Xp = Xs.copy(); Xp[:, j] = Xp[rng.permutation(len(ys)), j]
            d.append(base - roc_auc_score(ys, predict(m, Xp)))
        out["feature"][c] = (float(np.mean(d)), float(np.std(d)))
    groups = S["v2_groups"] if name.endswith("v2") else {k: [v] for k, v in S["biswas15_map"].items()}
    for gname, gcols in groups.items():
        js = [cols.index(c) for c in gcols if c in cols]
        if not js:
            continue
        d = []
        for _ in range(reps):
            Xp = Xs.copy(); perm = rng.permutation(len(ys)); Xp[:, js] = Xp[perm][:, js]
            d.append(base - roc_auc_score(ys, predict(m, Xp)))
        out["group"][gname] = (float(np.mean(d)), float(np.std(d)))
    # partial dependence (model response) for every feature, averaged over 5,000 rows
    pdp = {}
    sub = Xs[:5000]
    for j, c in enumerate(cols):
        grid = np.nanpercentile(Xs[:, j], np.linspace(2, 98, 20))
        vals = []
        for gv in grid:
            Xp = sub.copy(); Xp[:, j] = gv
            vals.append(float(np.mean(predict(m, Xp))))
        pdp[c] = dict(grid=grid.tolist(), mean_response=vals)
    out["pdp"] = pdp
    with open(os.path.join(OUT, f"importance__trackA__{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[importance] {name}: top-5 " + ", ".join(f"{k}={v[0]:.4f}" for k, v in sorted(out['feature'].items(), key=lambda kv: -kv[1][0])[:5]), flush=True)


def exp_B1(T, S, names=("RF_v2", "RF_biswas15", "RF_v1_corrlabel", "RF_cdr7static", "MaxEnt_v2", "MaxEnt_biswas15_LQHP")):
    meta = json.load(open(os.path.join(REC, "FEATURE_TABLE_v2_report.json")))["partitions_meta"]["b1_val_blocks"]
    pop = T["pop_all"].to_numpy(); fold = T["b1_fold"].to_numpy(); blk = T["block_2deg"].to_numpy()
    for n in names:
        for k in range(3):
            te = pop & (fold == k)
            vb = np.array(meta[str(k)])
            va = pop & (fold >= 0) & (fold != k) & np.isin(blk, vb)
            tr = pop & (fold >= 0) & (fold != k) & ~np.isin(blk, vb)
            run_model(n, T, S, tr, va, te, "B1", extra=dict(split=f"B1 fold {k} (CDR-PINO 2deg block permutation, block-carved validation)", fold=k),
                      blocks=blk[te])


def exp_B1hist(T, S):
    """Historical Step 7 spatial CV: GroupKFold(3) on 2deg blocks, per-fold median imputation, v1 features, historical label."""
    from sklearn.model_selection import GroupKFold
    cols = S["v1_57"]
    X = T[cols].to_numpy(np.float32); y = T["y_v1_hist_shifted"].to_numpy()
    lon, lat = T.lon.to_numpy(), T.lat.to_numpy()
    # historical Step 7 definition: blocks anchored at 0 deg (floor(lon/2), floor(lat/2)), string ids
    groups = pd.Series(np.floor(lon / 2).astype(int).astype(str)) + "_" + pd.Series(np.floor(lat / 2).astype(int).astype(str))
    groups = groups.values
    for k, (tri, tei) in enumerate(GroupKFold(n_splits=3).split(X, y, groups)):
        Xtr, Xte = impute(X[tri], X[tei])
        for kind in (("RF",) if os.environ.get("B1HIST_RF_ONLY") == "1" else ("RF", "MX_HIST")):
            if kind == "RF":
                m, tt = fit_rf(Xtr, y[tri])
            else:
                m, tt = fit_mx(Xtr, y[tri], MX_HIST)
            p = predict(m, Xte)
            evaluate(f"{kind}_v1_histlabel_fold{k}", "B1hist_v2origin", y[tei], p, None, None, {"all": np.ones(len(tei), bool)},
                     dict(model=kind, features="v1_57", label="y_v1_hist_shifted", split=f"GroupKFold(3) fold {k} (historical)", fold=k,
                          fit_sec=tt, ci=False))


def exp_B2(T, S, names=("RF_v2", "RF_biswas15", "RF_cdr7static", "MaxEnt_v2", "MaxEnt_biswas15_LQHP")):
    meta = json.load(open(os.path.join(REC, "FEATURE_TABLE_v2_report.json")))["partitions_meta"]["b2_val_blocks"]
    pop = T["pop_all"].to_numpy(); reg = T["b2_region"].to_numpy(); blk = T["block_2deg"].to_numpy()
    for n in names:
        for r in range(6):
            te = pop & (reg == r)
            vb = np.array(meta[str(r)])
            va = pop & (reg != r) & np.isin(blk, vb)
            tr = pop & (reg != r) & ~np.isin(blk, vb)
            run_model(n, T, S, tr, va, te, "B2", extra=dict(split=f"B2 region {r} (CDR-PINO KMeans regions)", region=r), blocks=blk[te])


def exp_B3(T, S):
    """Static-model temporal analogue: fit on 'burned in non-test years', score 'burned in test years'
    (2000, 2008, 2009, 2015 -- CDR-PINO B3 years), same pixels. Includes the PERSISTENCE null model."""
    pop = T["pop_all"].to_numpy(); sp = T["v2_split"].to_numpy()
    tr, va = pop & (sp != 1), pop & (sp == 1)
    ytr = T["y_nontestyears"].to_numpy(); yte = T["y_testyears"].to_numpy()
    forest = T["v2_forest_frac_2001"].to_numpy() > 0
    te = pop
    pers = ytr[te].astype(float)  # binary persistence: burned in any non-test year
    for tag, p in (("persistence_null_everburned_trainyears", pers),):
        evaluate(tag, "B3static", yte[te], p, None, None, {"all": np.ones(te.sum(), bool), "forest": forest[te]},
                 dict(model="null", label="y_testyears", note="score = pixel burned in any non-test year"))
    for n in ("RF_v2", "RF_biswas15", "MaxEnt_v2"):
        kind, fkey, _ = MODELS[n]
        cols = S[fkey]; X = T[cols].to_numpy(np.float32)
        Xtr, Xte = impute(X[tr], X[te])
        m, tt = fit_rf(Xtr, ytr[tr]) if kind == "RF" else fit_mx(Xtr, ytr[tr], MX_HIST)
        p = predict(m, Xte)
        evaluate(n, "B3static", yte[te], p, None, None, {"all": np.ones(te.sum(), bool), "forest": forest[te]},
                 dict(model=kind, features=fkey, label_train="y_nontestyears", label_test="y_testyears", fit_sec=tt,
                      note="same pixels in train and test; temporal-only generalisation"))


def exp_maxent_n(T, S):
    tr, va, te = split_masks(T)
    cols = S["v2_57"]; X = T[cols].to_numpy(np.float32); y = T["y_all"].to_numpy()
    Xtr, Xva, Xte = impute(X[tr], X[va], X[te])
    ytr = y[tr]
    plan = [(50000, s) for s in (42, 43, 44)] + [(100000, s) for s in (42, 43, 44)] + [(150000, s) for s in (42, 43, 44)] + [(300000, 42), (500000, 42)]
    for n, seed in plan:
        m, tt = fit_mx(Xtr, ytr, MX_HIST, n_sub=n, seed=seed)
        p_va, p_te = predict(m, Xva), predict(m, Xte)
        forest = T["v2_forest_frac_2001"].to_numpy() > 0
        evaluate(f"MaxEnt_v2_n{n}_seed{seed}", "maxent_n", y[te], p_te, y[va], p_va, {"all": np.ones(te.sum(), bool), "forest": forest[te]},
                 dict(model="MaxEnt(elapid)", features="v2_57", hyperparameters=MX_HIST, train_rows=n, subsample_seed=seed, fit_sec=tt,
                      val_pops={"all": np.ones(va.sum(), bool), "forest": forest[va]}, ci=False))


if __name__ == "__main__":
    which = sys.argv[1]
    T, S = load()
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    {"repro": lambda: exp_repro(T, S), "trackA": lambda: exp_trackA(T, S, only), "B1": lambda: exp_B1(T, S, *( [only] if only else [])),
     "B1hist": lambda: exp_B1hist(T, S), "B2": lambda: exp_B2(T, S, *([only] if only else [])), "B3": lambda: exp_B3(T, S),
     "maxent_n": lambda: exp_maxent_n(T, S)}[which]()
