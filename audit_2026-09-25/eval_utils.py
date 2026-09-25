"""
Evaluation statistics for the audit.

metrics()        : ROC-AUC, AP, prevalence, Brier, log-loss, ECE(10 bins), and threshold
                   metrics (precision / recall=sensitivity / specificity / F1) at thresholds
                   chosen on VALIDATION data only (max-F1 and Youden-J); never tuned on test.
boot_ci()        : stratified bootstrap percentile CIs for AUC and AP.
delong_paired()  : DeLong et al. (1988) test for two correlated ROC AUCs on the SAME cases,
                   fast O(n log n) midrank algorithm of Sun & Xu (2014).
block_boot_diff(): cluster (block) bootstrap CI for a paired AUC difference when test cases
                   are spatially clustered (resamples whole blocks).
"""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def _thr_maxf1(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(th) else 0
    return float(th[i]) if len(th) else 0.5


def _thr_youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return float(min(th[i], 1.0))


def _at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return dict(threshold=t, precision=prec, recall_sensitivity=rec, specificity=spec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), tp=tp, fp=fp, fn=fn, tn=tn)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0; rel = []
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
            rel.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return float(e), rel


def metrics(y_test, p_test, y_val=None, p_val=None):
    y = np.asarray(y_test).astype(int); p = np.clip(np.asarray(p_test, float), 1e-7, 1 - 1e-7)
    out = dict(n=int(len(y)), n_pos=int(y.sum()), prevalence=float(y.mean()))
    if len(np.unique(y)) < 2:
        return out
    out.update(roc_auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
               brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
    out["ece"], out["reliability"] = ece(y, p)
    if y_val is not None and len(np.unique(y_val)) == 2:
        yv = np.asarray(y_val).astype(int); pv = np.asarray(p_val, float)
        out["at_val_maxF1"] = _at(y, p, _thr_maxf1(yv, pv))
        out["at_val_youden"] = _at(y, p, _thr_youden(yv, pv))
    out["at_0.5"] = _at(y, p, 0.5)
    return out


def boot_ci(y, p, n=200, seed=0):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    rng = np.random.RandomState(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    a, b = [], []
    for _ in range(n):
        s = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        a.append(roc_auc_score(y[s], p[s])); b.append(average_precision_score(y[s], p[s]))
    return dict(auc_ci95=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                ap_ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def _midrank_structural(p, y):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tz = rankdata(np.r_[pos, neg])
    tx, ty = rankdata(pos), rankdata(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n          # structural components for positives
    v10 = 1.0 - (tz[m:] - ty) / m    # for negatives
    return auc, v01, v10


def delong_paired(y, p1, p2):
    y = np.asarray(y).astype(int)
    a1, v01_1, v10_1 = _midrank_structural(np.asarray(p1, float), y)
    a2, v01_2, v10_2 = _midrank_structural(np.asarray(p2, float), y)
    m, n = len(v01_1), len(v10_1)
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(max(var, 1e-300))
    se = np.sqrt(max(var, 0))
    return dict(auc1=float(a1), auc2=float(a2), diff=float(a1 - a2), se=float(se),
                ci95=[float(a1 - a2 - 1.96 * se), float(a1 - a2 + 1.96 * se)], z=float(z), p=float(2 * norm.sf(abs(z))))


def block_boot_diff(y, p1, p2, blocks, n=500, seed=0, metric="auc"):
    y = np.asarray(y).astype(int); blocks = np.asarray(blocks)
    ub = np.unique(blocks); idx_of = {b: np.where(blocks == b)[0] for b in ub}
    f = roc_auc_score if metric == "auc" else average_precision_score
    rng = np.random.RandomState(seed); d = []
    for _ in range(n):
        s = np.concatenate([idx_of[b] for b in rng.choice(ub, len(ub))])
        if len(np.unique(y[s])) < 2:
            continue
        d.append(f(y[s], p1[s]) - f(y[s], p2[s]))
    d = np.array(d)
    return dict(diff=float(f(y, p1) - f(y, p2)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), n_blocks=int(len(ub)), n_boot=int(len(d)))
