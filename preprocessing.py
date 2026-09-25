"""
Step 7 preprocessing -- the single source of truth for how Step 6's integrated pixel table
becomes model-ready partitions. Updated 2026-09-25 to the corrected ("v2") evaluation
protocol of the end-to-end audit of record (results/FULL_METHODOLOGY_AUDIT.md,
Integrated_Analysis/AUDIT_2026-09-25.md).

v2 protocol (reproduces the audit exactly):
  - Input: Step 6's v2 `Integrated_FireRisk_Pixels.parquet`. Rows = India-mask pixels with a
    finite NDVI mean, ROW-MAJOR by `grid_index`. Columns: lon, lat, grid_index, fire_count,
    fire_ever + 55 features. Feature list = every column except ID_COLS + LABEL_COLS.
  - Split: two-stage stratified random split on row position (vid = arange(n)):
        test  = train_test_split(vid, test_size=0.20, stratify=fire_ever, random_state=42)
        val   = train_test_split(rest, test_size=0.15/0.80, stratify=fire_ever[rest], random_state=42)
    -> 65/15/20 train/val/test. This is byte-for-byte the construction the audit used for its
    `v2_split` (results/code/recalc_04_ndvi.py), so partitions are identical when the row order
    (row-major India pixels with finite NDVI) is identical.
  - Median imputation is fitted on TRAINING rows only (v1 filled NaNs with medians over all
    rows before splitting -- a small train/test leak the audit removed). A median that is itself
    non-finite (all-NaN column in training) falls back to 0.0, as in the audit.
  - Populations: all pixels, and forest pixels (forest_frac_baseline > 0, i.e. the 2001 forest
    fraction). Forest is the PRIMARY evaluation population: forest fraction alone reaches
    AUC ~0.91 over all pixels, so all-pixel AUCs are dominated by the forest/non-forest contrast.

The legacy helpers `load_dataframe()` / `build_train_val_test_split()` are kept (same names,
same return signature) so hp_search_rf.py / hp_search_maxent.py keep working, but they now
use the v2 split and training-only imputation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(HERE, "Integrated_Outputs", "Integrated_FireRisk_Pixels.parquet")
RANDOM_STATE = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.20
ID_COLS = ["lon", "lat", "grid_index"]
LABEL_COLS = ["fire_count", "fire_ever"]
DROP_COLS = ID_COLS + LABEL_COLS
TARGET_COL = "fire_ever"
FOREST_COL = "forest_frac_baseline"
N_EXPECTED_FEATURES = 55

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def feature_columns(df):
    """Every column except identifiers and labels (dynamic, as Step 7 has always been)."""
    return [c for c in df.columns if c not in DROP_COLS]


def label_of(df):
    return (df[TARGET_COL].to_numpy() > 0).astype(np.int8)


def forest_mask(df):
    return np.nan_to_num(df[FOREST_COL].to_numpy(np.float64), nan=0.0) > 0


def v2_split(y, random_state=RANDOM_STATE):
    """Return an int8 array (0=train, 1=val, 2=test) over rows, built exactly as the audit's
    `v2_split` (two-stage stratified train_test_split on row positions)."""
    y = np.asarray(y).astype(int)
    vid = np.arange(len(y))
    trv, te = train_test_split(vid, test_size=0.20, stratify=y, random_state=random_state)
    tr, va = train_test_split(trv, test_size=0.15 / 0.80, stratify=y[trv], random_state=random_state)
    split = np.full(len(y), -1, np.int8)
    split[tr] = 0
    split[va] = 1
    split[te] = 2
    assert (split >= 0).all()
    return split


def fit_median(X_train):
    """Column medians over TRAINING rows only; non-finite medians fall back to 0.0."""
    med = np.nanmedian(np.asarray(X_train, dtype=np.float32), axis=0)
    return np.where(np.isfinite(med), med, 0.0).astype(np.float32)


def apply_median(X, med):
    X = np.asarray(X, dtype=np.float32)
    return np.where(np.isfinite(X), X, med).astype(np.float32)


def impute(X_train, *others):
    """Fit on X_train, apply to X_train and every other array. Returns ([filled...], med)."""
    med = fit_median(X_train)
    return [apply_median(X_train, med)] + [apply_median(X, med) for X in others], med


# --------------------------------------------------------------------------- legacy API
def load_dataframe(path=PARQUET_PATH):
    """Load the pixel table. NO imputation here any more (v2: imputation is fitted on the
    training partition only, inside build_train_val_test_split)."""
    df = pd.read_parquet(path)
    return df, feature_columns(df)


def build_train_val_test_split(df, feature_cols, random_state=RANDOM_STATE,
                               val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """v2 65/15/20 split with training-only median imputation. Returns pandas objects:
    X_train, X_val, X_test, y_train, y_val, y_test (index = original row positions)."""
    if (val_frac, test_frac) != (VAL_FRAC, TEST_FRAC):
        raise ValueError("The v2 protocol fixes val/test fractions at 0.15/0.20.")
    y = label_of(df)
    split = v2_split(y, random_state)
    X = df[feature_cols].to_numpy(np.float32)
    (Xtr, Xva, Xte), _ = impute(X[split == 0], X[split == 1], X[split == 2])
    out = []
    for part, Xp in ((0, Xtr), (1, Xva), (2, Xte)):
        idx = np.flatnonzero(split == part)
        out.append(pd.DataFrame(Xp, columns=feature_cols, index=df.index[idx]))
    ys = [pd.Series(y[split == p].astype(int), index=df.index[np.flatnonzero(split == p)], name=TARGET_COL) for p in (0, 1, 2)]
    return out[0], out[1], out[2], ys[0], ys[1], ys[2]


if __name__ == "__main__":
    df, feature_cols = load_dataframe()
    print(f"Loaded {len(df):,} rows, {len(feature_cols)} features (expected {N_EXPECTED_FEATURES})")
    missing = [c for c in ID_COLS + LABEL_COLS + [FOREST_COL] if c not in df.columns]
    if missing:
        print(f"  WARNING: missing v2 columns {missing} -- is this still the v1 parquet?")
    y = label_of(df)
    split = v2_split(y)
    fm = forest_mask(df) if FOREST_COL in df.columns else np.zeros(len(df), bool)
    for p, name in SPLIT_NAMES.items():
        m = split == p
        print(f"  {name:5s}: {m.sum():>9,} ({m.mean():.3f})  prevalence all={y[m].mean():.4f}"
              f"  forest={y[m & fm].mean() if (m & fm).any() else float('nan'):.4f}  (forest rows {int((m & fm).sum()):,})")
