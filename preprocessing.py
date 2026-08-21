"""
Step 7 preprocessing -- the single, standard source of truth for how Step 6's
integrated pixel table becomes model-ready train/validation/test splits. Factored out
of the notebook so every training/evaluation/hyperparameter-selection cell shares one
split definition instead of each re-implementing (and risking silent drift in) its own.

Standard protocol (adopted 2026-08-21, replacing the old train/test-only convention):
  - Genuine 65/15/20% train/validation/test split, stratified by the fire_ever label,
    same RANDOM_STATE=42 this notebook has always used for reproducibility.
  - Validation is used for any hyperparameter decision (e.g. comparing RF max_depth/
    n_estimators candidates); test is touched exactly once, at the end, for the final
    reported number -- mirroring the protocol just adopted for CDR-PINN
    (Physics_Informed_FireRisk_Model/cdr_pinn/preprocessing.py).
  - Reads Step 6's post-2026-08-21 corrected parquet (forest_frac_recent/current/loss
    dropped as a data-leakage fix -- only forest_frac_baseline, 2001, survives). This
    module doesn't special-case that; it just reads whatever columns the current
    parquet actually has, same as the notebook's own dynamic feature_cols pattern.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PARQUET_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Integrated_Analysis\Integrated_Outputs\Integrated_FireRisk_Pixels.parquet"
RANDOM_STATE = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.20
DROP_COLS = ["lon", "lat", "fire_count", "fire_ever"]
TARGET_COL = "fire_ever"


def load_dataframe():
    df = pd.read_parquet(PARQUET_PATH)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    # median-fill on the whole X pre-split, matching this notebook's own established
    # convention (Step 7's existing cells do this before any split) -- not repeated
    # per-split, to stay directly comparable with the notebook's existing random-split
    # numbers, which use the same pre-split fill.
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    return df, feature_cols


def build_train_val_test_split(df, feature_cols, random_state=RANDOM_STATE,
                                val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """Two-stage stratified split: first carve off test_frac, then split the remainder
    into train/val so the final proportions are exactly train/val/test =
    (1-val_frac-test_frac)/val_frac/test_frac of the full dataset."""
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=random_state)

    relative_val_frac = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val_frac, stratify=y_trainval,
        random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df, feature_cols = load_dataframe()
    print(f"Loaded {len(df):,} rows, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    X_train, X_val, X_test, y_train, y_val, y_test = build_train_val_test_split(df, feature_cols)
    n = len(df)
    print(f"train={len(X_train):,} ({len(X_train)/n:.3f})  "
          f"val={len(X_val):,} ({len(X_val)/n:.3f})  "
          f"test={len(X_test):,} ({len(X_test)/n:.3f})")
    print(f"positive rate -- train={y_train.mean():.4f}  val={y_val.mean():.4f}  test={y_test.mean():.4f}")
    for bad in ["forest_frac_recent", "forest_frac_current", "forest_loss_baseline_to_recent"]:
        print(f"  {bad} present: {bad in feature_cols}")
    print(f"  forest_frac_baseline present: {'forest_frac_baseline' in feature_cols}")
