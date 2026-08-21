"""
Validated hyperparameter search for the Random Forest fire-susceptibility model --
closes a real gap: this model's hyperparameters (n_estimators=200, max_depth=20,
min_samples_leaf=5) were literature defaults, never tuned against held-out data. This
script searches a small, real grid, selects by VALIDATION AUC (preprocessing.py's
65/15/20 split), and reports test AUC once, for the winner only.

Small grid, not an exhaustive search -- max_depth and min_samples_leaf are the two
knobs that actually control overfitting for a tree ensemble (n_estimators mainly
affects variance/stability, not bias, past a few hundred trees, so it's held fixed at
the existing 200 -- literature-standard reasoning, not a shortcut).
"""
import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocessing import load_dataframe, build_train_val_test_split, RANDOM_STATE

OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Integrated_Analysis\Model_Outputs\rf_hp_search_result.json"

GRID = [
    dict(max_depth=20, min_samples_leaf=5),   # current default, kept as a baseline candidate
    dict(max_depth=15, min_samples_leaf=5),   # shallower -- less overfitting risk
    dict(max_depth=25, min_samples_leaf=3),   # deeper, less-restrictive leaves
    dict(max_depth=20, min_samples_leaf=10),  # same depth, more conservative leaves
]


def main():
    print("Loading data and building 65/15/20 split...")
    df, feature_cols = load_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = build_train_val_test_split(df, feature_cols)
    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  features={len(feature_cols)}")

    results = []
    t0 = time.time()
    for params in GRID:
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, **params)
        model.fit(X_train, y_train)
        val_score = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_score)
        val_ap = average_precision_score(y_val, val_score)
        results.append({"params": params, "val_auc": float(val_auc), "val_ap": float(val_ap)})
        print(f"  {params} -> val_auc={val_auc:.4f}, val_ap={val_ap:.4f}  ({time.time()-t0:.0f}s elapsed)")

    winner = max(results, key=lambda r: r["val_auc"])
    print(f"\nWinner (by validation AUC): {winner['params']}, val_auc={winner['val_auc']:.4f}")

    print("Refitting winner and evaluating on TEST (touched once, for the winner only)...")
    final_model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, **winner["params"])
    final_model.fit(X_train, y_train)
    test_score = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_score)
    test_ap = average_precision_score(y_test, test_score)
    print(f"Final TEST (winner, untouched by selection): ROC-AUC={test_auc:.4f}, AP={test_ap:.4f}")

    importances = sorted(zip(feature_cols, final_model.feature_importances_), key=lambda kv: -kv[1])
    print("\nTop-10 Gini importance (tuned model):")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")

    out = {
        "grid_results": results, "winner_params": winner["params"], "winner_val_auc": winner["val_auc"],
        "test_auc": float(test_auc), "test_ap": float(test_ap),
        "top_importances": [(n, float(i)) for n, i in importances[:15]],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
