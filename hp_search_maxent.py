"""
Validated hyperparameter search for the MaxEnt baseline -- closes the last item
flagged in FULL_EXPERIMENT_LOG.md's rigor audit: MaxEnt has been trained with
elapid's library defaults this entire study, never tuned against held-out data
(unlike the Random Forest, see hp_search_rf.py).

Tunes `beta_multiplier` -- MaxEnt's regularization multiplier, by far the single
most-tuned MaxEnt hyperparameter in the species-distribution-modeling literature
(Merow, Smith & Silander 2013, Ecography 36; Radosavljevic & Anderson 2014, J.
Biogeography 41 -- both establish beta_multiplier/"RM" as the primary complexity
control and recommend searching it over a small grid rather than trusting the
software default). elapid's own default is 1.5; the classic Maxent.jar default is
1.0 -- both are included in the grid below alongside more/less regularized values.

feature_types is deliberately held fixed at ['linear','hinge','product'] (already
matches the notebook's existing choice and Biswas et al.'s own MaxEnt setup) --
searching that jointly with beta_multiplier would multiply the grid size for a
combination the literature treats as a secondary concern once features are
reasonably chosen.

Uses the same 150,000-row stratified training subsample as the notebook's MaxEnt
cells (full-dataset MaxEnt fits are infeasible -- see the notebook's own timing
benchmark), drawn from X_train only. Selection is by VALIDATION AUC
(preprocessing.py's 65/15/20 split); test is not touched by this search at all.
"""
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocessing import load_dataframe, build_train_val_test_split, RANDOM_STATE

OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Integrated_Analysis\Model_Outputs\maxent_hp_search_result.json"
MAXENT_TRAIN_SIZE = 150_000
MAXENT_N_CPUS = 8

BETA_GRID = [0.5, 1.0, 1.5, 2.5, 4.0]  # 1.5=elapid default, 1.0=classic Maxent.jar default


def main():
    import elapid
    from elapid import MaxentModel
    print(f"elapid version: {elapid.__version__}")

    print("Loading data and building 65/15/20 split...")
    df, feature_cols = load_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = build_train_val_test_split(df, feature_cols)
    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}  features={len(feature_cols)}")

    maxent_train_frac = MAXENT_TRAIN_SIZE / len(X_train)
    X_train_maxent, _, y_train_maxent, _ = train_test_split(
        X_train, y_train, train_size=maxent_train_frac, stratify=y_train, random_state=RANDOM_STATE)
    print(f"MaxEnt training subsample: {len(X_train_maxent):,} rows "
          f"({100*y_train_maxent.mean():.2f}% fire)")

    results = []
    t0 = time.time()
    for beta in BETA_GRID:
        model = MaxentModel(
            feature_types=['linear', 'hinge', 'product'],
            beta_multiplier=beta,
            random_state=RANDOM_STATE,
            n_cpus=MAXENT_N_CPUS,
        )
        model.fit(X_train_maxent, y_train_maxent)
        val_score = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_score)
        val_ap = average_precision_score(y_val, val_score)
        elapsed = time.time() - t0
        results.append({"beta_multiplier": beta, "val_auc": float(val_auc), "val_ap": float(val_ap)})
        print(f"  beta_multiplier={beta} -> val_auc={val_auc:.4f}, val_ap={val_ap:.4f}  ({elapsed:.0f}s elapsed)")

    winner = max(results, key=lambda r: r["val_auc"])
    print(f"\nWinner (by validation AUC): beta_multiplier={winner['beta_multiplier']}, val_auc={winner['val_auc']:.4f}")

    print("Refitting winner and evaluating on TEST (touched once, for the winner only)...")
    final_model = MaxentModel(
        feature_types=['linear', 'hinge', 'product'],
        beta_multiplier=winner["beta_multiplier"],
        random_state=RANDOM_STATE,
        n_cpus=MAXENT_N_CPUS,
    )
    final_model.fit(X_train_maxent, y_train_maxent)
    test_score = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_score)
    test_ap = average_precision_score(y_test, test_score)
    print(f"Final TEST (winner, untouched by selection): ROC-AUC={test_auc:.4f}, AP={test_ap:.4f}")

    out = {
        "grid_results": results, "winner_beta_multiplier": winner["beta_multiplier"],
        "winner_val_auc": winner["val_auc"], "test_auc": float(test_auc), "test_ap": float(test_ap),
        "maxent_train_size": MAXENT_TRAIN_SIZE,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
