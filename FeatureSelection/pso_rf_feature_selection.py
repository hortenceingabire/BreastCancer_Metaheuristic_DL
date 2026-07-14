"""
===========================================================================
PSO-Based Feature Selection with Random Forest
===========================================================================

Associated Manuscript
---------------------
Impact of Metaheuristic-Based Feature Selection on the Performance of
Deep Learning Models for Breast Cancer Classification

Authors
-------
Hortence Ingabire
Lúcia Valéria Ramos de Arruda

Journal
-------
Scientific Reports (under review)

Description
-----------
This script implements wrapper-based feature selection using Binary Particle
Swarm Optimization (PSO) with Random Forest as the fitness function.

The selected feature subset is evaluated using 5-fold stratified
cross-validation on the Breast Cancer Wisconsin Diagnostic (BCWD) dataset.

The implementation reproduces the experiments reported in the associated
manuscript, including:

• Binary PSO feature selection
• Random Forest fitness evaluation
• 5-fold stratified cross-validation
• Stable feature subset identification
• Performance metrics
• ROC curve
• Aggregated confusion matrix
• Convergence curves
• Feature selection frequency analysis

Software
--------
Python 3.10

License
-------
MIT License

===========================================================================
"""

# =========================================
# IMPORTS
# =========================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import MinMaxScaler

# =========================================================================
# SOFTWARE ENVIRONMENT
# =========================================================================
#
# Recommended environment:
#
# Python 3.10
#
# Package dependencies and tested versions are listed in:
#
#     requirements.txt
#
# =========================================================================

# =========================================
# REPRODUCIBILITY
# =========================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# =========================================
# PSO PARAMETERS
# =========================================
N_PARTICLES  = 50
N_ITERATIONS = 100
W            = 0.72    # inertia weight
C1           = 1.49    # cognitive (personal best) coefficient
C2           = 1.49    # social (global best) coefficient
V_MAX        = 6.0     # velocity clamping — prevents sigmoid saturation

# =========================================
# CV SETTINGS
# =========================================
N_FOLDS = 5

# =========================================
# SAVE DIRECTORY
# =========================================
results_dir = "PSO_RF_5FOLD_RESULTS"
os.makedirs(results_dir, exist_ok=True)
figures_dir = os.path.join(results_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# =========================================================================
# DATASET
# =========================================================================
#
# Breast Cancer Wisconsin Diagnostic (BCWD)
#
# This implementation uses the built-in scikit-learn version of the
# Breast Cancer Wisconsin Diagnostic dataset:
#
# from sklearn.datasets import load_breast_cancer
#
# The dataset originates from the UCI Machine Learning Repository.
#
# =========================================================================
data          = load_breast_cancer()
X             = data.data
y             = data.target
feature_names = np.array(data.feature_names)
n_features    = X.shape[1]

print(f"Dataset loaded: {X.shape[0]} samples, {n_features} features")
print(f"Class distribution — Malignant (0): {np.sum(y==0)}, Benign (1): {np.sum(y==1)}")

# =========================================
# PSO FEATURE SELECTION FUNCTION
# Runs inside a single fold
# =========================================
def run_pso_on_fold(X_train, y_train, X_val, y_val, fold_seed):
    """
    Runs Binary PSO feature selection on one fold.
    Fitness is evaluated on the validation portion of the fold.
    Scaler must be fitted on X_train only before passing — no leakage.

    Parameters
    ----------
    X_train   : array, training features (already scaled)
    y_train   : array, training labels
    X_val     : array, validation features (already scaled)
    y_val     : array, validation labels
    fold_seed : int, random seed for this fold

    Returns
    -------
    gbest_position : binary array of selected features
    gbest_score    : best validation F1 achieved
    convergence    : list of best score per iteration
    """

    np.random.seed(fold_seed)

    # ---------------------------
    # FITNESS FUNCTION
    # Weighted F1 on validation set
    # ---------------------------
    def fitness(position):
        selected = np.where(position == 1)[0]
        if len(selected) == 0:
            return 0.0

        clf = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=fold_seed
        )
        clf.fit(X_train[:, selected], y_train)
        pred = clf.predict(X_val[:, selected])

        return f1_score(y_val, pred, average='weighted', zero_division=0)

    # ---------------------------
    # INITIALIZATION
    # ---------------------------
    # Binary positions: each particle is a candidate feature subset
    positions  = np.random.randint(2, size=(N_PARTICLES, n_features))
    # Continuous velocities initialized in [-V_MAX/2, V_MAX/2]
    velocities = np.random.uniform(-V_MAX / 2, V_MAX / 2,
                                   size=(N_PARTICLES, n_features))

    # Evaluate initial fitness for all particles
    pbest_positions = positions.copy()
    pbest_scores    = np.array([fitness(p) for p in pbest_positions])

    # Global best
    gbest_idx      = np.argmax(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score    = pbest_scores[gbest_idx]

    # FIX: convergence curve storage
    convergence = []

    # ---------------------------
    # PSO MAIN LOOP
    # Binary PSO with S-shaped transfer function
    # ---------------------------
    for iteration in range(N_ITERATIONS):

        for i in range(N_PARTICLES):

            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)

            # Velocity update (standard PSO equation)
            velocities[i] = (
                W  * velocities[i]
                + C1 * r1 * (pbest_positions[i] - positions[i])
                + C2 * r2 * (gbest_position    - positions[i])
            )

            # FIX: Velocity clamping — prevents sigmoid saturation
            # Without this, all features get selected (sigmoid → 1)
            velocities[i] = np.clip(velocities[i], -V_MAX, V_MAX)

            # S-shaped transfer function → binary position
            sigmoid      = 1.0 / (1.0 + np.exp(-velocities[i]))
            new_position = (np.random.rand(n_features) < sigmoid).astype(int)

            # Evaluate new position
            score = fitness(new_position)

            # Update personal best
            if score > pbest_scores[i]:
                pbest_positions[i] = new_position.copy()
                pbest_scores[i]    = score

            # Update global best
            if score > gbest_score:
                gbest_position = new_position.copy()
                gbest_score    = score

            positions[i] = new_position

        # FIX: Store convergence point each iteration
        convergence.append(float(gbest_score))

        if (iteration + 1) % 10 == 0:
            n_sel = int(np.sum(gbest_position))
            print(f"  Iteration {iteration+1:3d}: "
                  f"Best Val F1 = {gbest_score:.4f} | "
                  f"Features = {n_sel}")

    return gbest_position, gbest_score, convergence


# =========================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# =========================================
skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=GLOBAL_SEED
)

# Storage across folds
fold_results      = []
all_convergence   = []
all_conf_matrices = []
all_y_true        = []
all_y_pred        = []
all_y_prob        = []

print(f"\n{'='*55}")
print(f"  PSO + Random Forest | {N_FOLDS}-Fold Stratified CV")
print(f"  Particles={N_PARTICLES} | Iterations={N_ITERATIONS}")
print(f"  w={W} | c1={C1} | c2={C2} | Vmax={V_MAX}")
print(f"{'='*55}")

for fold_idx, (train_index, test_index) in enumerate(
        skf.split(X, y), start=1):

    fold_seed = GLOBAL_SEED + fold_idx
    print(f"\n---------- FOLD {fold_idx}/{N_FOLDS} ----------")

    # ---------------------------
    # SPLIT THIS FOLD
    # ---------------------------
    X_fold_train, X_fold_test = X[train_index], X[test_index]
    y_fold_train, y_fold_test = y[train_index], y[test_index]

    # Inner validation split from training fold (20% of train)
    # Used only for PSO fitness evaluation — test fold never touched
    val_size  = int(0.2 * len(X_fold_train))
    val_idx   = np.random.choice(len(X_fold_train),
                                  val_size, replace=False)
    train_idx = np.setdiff1d(
        np.arange(len(X_fold_train)), val_idx
    )

    X_tr  = X_fold_train[train_idx]
    y_tr  = y_fold_train[train_idx]
    X_val = X_fold_train[val_idx]
    y_val = y_fold_train[val_idx]

    # ---------------------------
    # SCALING — fit on X_tr only
    # MinMax
    # ---------------------------
    scaler  = MinMaxScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # ---------------------------
    # RUN PSO ON THIS FOLD
    # ---------------------------
    print(f"  Running PSO feature selection...")
    gbest_position, gbest_val_f1, convergence = run_pso_on_fold(
        X_tr_s, y_tr,
        X_val_s, y_val,
        fold_seed
    )

    selected_idx   = np.where(gbest_position == 1)[0]
    selected_names = feature_names[selected_idx].tolist()

    # FIX: Fallback guard — if no features selected use all
    if len(selected_idx) == 0:
        print("  WARNING: No features selected — using all features.")
        selected_idx   = np.arange(n_features)
        selected_names = feature_names.tolist()

    print(f"  Selected {len(selected_idx)} features: {selected_names}")

    # ---------------------------
    # FINAL MODEL ON FULL FOLD TRAIN
    # FIX: Refit scaler on full fold training set — no leakage
    # ---------------------------
    scaler_final  = MinMaxScaler()
    X_train_final = scaler_final.fit_transform(X_fold_train)
    X_test_final  = scaler_final.transform(X_fold_test)

    final_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=fold_seed
    )
    final_clf.fit(
        X_train_final[:, selected_idx],
        y_fold_train
    )

    y_pred = final_clf.predict(X_test_final[:, selected_idx])
    y_prob = final_clf.predict_proba(
        X_test_final[:, selected_idx]
    )[:, 1]

    # ---------------------------
    # METRICS FOR THIS FOLD
    # ---------------------------
    acc  = accuracy_score(y_fold_test, y_pred)
    prec = precision_score(y_fold_test, y_pred,
                           average='weighted', zero_division=0)
    rec  = recall_score(y_fold_test, y_pred,
                        average='weighted', zero_division=0)
    f1   = f1_score(y_fold_test, y_pred,
                    average='weighted', zero_division=0)
    auc  = roc_auc_score(y_fold_test, y_prob)
    cm   = confusion_matrix(y_fold_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(classification_report(y_fold_test, y_pred,
                                 target_names=['Malignant','Benign']))

    # ---------------------------
    # STORE FOLD RESULTS
    # ---------------------------
    fold_results.append({
        "fold":              fold_idx,
        "n_selected":        len(selected_idx),
        "selected_features": selected_names,
        "val_f1":            round(gbest_val_f1, 4),
        "accuracy":          round(acc, 4),
        "precision":         round(prec, 4),
        "recall":            round(rec, 4),
        "f1_score":          round(f1, 4),
        "auc_roc":           round(auc, 4),
    })

    all_convergence.append(convergence)
    all_conf_matrices.append(cm)
    all_y_true.extend(y_fold_test.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_y_prob.extend(y_prob.tolist())

    # Save per-fold JSON
    fold_dir = os.path.join(results_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "results.json"), "w") as f:
        json.dump({
            **fold_results[-1],
            "convergence_curve": convergence
        }, f, indent=4)


# =========================================
# SUMMARY STATISTICS ACROSS FOLDS
# =========================================
metrics = ["accuracy", "precision", "recall",
           "f1_score", "auc_roc", "n_selected"]
summary = {}

for m in metrics:
    vals       = [r[m] for r in fold_results]
    mean_v     = np.mean(vals)
    std_v      = np.std(vals)
    cv_v       = (std_v / mean_v) * 100 if mean_v > 0 else 0
    summary[m] = {
        "values": vals,
        "mean":   round(mean_v, 4),
        "std":    round(std_v, 4),
        "cv":     round(cv_v, 2)
    }

print(f"\n{'='*55}")
print(f"  FINAL RESULTS — PSO + RF | {N_FOLDS}-Fold Stratified CV")
print(f"{'='*55}")
print(f"  {'Metric':<22} {'Mean':>8} {'± Std':>8} {'CV (%)':>8}")
print(f"  {'-'*48}")
for m in metrics:
    label = m.replace("_", " ").title()
    print(f"  {label:<22} "
          f"{summary[m]['mean']:>8.4f} "
          f"±{summary[m]['std']:>7.4f} "
          f"{summary[m]['cv']:>7.2f}%")

# =========================================
# SAVE SUMMARY CSV
# =========================================
rows = []
for r in fold_results:
    rows.append({
        "Fold":       r["fold"],
        "N Features": r["n_selected"],
        "Accuracy":   r["accuracy"],
        "Precision":  r["precision"],
        "Recall":     r["recall"],
        "F1-Score":   r["f1_score"],
        "AUC-ROC":    r["auc_roc"],
    })

results_df = pd.DataFrame(rows)
csv_path   = os.path.join(results_dir, "pso_rf_5fold_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

sum_path = os.path.join(results_dir, "pso_rf_5fold_summary.csv")
pd.DataFrame([
    {"Metric":   m.replace("_"," ").title(),
     "Mean":     summary[m]["mean"],
     "Std":      summary[m]["std"],
     "CV (%)":   summary[m]["cv"]}
    for m in metrics
]).to_csv(sum_path, index=False)

# =========================================
# FIGURE 1 — CONVERGENCE CURVES
# =========================================
plt.figure(figsize=(8, 5))
for i, conv in enumerate(all_convergence, 1):
    plt.plot(conv, alpha=0.7, linewidth=1.5, label=f"Fold {i}")
mean_conv = np.mean(all_convergence, axis=0)
plt.plot(mean_conv, color='black', linewidth=2.5,
         linestyle='--', label="Mean")
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Best Validation F1-Score", fontsize=12)
plt.title("PSO Convergence Curves — 5-Fold CV", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
conv_fig = os.path.join(figures_dir, "pso_convergence_curves.png")
plt.savefig(conv_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"Convergence figure saved to: {conv_fig}")

# =========================================
# FIGURE 2 — AGGREGATED CONFUSION MATRIX
# =========================================
cm_total = np.sum(all_conf_matrices, axis=0)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_total,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Malignant', 'Benign'],
    yticklabels=['Malignant', 'Benign'],
    linewidths=0.5
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Aggregated Confusion Matrix\nPSO + RF | 5-Fold CV", fontsize=12)
plt.tight_layout()
cm_fig = os.path.join(figures_dir, "pso_rf_confusion_matrix.png")
plt.savefig(cm_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to: {cm_fig}")

# =========================================
# FIGURE 3 — ROC CURVE
# =========================================
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
overall_auc  = roc_auc_score(all_y_true, all_y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', linewidth=2,
         label=f"PSO + RF (AUC = {overall_auc:.4f})")
plt.plot([0,1],[0,1], 'k--', linewidth=1,
         label="Random classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — PSO + RF | 5-Fold CV", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_fig = os.path.join(figures_dir, "pso_rf_roc_curve.png")
plt.savefig(roc_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC curve saved to: {roc_fig}")

# =========================================
# FIGURE 4 — FEATURE SELECTION FREQUENCY
# =========================================
feature_freq = np.zeros(n_features)
for r in fold_results:
    for fname in r["selected_features"]:
        idx = np.where(feature_names == fname)[0]
        if len(idx) > 0:
            feature_freq[idx[0]] += 1

freq_df = pd.DataFrame({
    "Feature":   feature_names,
    "Frequency": feature_freq
}).sort_values("Frequency", ascending=True)

plt.figure(figsize=(8, 7))
plt.barh(freq_df["Feature"], freq_df["Frequency"],
         color='darkorange', edgecolor='white')
plt.xlabel("Number of Folds Selected (out of 5)", fontsize=12)
plt.title("Feature Selection Frequency — PSO | 5-Fold CV",
          fontsize=13)
plt.axvline(x=3, color='red', linestyle='--',
            linewidth=1.5, label="Selected in ≥3 folds")
plt.legend(fontsize=10)
plt.tight_layout()
freq_fig = os.path.join(figures_dir, "pso_feature_frequency.png")
plt.savefig(freq_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"Feature frequency figure saved to: {freq_fig}")

# =========================================
# STABLE FEATURE SUBSET (selected in ≥3 folds)
# =========================================
stable = freq_df[freq_df["Frequency"] >= 3].sort_values(
    "Frequency", ascending=False
)
print(f"\nFeatures selected in ≥3 out of {N_FOLDS} folds "
      f"(stable PSO subset):")
if len(stable) > 0:
    for _, row in stable.iterrows():
        print(f"  {row['Feature']:<35} "
              f"selected in {int(row['Frequency'])}/{N_FOLDS} folds")
else:
    print("  No feature selected in ≥3 folds — "
          "high stochastic variability observed.")

# =========================================
# SAVE CONVERGENCE CSV
# =========================================
conv_df = pd.DataFrame(
    all_convergence,
    index=[f"Fold_{i}" for i in range(1, N_FOLDS+1)]
).T
conv_df.index.name = "Iteration"
conv_path = os.path.join(results_dir, "pso_convergence_curves.csv")
conv_df.to_csv(conv_path)
print(f"Convergence data saved to: {conv_path}")

# =========================================
# FINAL SUMMARY PRINT
# =========================================
print(f"\n{'='*55}")
print(f"  READY SUMMARY — PSO + RF")
print(f"  Evaluation: {N_FOLDS}-Fold Stratified Cross-Validation")
print(f"  Dataset: Breast Cancer Wisconsin ({X.shape[0]} samples)")
print(f"{'='*55}")
print(f"  Accuracy:   {summary['accuracy']['mean']:.4f} "
      f"± {summary['accuracy']['std']:.4f} "
      f"(CV={summary['accuracy']['cv']:.2f}%)")
print(f"  Precision:  {summary['precision']['mean']:.4f} "
      f"± {summary['precision']['std']:.4f} "
      f"(CV={summary['precision']['cv']:.2f}%)")
print(f"  Recall:     {summary['recall']['mean']:.4f} "
      f"± {summary['recall']['std']:.4f} "
      f"(CV={summary['recall']['cv']:.2f}%)")
print(f"  F1-Score:   {summary['f1_score']['mean']:.4f} "
      f"± {summary['f1_score']['std']:.4f} "
      f"(CV={summary['f1_score']['cv']:.2f}%)")
print(f"  AUC-ROC:    {summary['auc_roc']['mean']:.4f} "
      f"± {summary['auc_roc']['std']:.4f} "
      f"(CV={summary['auc_roc']['cv']:.2f}%)")
print(f"  Avg Features: {summary['n_selected']['mean']:.1f} "
      f"± {summary['n_selected']['std']:.2f} "
      f"/ {n_features}")
print(f"{'='*55}")
print("\nExperiment completed successfully.")
print(f"Results available in: {results_dir}")
print("Thank you for using this research software.")