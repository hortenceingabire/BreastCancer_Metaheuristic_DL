"""
===========================================================================
Classical Baseline Classifiers for Breast Cancer Classification
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
This script implements classical machine learning baseline classifiers
for breast cancer classification under two experimental conditions:

• Full feature set (30 features)
• PSO-selected feature subset (10 features)

The following baseline models are evaluated:

• Random Forest (RF)
• Extreme Gradient Boosting (XGBoost)

All models are evaluated using 5-fold stratified cross-validation on the
Breast Cancer Wisconsin Diagnostic (BCWD) dataset.

The implementation reproduces the baseline experiments reported in the
associated manuscript, including:

• Classical machine learning classification
• Full vs PSO-selected feature comparison
• 5-fold stratified cross-validation
• Performance metrics
• ROC curves
• Aggregated confusion matrices
• Statistical comparison using paired t-tests

These baseline models provide reference performance for comparison with
the proposed deep learning architectures (CNN, GRU, and CNN-GRU) using
identical preprocessing, feature subsets, cross-validation strategy,
evaluation metrics, and statistical analyses.

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    classification_report, roc_curve
)
from scipy.stats import ttest_rel
from xgboost import XGBClassifier

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
# CV SETTINGS — identical to all DL models
# =========================================
N_FOLDS = 5

# =========================================
# SAVE DIRECTORIES
# =========================================
results_dir = "RF_XGB_5FOLD_RESULTS"
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
X_raw         = pd.DataFrame(data.data, columns=data.feature_names)
y             = data.target
feature_names = np.array(data.feature_names)
n_features    = X_raw.shape[1]

print(f"Dataset: {X_raw.shape[0]} samples | {n_features} features")
print(f"Class distribution — Malignant (0): {np.sum(y==0)} | "
      f"Benign (1): {np.sum(y==1)}")

# =========================================
# PSO STABLE FEATURE SUBSET
# Features selected in ≥3/5 folds from PSO-RF 5-fold results
# Identical to CNN, GRU, CNN-GRU — ensures fair comparison
# =========================================
PSO_SELECTED_FEATURES = [
    'fractal dimension error',   # selected in 4/5 folds
    'smoothness error',          # selected in 4/5 folds
    'mean compactness',          # selected in 4/5 folds
    'worst concave points',      # selected in 3/5 folds
    'worst area',                # selected in 3/5 folds
    'mean texture',              # selected in 3/5 folds
    'worst smoothness',          # selected in 3/5 folds
    'worst concavity',           # selected in 3/5 folds
    'perimeter error',           # selected in 3/5 folds
    'concavity error',           # selected in 3/5 folds
]

X_full     = X_raw.values
X_selected = X_raw[PSO_SELECTED_FEATURES].values

print(f"\nFull feature set:    {X_full.shape[1]} features")
print(f"PSO selected subset: {X_selected.shape[1]} features")
print(f"PSO features: {PSO_SELECTED_FEATURES}")


# =========================================
# MODEL BUILDERS
# =========================================
def build_rf(seed):
    """
    Random Forest classifier.
    100 estimators — consistent with RF used
    as fitness function in PSO-RF feature selection.
    """
    return RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=seed
    )

def build_xgb(seed):
    """
    XGBoost classifier.
    Standard hyperparameters — no tuning applied,
    consistent with baseline comparison purpose.
    """
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss',
        random_state=seed,
        verbosity=0
    )


# =========================================
# EVALUATION FUNCTION
# Runs 5-fold CV for a given model and feature set
# Identical CV structure to all DL experiments
# =========================================
def run_cv(X, model_name, model_builder, condition_name,
           feature_count):
    """
    Runs 5-fold stratified CV for a classical classifier.

    Parameters
    ----------
    X              : feature matrix (n_samples, n_features)
    model_name     : string name for reporting (RF / XGB)
    model_builder  : function that returns a fresh model
    condition_name : string label (Full_Features / PSO_Selected)
    feature_count  : number of features (for reporting)

    Returns
    -------
    fold_results  : list of per-fold metric dicts
    all_y_true    : aggregated true labels
    all_y_pred    : aggregated predicted labels
    all_y_prob    : aggregated predicted probabilities
    cm_total      : aggregated confusion matrix
    """

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=GLOBAL_SEED
    )

    fold_results  = []
    all_conf_mats = []
    all_y_true    = []
    all_y_pred    = []
    all_y_prob    = []

    print(f"\n{'='*55}")
    print(f"  {model_name} | {condition_name} | "
          f"{N_FOLDS}-Fold Stratified CV")
    print(f"  Features: {feature_count}")
    print(f"{'='*55}")

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(X, y), start=1):

        fold_seed = GLOBAL_SEED + fold_idx
        np.random.seed(fold_seed)

        print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")

        # ---------------------------
        # SPLIT
        # ---------------------------
        X_fold_train = X[train_idx]
        X_fold_test  = X[test_idx]
        y_fold_train = y[train_idx]
        y_fold_test  = y[test_idx]

        # ---------------------------
        # SCALING — fit on train only
        # MinMax consistent with DL models
        # ---------------------------
        scaler        = MinMaxScaler()
        X_train_final = scaler.fit_transform(X_fold_train)
        X_test_final  = scaler.transform(X_fold_test)

        # ---------------------------
        # TRAIN MODEL
        # ---------------------------
        model = model_builder(fold_seed)
        model.fit(X_train_final, y_fold_train)

        # ---------------------------
        # EVALUATE ON TEST FOLD
        # ---------------------------
        y_pred   = model.predict(X_test_final)
        y_prob   = model.predict_proba(X_test_final)[:, 1]

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
        print(classification_report(
            y_fold_test, y_pred,
            target_names=['Malignant', 'Benign']
        ))

        fold_results.append({
            "fold":      fold_idx,
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1_score":  round(f1,   4),
            "auc_roc":   round(auc,  4),
        })

        all_conf_mats.append(cm)
        all_y_true.extend(y_fold_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        # Save per-fold JSON
        fold_dir = os.path.join(
            results_dir, model_name, condition_name,
            f"fold_{fold_idx}"
        )
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "results.json"), "w") as f:
            json.dump(fold_results[-1], f, indent=4)

    cm_total = np.sum(all_conf_mats, axis=0)

    return (fold_results, all_y_true, all_y_pred,
            all_y_prob, cm_total)


# =========================================
# SUMMARY STATISTICS HELPER
# =========================================
def summarize(fold_results):
    metrics = ["accuracy", "precision", "recall",
               "f1_score", "auc_roc"]
    summary = {}
    for m in metrics:
        vals   = [r[m] for r in fold_results]
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        cv_v   = (std_v / mean_v) * 100 if mean_v > 0 else 0
        summary[m] = {
            "values": vals,
            "mean":   round(mean_v, 4),
            "std":    round(std_v,  4),
            "cv":     round(cv_v,   2)
        }
    return summary


# =========================================
# RUN ALL FOUR CONDITIONS
# RF  — Full Features (30)
# RF  — PSO Selected (10)
# XGB — Full Features (30)
# XGB — PSO Selected (10)
# =========================================

# --- RF Full ---
(res_rf_full, yt_rf_full, yp_rf_full,
 yprob_rf_full, cm_rf_full) = run_cv(
    X_full, "RF", build_rf,
    "Full_Features", n_features
)
summary_rf_full = summarize(res_rf_full)

# --- RF PSO ---
(res_rf_sel, yt_rf_sel, yp_rf_sel,
 yprob_rf_sel, cm_rf_sel) = run_cv(
    X_selected, "RF", build_rf,
    "PSO_Selected", len(PSO_SELECTED_FEATURES)
)
summary_rf_sel = summarize(res_rf_sel)

# --- XGB Full ---
(res_xgb_full, yt_xgb_full, yp_xgb_full,
 yprob_xgb_full, cm_xgb_full) = run_cv(
    X_full, "XGB", build_xgb,
    "Full_Features", n_features
)
summary_xgb_full = summarize(res_xgb_full)

# --- XGB PSO ---
(res_xgb_sel, yt_xgb_sel, yp_xgb_sel,
 yprob_xgb_sel, cm_xgb_sel) = run_cv(
    X_selected, "XGB", build_xgb,
    "PSO_Selected", len(PSO_SELECTED_FEATURES)
)
summary_xgb_sel = summarize(res_xgb_sel)


# =========================================
# PAIRED T-TESTS — Full vs PSO per model
# =========================================
def ttest_pair(res_full, res_sel, metric):
    full_vals = [r[metric] for r in res_full]
    sel_vals  = [r[metric] for r in res_sel]
    t, p      = ttest_rel(full_vals, sel_vals)
    return round(t, 4), round(p, 4)

t_rf_acc,  p_rf_acc  = ttest_pair(res_rf_full,  res_rf_sel,  "accuracy")
t_rf_f1,   p_rf_f1   = ttest_pair(res_rf_full,  res_rf_sel,  "f1_score")
t_xgb_acc, p_xgb_acc = ttest_pair(res_xgb_full, res_xgb_sel, "accuracy")
t_xgb_f1,  p_xgb_f1  = ttest_pair(res_xgb_full, res_xgb_sel, "f1_score")

print(f"\n{'='*55}")
print(f"  PAIRED T-TEST RESULTS")
print(f"{'='*55}")
print(f"  RF  Accuracy: t={t_rf_acc},  p={p_rf_acc} "
      f"{'(significant)' if p_rf_acc  < 0.05 else '(not significant)'}")
print(f"  RF  F1-Score: t={t_rf_f1},   p={p_rf_f1} "
      f"{'(significant)' if p_rf_f1   < 0.05 else '(not significant)'}")
print(f"  XGB Accuracy: t={t_xgb_acc}, p={p_xgb_acc} "
      f"{'(significant)' if p_xgb_acc < 0.05 else '(not significant)'}")
print(f"  XGB F1-Score: t={t_xgb_f1},  p={p_xgb_f1} "
      f"{'(significant)' if p_xgb_f1  < 0.05 else '(not significant)'}")


# =========================================
# PRINT FULL COMPARISON TABLE
# =========================================
metrics_labels = ["Accuracy", "Precision", "Recall",
                  "F1-Score", "AUC-ROC"]
metrics_keys   = ["accuracy", "precision", "recall",
                  "f1_score", "auc_roc"]

print(f"\n{'='*70}")
print(f"  BASELINE RESULTS | {N_FOLDS}-Fold Stratified CV")
print(f"{'='*70}")
print(f"  {'Metric':<14} {'RF Full':>14} {'RF PSO':>14} "
      f"{'XGB Full':>14} {'XGB PSO':>14}")
print(f"  {'-'*66}")
for label, key in zip(metrics_labels, metrics_keys):
    rf_f  = summary_rf_full[key]
    rf_s  = summary_rf_sel[key]
    xgb_f = summary_xgb_full[key]
    xgb_s = summary_xgb_sel[key]
    print(f"  {label:<14} "
          f"{rf_f['mean']:.4f}±{rf_f['std']:.4f}  "
          f"{rf_s['mean']:.4f}±{rf_s['std']:.4f}  "
          f"{xgb_f['mean']:.4f}±{xgb_f['std']:.4f}  "
          f"{xgb_s['mean']:.4f}±{xgb_s['std']:.4f}")


# =========================================
# SAVE RESULTS CSV
# =========================================
rows = []
for fold_idx in range(N_FOLDS):
    rows.append({
        "Fold":              fold_idx + 1,
        "RF_Full_Accuracy":  res_rf_full[fold_idx]["accuracy"],
        "RF_Full_F1":        res_rf_full[fold_idx]["f1_score"],
        "RF_Full_AUC":       res_rf_full[fold_idx]["auc_roc"],
        "RF_PSO_Accuracy":   res_rf_sel[fold_idx]["accuracy"],
        "RF_PSO_F1":         res_rf_sel[fold_idx]["f1_score"],
        "RF_PSO_AUC":        res_rf_sel[fold_idx]["auc_roc"],
        "XGB_Full_Accuracy": res_xgb_full[fold_idx]["accuracy"],
        "XGB_Full_F1":       res_xgb_full[fold_idx]["f1_score"],
        "XGB_Full_AUC":      res_xgb_full[fold_idx]["auc_roc"],
        "XGB_PSO_Accuracy":  res_xgb_sel[fold_idx]["accuracy"],
        "XGB_PSO_F1":        res_xgb_sel[fold_idx]["f1_score"],
        "XGB_PSO_AUC":       res_xgb_sel[fold_idx]["auc_roc"],
    })

results_df = pd.DataFrame(rows)
csv_path   = os.path.join(
    results_dir, "baseline_5fold_results.csv"
)
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")


# =========================================
# FIGURE 1 — CONFUSION MATRICES (2x2 grid)
# RF Full | RF PSO | XGB Full | XGB PSO
# =========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
configs_cm = [
    (axes[0,0], cm_rf_full,  "RF — Full Features (30)"),
    (axes[0,1], cm_rf_sel,   "RF — PSO Selected (10)"),
    (axes[1,0], cm_xgb_full, "XGB — Full Features (30)"),
    (axes[1,1], cm_xgb_sel,  "XGB — PSO Selected (10)"),
]
for ax, cm, title in configs_cm:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12)
plt.suptitle(
    "Aggregated Confusion Matrices — RF & XGB Baselines\n"
    "5-Fold Stratified CV | Full vs PSO-Selected Features",
    fontsize=13, y=1.01
)
plt.tight_layout()
cm_path = os.path.join(
    figures_dir, "baseline_confusion_matrices.png"
)
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrices saved to: {cm_path}")


# =========================================
# FIGURE 2 — ROC CURVES (all 4 conditions)
# RF and XGB, Full and PSO on one plot
# =========================================
fig, ax = plt.subplots(figsize=(7, 6))
roc_configs = [
    (yt_rf_full,  yprob_rf_full,  "RF — Full (30)",    "steelblue",   "-"),
    (yt_rf_sel,   yprob_rf_sel,   "RF — PSO (10)",     "steelblue",   "--"),
    (yt_xgb_full, yprob_xgb_full, "XGB — Full (30)",   "darkorange",  "-"),
    (yt_xgb_sel,  yprob_xgb_sel,  "XGB — PSO (10)",    "darkorange",  "--"),
]
for yt, yp, label, color, ls in roc_configs:
    fpr, tpr, _ = roc_curve(yt, yp)
    auc_val = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            linestyle=ls,
            label=f"{label} (AUC={auc_val:.4f})")
ax.plot([0,1],[0,1], 'k--', linewidth=1,
        label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(
    "ROC Curves — RF & XGB Baselines | 5-Fold CV",
    fontsize=13
)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(figures_dir, "baseline_roc_curves.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC curves saved to: {roc_path}")


# =========================================
# FIGURE 3 — BAR COMPARISON
# All 4 conditions across all 5 metrics
# =========================================
x     = np.arange(len(metrics_labels))
width = 0.20

rf_full_means  = [summary_rf_full[k]['mean']  for k in metrics_keys]
rf_sel_means   = [summary_rf_sel[k]['mean']   for k in metrics_keys]
xgb_full_means = [summary_xgb_full[k]['mean'] for k in metrics_keys]
xgb_sel_means  = [summary_xgb_sel[k]['mean']  for k in metrics_keys]

rf_full_stds   = [summary_rf_full[k]['std']   for k in metrics_keys]
rf_sel_stds    = [summary_rf_sel[k]['std']    for k in metrics_keys]
xgb_full_stds  = [summary_xgb_full[k]['std']  for k in metrics_keys]
xgb_sel_stds   = [summary_xgb_sel[k]['std']   for k in metrics_keys]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 1.5*width, rf_full_means,  width,
       yerr=rf_full_stds,  capsize=3,
       color='steelblue',  alpha=0.85,
       label='RF — Full (30)')
ax.bar(x - 0.5*width, rf_sel_means,   width,
       yerr=rf_sel_stds,   capsize=3,
       color='steelblue',  alpha=0.50,
       label='RF — PSO (10)')
ax.bar(x + 0.5*width, xgb_full_means, width,
       yerr=xgb_full_stds, capsize=3,
       color='darkorange', alpha=0.85,
       label='XGB — Full (30)')
ax.bar(x + 1.5*width, xgb_sel_means,  width,
       yerr=xgb_sel_stds,  capsize=3,
       color='darkorange', alpha=0.50,
       label='XGB — PSO (10)')
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    "Baseline Performance: RF & XGB | Full vs PSO Features\n"
    "5-Fold Stratified CV (Mean ± Std)", fontsize=12
)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(0.88, 1.02)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
bar_path = os.path.join(
    figures_dir, "baseline_metric_comparison.png"
)
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Metric comparison saved to: {bar_path}")


# =========================================
# FIGURE 4 — COMPLETE MODEL COMPARISON
# All models on PSO-selected features
# RF, XGB, CNN, GRU, CNN-GRU side by side
# =========================================

# Results from DL experiments (paste from results)
dl_results = {
    # CNN — PSO selected, 10 features
    "CNN":     {"accuracy": 0.9649, "f1_score": 0.9646,
                "auc_roc": 0.9919, "acc_std": 0.0200,
                "f1_std":  0.0202, "auc_std": 0.0073},
    # GRU — PSO selected, 10 features
    "GRU":     {"accuracy": 0.9297, "f1_score": 0.9293,
                "auc_roc": 0.9826, "acc_std": 0.0077,
                "f1_std":  0.0078, "auc_std": 0.0063},
    # CNN-GRU — PSO selected, 10 features
    "CNN-GRU": {"accuracy": 0.9613, "f1_score": 0.9610,
                "auc_roc": 0.9890, "acc_std": 0.0153,
                "f1_std":  0.0156, "auc_std": 0.0075},
}

all_models = ["RF", "XGB", "CNN", "GRU", "CNN-GRU"]
all_acc    = [
    summary_rf_sel["accuracy"]["mean"],
    summary_xgb_sel["accuracy"]["mean"],
    dl_results["CNN"]["accuracy"],
    dl_results["GRU"]["accuracy"],
    dl_results["CNN-GRU"]["accuracy"],
]
all_f1     = [
    summary_rf_sel["f1_score"]["mean"],
    summary_xgb_sel["f1_score"]["mean"],
    dl_results["CNN"]["f1_score"],
    dl_results["GRU"]["f1_score"],
    dl_results["CNN-GRU"]["f1_score"],
]
all_auc    = [
    summary_rf_sel["auc_roc"]["mean"],
    summary_xgb_sel["auc_roc"]["mean"],
    dl_results["CNN"]["auc_roc"],
    dl_results["GRU"]["auc_roc"],
    dl_results["CNN-GRU"]["auc_roc"],
]
all_acc_std = [
    summary_rf_sel["accuracy"]["std"],
    summary_xgb_sel["accuracy"]["std"],
    dl_results["CNN"]["acc_std"],
    dl_results["GRU"]["acc_std"],
    dl_results["CNN-GRU"]["acc_std"],
]
all_f1_std = [
    summary_rf_sel["f1_score"]["std"],
    summary_xgb_sel["f1_score"]["std"],
    dl_results["CNN"]["f1_std"],
    dl_results["GRU"]["f1_std"],
    dl_results["CNN-GRU"]["f1_std"],
]

# Color coding: classical = steelblue, DL = darkorange
colors = ['steelblue','steelblue',
          'darkorange','darkorange','darkorange']

x     = np.arange(len(all_models))
width = 0.30

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy subplot
axes[0].bar(x, all_acc, width,
            yerr=all_acc_std, capsize=4,
            color=colors, alpha=0.85,
            edgecolor='white')
axes[0].set_ylabel("Mean Accuracy", fontsize=12)
axes[0].set_title(
    "Accuracy — All Models\nPSO-Selected Features (10)",
    fontsize=12
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(all_models, fontsize=11)
axes[0].set_ylim(0.88, 1.02)
axes[0].grid(axis='y', alpha=0.3)
for i, (v, s) in enumerate(zip(all_acc, all_acc_std)):
    axes[0].text(i, v + s + 0.003,
                 f"{v:.4f}", ha='center',
                 va='bottom', fontsize=8)

# F1-Score subplot
axes[1].bar(x, all_f1, width,
            yerr=all_f1_std, capsize=4,
            color=colors, alpha=0.85,
            edgecolor='white')
axes[1].set_ylabel("Mean F1-Score", fontsize=12)
axes[1].set_title(
    "F1-Score — All Models\nPSO-Selected Features (10)",
    fontsize=12
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(all_models, fontsize=11)
axes[1].set_ylim(0.88, 1.02)
axes[1].grid(axis='y', alpha=0.3)
for i, (v, s) in enumerate(zip(all_f1, all_f1_std)):
    axes[1].text(i, v + s + 0.003,
                 f"{v:.4f}", ha='center',
                 va='bottom', fontsize=8)

# Shared legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue',  alpha=0.85,
          label='Classical (RF, XGB)'),
    Patch(facecolor='darkorange', alpha=0.85,
          label='Deep Learning (CNN, GRU, CNN-GRU)')
]
fig.legend(handles=legend_elements, fontsize=11,
           loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.02))
plt.suptitle(
    "Complete Model Comparison — PSO-Selected Features (10)\n"
    "5-Fold Stratified CV (Mean ± Std)",
    fontsize=13, y=1.08
)
plt.tight_layout()
compare_path = os.path.join(
    figures_dir, "all_models_comparison.png"
)
plt.savefig(compare_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Complete model comparison saved to: {compare_path}")


# =========================================
# FINAL PAPER-READY SUMMARY
# =========================================
print(f"\n{'='*70}")
print(f"  PAPER-READY SUMMARY — RF & XGB Baselines")
print(f"  Evaluation: {N_FOLDS}-Fold Stratified Cross-Validation")
print(f"  Dataset: Breast Cancer Wisconsin "
      f"({X_raw.shape[0]} samples)")
print(f"{'='*70}")

for model_name, summ_full, summ_sel, t_acc, p_acc, t_f1, p_f1 in [
    ("Random Forest",
     summary_rf_full,  summary_rf_sel,
     t_rf_acc,  p_rf_acc,  t_rf_f1,  p_rf_f1),
    ("XGBoost",
     summary_xgb_full, summary_xgb_sel,
     t_xgb_acc, p_xgb_acc, t_xgb_f1, p_xgb_f1),
]:
    print(f"\n  {model_name.upper()}")
    print(f"  {'Metric':<12} {'Full (30)':>16} {'PSO (10)':>16} "
          f"{'Δ':>8}")
    print(f"  {'-'*54}")
    for label, key in zip(metrics_labels, metrics_keys):
        mf    = summ_full[key]['mean']
        sf    = summ_full[key]['std']
        ms    = summ_sel[key]['mean']
        ss    = summ_sel[key]['std']
        delta = ms - mf
        print(f"  {label:<12} {mf:.4f}±{sf:.4f}   "
              f"{ms:.4f}±{ss:.4f}   {delta:+.4f}")
    print(f"\n  Paired t-test (Accuracy): "
          f"t={t_acc}, p={p_acc} "
          f"{'→ significant' if p_acc < 0.05 else '→ not significant'}")
    print(f"  Paired t-test (F1-Score): "
          f"t={t_f1},  p={p_f1} "
          f"{'→ significant' if p_f1  < 0.05 else '→ not significant'}")

print(f"\n{'='*70}")
print(f"  COMPLETE MODEL RANKING — PSO Selected Features")
print(f"{'='*70}")
print(f"  {'Model':<12} {'Accuracy':>12} {'F1-Score':>12} "
      f"{'AUC-ROC':>12}")
print(f"  {'-'*50}")
ranking = sorted(
    zip(all_models, all_acc, all_f1, all_auc),
    key=lambda x: x[1], reverse=True
)
for rank, (model, acc, f1, auc) in enumerate(ranking, 1):
    print(f"  {rank}. {model:<10} {acc:.4f}        "
          f"{f1:.4f}        {auc:.4f}")
print(f"{'='*70}")
print("\nExperiment completed successfully.")
print(f"Results available in: {results_dir}")
print("Thank you for using this research software.")