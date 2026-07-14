"""
===========================================================================
CNN Classification Using PSO-Selected Features
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
This script implements breast cancer classification using a
Convolutional Neural Network (CNN) trained under two experimental
conditions:

• Full feature set (30 features)
• PSO-selected feature subset (10 features)

The models are evaluated using 5-fold stratified cross-validation on the
Breast Cancer Wisconsin Diagnostic (BCWD) dataset.

The implementation reproduces the experiments reported in the associated
manuscript, including:

• CNN classification
• Full vs PSO-selected feature comparison
• 5-fold stratified cross-validation
• Early stopping
• Performance metrics
• ROC curves
• Aggregated confusion matrices
• Training history analysis
• Paired statistical comparison

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
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    classification_report, roc_curve
)
from scipy.stats import ttest_rel

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten,
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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
tf.random.set_seed(GLOBAL_SEED)

# =========================================
# CV SETTINGS — identical to GRU, CNN-GRU & PSO-RF
# =========================================
N_FOLDS = 5

# =========================================
# CNN HYPERPARAMETERS
# =========================================
EPOCHS     = 100
BATCH_SIZE = 16
PATIENCE   = 5

# =========================================
# SAVE DIRECTORIES
# =========================================
results_dir = "CNN_5FOLD_RESULTS"
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
# Identical to GRU and CNN-GRU — ensures fair DL comparison
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
# CNN MODEL BUILDER
# Identical architecture for both conditions
# =========================================
def build_cnn(n_input_features, seed):
    """
    Builds CNN model for 1D tabular input.
    Architecture kept identical across both
    full and selected feature conditions.

    Parameters
    ----------
    n_input_features : int, number of input features
    seed             : int, random seed for reproducibility
    """
    tf.random.set_seed(seed)

    model = Sequential([
        Conv1D(filters=64, kernel_size=3,
               activation='relu',
               input_shape=(n_input_features, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(2,  activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================
# EVALUATION FUNCTION
# Runs 5-fold CV for a given feature set
# Identical structure to GRU and CNN-GRU run_cv()
# =========================================
def run_cv(X, condition_name, feature_count):
    """
    Runs 5-fold stratified CV for CNN on a given feature set.

    Parameters
    ----------
    X              : feature matrix (n_samples, n_features)
    condition_name : string label for saving results
    feature_count  : number of features (for reporting)

    Returns
    -------
    fold_results  : list of per-fold metric dicts
    all_y_true    : aggregated true labels
    all_y_pred    : aggregated predicted labels
    all_y_prob    : aggregated predicted probabilities
    all_history   : training history per fold
    cm_total      : aggregated confusion matrix
    """

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=GLOBAL_SEED
    )

    fold_results   = []
    all_conf_mats  = []
    all_y_true     = []
    all_y_pred     = []
    all_y_prob     = []
    all_history    = []

    print(f"\n{'='*55}")
    print(f"  CNN | {condition_name} | {N_FOLDS}-Fold Stratified CV")
    print(f"  Features: {feature_count} | "
          f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"{'='*55}")

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(X, y), start=1):

        fold_seed = GLOBAL_SEED + fold_idx
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")

        # ---------------------------
        # SPLIT
        # ---------------------------
        X_fold_train = X[train_idx]
        X_fold_test  = X[test_idx]
        y_fold_train = y[train_idx]
        y_fold_test  = y[test_idx]

        # Inner validation split (20% of fold train)
        # Used only for EarlyStopping — not for evaluation
        val_size = int(0.2 * len(X_fold_train))
        val_idx  = np.random.choice(
            len(X_fold_train), val_size, replace=False
        )
        tr_idx   = np.setdiff1d(
            np.arange(len(X_fold_train)), val_idx
        )

        X_tr  = X_fold_train[tr_idx]
        y_tr  = y_fold_train[tr_idx]
        X_val = X_fold_train[val_idx]
        y_val = y_fold_train[val_idx]

        # ---------------------------
        # SCALING — fit on X_tr only
        # ---------------------------
        scaler  = MinMaxScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # ---------------------------
        # FINAL SCALER — refit on full fold train
        # No leakage — test fold never seen
        # ---------------------------
        scaler_final  = MinMaxScaler()
        X_train_final = scaler_final.fit_transform(X_fold_train)
        X_test_final  = scaler_final.transform(X_fold_test)

        # Reshape for Conv1D: (samples, timesteps, channels)
        n_feat        = X_tr_s.shape[1]
        X_tr_s        = X_tr_s.reshape(-1, n_feat, 1)
        X_val_s       = X_val_s.reshape(-1, n_feat, 1)
        X_train_final = X_train_final.reshape(-1, n_feat, 1)
        X_test_final  = X_test_final.reshape(-1, n_feat, 1)

        # One-hot encode labels
        y_tr_cat    = to_categorical(y_tr,         num_classes=2)
        y_val_cat   = to_categorical(y_val,        num_classes=2)
        y_train_cat = to_categorical(y_fold_train, num_classes=2)

        # ---------------------------
        # BUILD & TRAIN MODEL
        # ---------------------------
        model = build_cnn(n_feat, fold_seed)

        # Print model summary only on first fold, full features
        if fold_idx == 1 and feature_count == n_features:
            print("\n  Model Architecture:")
            model.summary()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_tr_s, y_tr_cat,
            validation_data=(X_val_s, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        epochs_run = len(history.history['loss'])
        print(f"  Epochs run: {epochs_run} "
              f"(early stop patience={PATIENCE})")

        # Save full training history for overfitting analysis
        all_history.append({
            "fold":       fold_idx,
            "train_loss": history.history['loss'],
            "val_loss":   history.history['val_loss'],
            "train_acc":  history.history['accuracy'],
            "val_acc":    history.history['val_accuracy'],
            "epochs_run": epochs_run
        })

        # ---------------------------
        # EVALUATE ON TEST FOLD
        # ---------------------------
        y_prob_2d = model.predict(X_test_final, verbose=0)
        y_pred    = np.argmax(y_prob_2d, axis=1)
        y_prob    = y_prob_2d[:, 1]

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
            "epochs":    epochs_run
        })

        all_conf_mats.append(cm)
        all_y_true.extend(y_fold_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        # Save per-fold JSON
        fold_dir = os.path.join(
            results_dir, condition_name, f"fold_{fold_idx}"
        )
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "results.json"), "w") as f:
            json.dump(fold_results[-1], f, indent=4)

    cm_total = np.sum(all_conf_mats, axis=0)

    return (fold_results, all_y_true, all_y_pred,
            all_y_prob, all_history, cm_total)


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
# TRAINING HISTORY HELPER
# Pads fold histories to same length with NaN
# =========================================
def prep_history(hist_list):
    """Pads all fold histories to same length with NaN."""
    max_ep = max(h["epochs_run"] for h in hist_list)
    arrays = {k: np.full((N_FOLDS, max_ep), np.nan)
              for k in ["train_loss", "val_loss",
                         "train_acc",  "val_acc"]}
    for i, h in enumerate(hist_list):
        ep = h["epochs_run"]
        arrays["train_loss"][i, :ep] = h["train_loss"]
        arrays["val_loss"][i,   :ep] = h["val_loss"]
        arrays["train_acc"][i,  :ep] = h["train_acc"]
        arrays["val_acc"][i,    :ep] = h["val_acc"]
    means = {k: np.nanmean(v, axis=0)
             for k, v in arrays.items()}
    return arrays, means, max_ep


# =========================================
# RUN CONDITION 1 — FULL FEATURE SET (30)
# =========================================
(res_full, yt_full, yp_full,
 yprob_full, hist_full, cm_full) = run_cv(
    X_full, "Full_Features", n_features
)
summary_full = summarize(res_full)

# =========================================
# RUN CONDITION 2 — PSO SELECTED (10)
# =========================================
(res_sel, yt_sel, yp_sel,
 yprob_sel, hist_sel, cm_sel) = run_cv(
    X_selected, "PSO_Selected", len(PSO_SELECTED_FEATURES)
)
summary_sel = summarize(res_sel)


# =========================================
# PAIRED T-TEST — Full vs PSO Selected
# =========================================
acc_full   = [r["accuracy"] for r in res_full]
acc_sel    = [r["accuracy"] for r in res_sel]
t_stat,    p_value    = ttest_rel(acc_full, acc_sel)

f1_full    = [r["f1_score"] for r in res_full]
f1_sel     = [r["f1_score"] for r in res_sel]
t_stat_f1, p_value_f1 = ttest_rel(f1_full, f1_sel)

print(f"\n{'='*55}")
print(f"  PAIRED T-TEST: Full vs PSO-Selected Features")
print(f"{'='*55}")
print(f"  Accuracy — t={t_stat:.4f}, p={p_value:.4f} "
      f"{'(significant)' if p_value < 0.05 else '(not significant)'}")
print(f"  F1-Score  — t={t_stat_f1:.4f}, p={p_value_f1:.4f} "
      f"{'(significant)' if p_value_f1 < 0.05 else '(not significant)'}")


# =========================================
# PRINT COMPARISON TABLE
# =========================================
metrics_labels = ["Accuracy", "Precision", "Recall",
                  "F1-Score", "AUC-ROC"]
metrics_keys   = ["accuracy", "precision", "recall",
                  "f1_score", "auc_roc"]

print(f"\n{'='*65}")
print(f"  CNN RESULTS COMPARISON | {N_FOLDS}-Fold Stratified CV")
print(f"{'='*65}")
print(f"  {'Metric':<18} {'Full (30 feat)':>18} "
      f"{'PSO (10 feat)':>18} {'Δ':>8}")
print(f"  {'-'*62}")
for label, key in zip(metrics_labels, metrics_keys):
    mf    = summary_full[key]['mean']
    sf    = summary_full[key]['std']
    ms    = summary_sel[key]['mean']
    ss    = summary_sel[key]['std']
    delta = ms - mf
    print(f"  {label:<18} {mf:.4f}±{sf:.4f}   "
          f"{ms:.4f}±{ss:.4f}   {delta:+.4f}")
print(f"\n  Paired t-test (Accuracy): "
      f"t={t_stat:.4f}, p={p_value:.4f}")
print(f"  Paired t-test (F1-Score): "
      f"t={t_stat_f1:.4f}, p={p_value_f1:.4f}")


# =========================================
# SAVE RESULTS CSV
# =========================================
rows = []
for r_f, r_s in zip(res_full, res_sel):
    rows.append({
        "Fold":           r_f["fold"],
        "Full_Accuracy":  r_f["accuracy"],
        "Full_Precision": r_f["precision"],
        "Full_Recall":    r_f["recall"],
        "Full_F1":        r_f["f1_score"],
        "Full_AUC":       r_f["auc_roc"],
        "Full_Epochs":    r_f["epochs"],
        "PSO_Accuracy":   r_s["accuracy"],
        "PSO_Precision":  r_s["precision"],
        "PSO_Recall":     r_s["recall"],
        "PSO_F1":         r_s["f1_score"],
        "PSO_AUC":        r_s["auc_roc"],
        "PSO_Epochs":     r_s["epochs"],
    })

results_df = pd.DataFrame(rows)
csv_path   = os.path.join(results_dir, "cnn_5fold_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")


# =========================================
# FIGURE 1 — COMBINED TRAINING HISTORY
# 2 rows × 4 columns:
#   Row 1: Loss curves     — Full (cols 0-1) | PSO (cols 2-3)
#   Row 2: Accuracy curves — Full (cols 0-1) | PSO (cols 2-3)
# Each panel: all 5 folds (light) + mean (black dashed)
# Single figure for direct Full vs PSO comparison
# Consistent with GRU and CNN-GRU figures
# =========================================
arrays_full, means_full, max_ep_full = prep_history(hist_full)
arrays_sel,  means_sel,  max_ep_sel  = prep_history(hist_sel)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle(
    "CNN Training History — Full Features (30) vs PSO Selected (10)\n"
    "5-Fold Stratified CV  |  Individual folds (light) + Mean (dashed)",
    fontsize=13, y=1.01
)

configs = [
    (0, 0, arrays_full, means_full, max_ep_full,
     "train_loss", "Training Loss\n[Full Features]",
     "Loss",      "steelblue"),
    (0, 1, arrays_full, means_full, max_ep_full,
     "val_loss",   "Validation Loss\n[Full Features]",
     "Loss",      "steelblue"),
    (0, 2, arrays_sel,  means_sel,  max_ep_sel,
     "train_loss", "Training Loss\n[PSO Selected]",
     "Loss",      "darkorange"),
    (0, 3, arrays_sel,  means_sel,  max_ep_sel,
     "val_loss",   "Validation Loss\n[PSO Selected]",
     "Loss",      "darkorange"),
    (1, 0, arrays_full, means_full, max_ep_full,
     "train_acc",  "Training Accuracy\n[Full Features]",
     "Accuracy",  "steelblue"),
    (1, 1, arrays_full, means_full, max_ep_full,
     "val_acc",    "Validation Accuracy\n[Full Features]",
     "Accuracy",  "steelblue"),
    (1, 2, arrays_sel,  means_sel,  max_ep_sel,
     "train_acc",  "Training Accuracy\n[PSO Selected]",
     "Accuracy",  "darkorange"),
    (1, 3, arrays_sel,  means_sel,  max_ep_sel,
     "val_acc",    "Validation Accuracy\n[PSO Selected]",
     "Accuracy",  "darkorange"),
]

for (row, col, arrs, mns, max_ep,
     key, title, ylabel, color) in configs:
    ax       = axes[row, col]
    fold_dat = arrs[key]
    mean_dat = mns[key]
    epochs_x = np.arange(1, max_ep + 1)

    # Individual folds — light, thin, no visual bias
    for i in range(N_FOLDS):
        valid = ~np.isnan(fold_dat[i])
        ax.plot(epochs_x[valid], fold_dat[i, valid],
                color=color, alpha=0.25,
                linewidth=0.9,
                label=f"Fold {i+1}")

    # Mean curve — prominent black dashed
    valid_m = ~np.isnan(mean_dat)
    ax.plot(epochs_x[valid_m], mean_dat[valid_m],
            color='black', linewidth=2.0,
            linestyle='--', label='Mean', zorder=5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc='best',
              ncol=2, framealpha=0.6)

plt.tight_layout()
hist_path = os.path.join(
    figures_dir, "cnn_training_history_combined.png"
)
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Combined training history saved to: {hist_path}")


# =========================================
# FIGURE 2 — EPOCHS PER FOLD (both conditions)
# Shows training evolution and early stopping behavior
# Consistent with GRU and CNN-GRU epochs figure
# =========================================
fold_labels = [f"Fold {i}" for i in range(1, N_FOLDS+1)]
epochs_full = [h["epochs_run"] for h in hist_full]
epochs_sel  = [h["epochs_run"] for h in hist_sel]

xf    = np.arange(N_FOLDS)
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(xf - width/2, epochs_full, width,
       color='steelblue', alpha=0.85,
       label='Full Features (30)')
ax.bar(xf + width/2, epochs_sel, width,
       color='darkorange', alpha=0.85,
       label='PSO Selected (10)')
ax.axhline(y=EPOCHS, color='red', linestyle='--',
           linewidth=1.2, label=f'Max epochs ({EPOCHS})')
for i, (ef, es) in enumerate(zip(epochs_full, epochs_sel)):
    ax.text(i - width/2, ef + 0.8, str(ef),
            ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, es + 0.8, str(es),
            ha='center', va='bottom', fontsize=9)
ax.set_xticks(xf)
ax.set_xticklabels(fold_labels, fontsize=11)
ax.set_ylabel("Epochs Run", fontsize=12)
ax.set_title(
    "Training Epochs per Fold — CNN\n"
    "Early Stopping (patience=5) | Full vs PSO Features",
    fontsize=12
)
ax.legend(fontsize=10)
ax.set_ylim(0, EPOCHS + 15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
ep_path = os.path.join(figures_dir, "cnn_epochs_per_fold.png")
plt.savefig(ep_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Epochs per fold saved to: {ep_path}")


# =========================================
# FIGURE 3 — CONFUSION MATRICES (side by side)
# =========================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, title in zip(
        axes,
        [cm_full, cm_sel],
        ["CNN — Full Features (30)",
         "CNN — PSO Selected (10)"]):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12)
plt.suptitle(
    "Aggregated Confusion Matrices — CNN | 5-Fold CV",
    fontsize=13, y=1.02
)
plt.tight_layout()
cm_path = os.path.join(figures_dir, "cnn_confusion_matrices.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Confusion matrices saved to: {cm_path}")


# =========================================
# FIGURE 4 — ROC CURVES (both conditions)
# =========================================
fig, ax = plt.subplots(figsize=(6, 5))
for yt, yp, label, color in [
    (yt_full, yprob_full, "Full Features (30)", "steelblue"),
    (yt_sel,  yprob_sel,  "PSO Selected (10)",  "darkorange")
]:
    fpr, tpr, _ = roc_curve(yt, yp)
    auc_val = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{label} (AUC={auc_val:.4f})")
ax.plot([0,1],[0,1], 'k--', linewidth=1,
        label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — CNN | 5-Fold CV", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(figures_dir, "cnn_roc_curves.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC curves saved to: {roc_path}")


# =========================================
# FIGURE 5 — BAR COMPARISON of metrics
# =========================================
xb    = np.arange(len(metrics_labels))
width = 0.35
full_means = [summary_full[k]['mean'] for k in metrics_keys]
sel_means  = [summary_sel[k]['mean']  for k in metrics_keys]
full_stds  = [summary_full[k]['std']  for k in metrics_keys]
sel_stds   = [summary_sel[k]['std']   for k in metrics_keys]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(xb - width/2, full_means, width,
       yerr=full_stds, capsize=4,
       color='steelblue', alpha=0.85,
       label='Full Features (30)')
ax.bar(xb + width/2, sel_means, width,
       yerr=sel_stds, capsize=4,
       color='darkorange', alpha=0.85,
       label='PSO Selected (10)')
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    "CNN Performance: Full vs PSO-Selected Features\n"
    "5-Fold Stratified CV (Mean ± Std)", fontsize=12
)
ax.set_xticks(xb)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(0.88, 1.02)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
bar_path = os.path.join(figures_dir, "cnn_metric_comparison.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Metric comparison saved to: {bar_path}")


# =========================================
# FINAL SUMMARY
# =========================================
print(f"\n{'='*65}")
print(f"  READY SUMMARY — CNN")
print(f"  Evaluation: {N_FOLDS}-Fold Stratified Cross-Validation")
print(f"  Dataset: Breast Cancer Wisconsin "
      f"({X_raw.shape[0]} samples)")
print(f"{'='*65}")
print(f"\n  FULL FEATURE SET (30 features):")
for label, key in zip(metrics_labels, metrics_keys):
    print(f"    {label:<12} {summary_full[key]['mean']:.4f} "
          f"± {summary_full[key]['std']:.4f} "
          f"(CV={summary_full[key]['cv']:.2f}%)")
print(f"\n  PSO SELECTED FEATURES (10 features):")
for label, key in zip(metrics_labels, metrics_keys):
    print(f"    {label:<12} {summary_sel[key]['mean']:.4f} "
          f"± {summary_sel[key]['std']:.4f} "
          f"(CV={summary_sel[key]['cv']:.2f}%)")
print(f"\n  EPOCHS (Full): "
      f"{epochs_full} → Mean={np.mean(epochs_full):.1f}")
print(f"  EPOCHS (PSO):  "
      f"{epochs_sel}  → Mean={np.mean(epochs_sel):.1f}")
print(f"\n  STATISTICAL TEST (Paired t-test, α=0.05):")
print(f"    Accuracy: t={t_stat:.4f}, p={p_value:.4f} "
      f"{'→ significant' if p_value < 0.05 else '→ not significant'}")
print(f"    F1-Score: t={t_stat_f1:.4f}, p={p_value_f1:.4f} "
      f"{'→ significant' if p_value_f1 < 0.05 else '→ not significant'}")
print(f"{'='*65}")
print("\nExperiment completed successfully.")
print(f"Results available in: {results_dir}")
print("Thank you for using this research software.")