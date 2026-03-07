# -*- coding: utf-8 -*-
"""
Sentinel-63 v3 — Ultra-High-Accuracy Fault Detection Pipeline (Target ≥ 99.80%)
=================================================================================
Two-level stacking ensemble with:
  Level-1 :  XGBoost, LightGBM, CatBoost, Extra-Trees, Random Forest
  Level-2 :  Logistic Regression meta-learner
Extensive feature engineering, threshold optimisation.
"""

import os, sys, warnings, time, itertools
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────
# 1.  PATHS
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "TRAIN.csv")
TEST_PATH   = os.path.join(BASE_DIR, "TEST.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "FINAL.csv")

# ──────────────────────────────────────────────
# 2.  LOAD DATA
# ──────────────────────────────────────────────
print("=" * 70)
print("  Sentinel-63 v3 — Ultra-High-Accuracy Fault Detection")
print("=" * 70)
print("\n[1/8] Loading data ...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"      TRAIN shape : {train_df.shape}")
print(f"      TEST  shape : {test_df.shape}")

feature_cols = [c for c in train_df.columns if c != "Class"]
X_train_raw  = train_df[feature_cols].values.copy()
y_train      = train_df["Class"].values.copy()

test_ids   = test_df["ID"].values
X_test_raw = test_df[feature_cols].values.copy()

print(f"      Class 0: {(y_train==0).sum():,}   Class 1: {(y_train==1).sum():,}")

# ──────────────────────────────────────────────
# 3.  FEATURE ENGINEERING  (expanded)
# ──────────────────────────────────────────────
print("\n[2/8] Feature engineering ...")

def build_features(X, orig_n_features=47):
    """Return enriched feature matrix."""
    cols = [f"F{i:02d}" for i in range(1, orig_n_features + 1)]
    df = pd.DataFrame(X, columns=cols)

    # ---------- row-level statistics ----------
    df["row_mean"]   = df[cols].mean(axis=1)
    df["row_std"]    = df[cols].std(axis=1)
    df["row_min"]    = df[cols].min(axis=1)
    df["row_max"]    = df[cols].max(axis=1)
    df["row_range"]  = df["row_max"] - df["row_min"]
    df["row_median"] = df[cols].median(axis=1)
    df["row_skew"]   = df[cols].skew(axis=1)
    df["row_kurt"]   = df[cols].kurtosis(axis=1)
    df["row_sum"]    = df[cols].sum(axis=1)
    df["row_q10"]    = df[cols].quantile(0.10, axis=1)
    df["row_q25"]    = df[cols].quantile(0.25, axis=1)
    df["row_q75"]    = df[cols].quantile(0.75, axis=1)
    df["row_q90"]    = df[cols].quantile(0.90, axis=1)
    df["row_iqr"]    = df["row_q75"] - df["row_q25"]
    df["row_cv"]     = df["row_std"] / (df["row_mean"].abs() + 1e-8)
    df["row_energy"] = (df[cols] ** 2).sum(axis=1)
    df["row_rms"]    = np.sqrt(df["row_energy"] / len(cols))

    # ---------- group statistics ----------
    grp1 = [f"F{i:02d}" for i in range(1, 10)]   # F01-F09
    grp2 = [f"F{i:02d}" for i in range(10, 20)]  # F10-F19
    grp3 = [f"F{i:02d}" for i in range(20, 30)]  # F20-F29
    grp4 = [f"F{i:02d}" for i in range(30, 40)]  # F30-F39
    grp5 = [f"F{i:02d}" for i in range(40, 48)]  # F40-F47
    for gname, gcols in [("g1", grp1), ("g2", grp2), ("g3", grp3),
                         ("g4", grp4), ("g5", grp5)]:
        df[f"{gname}_mean"] = df[gcols].mean(axis=1)
        df[f"{gname}_std"]  = df[gcols].std(axis=1)
        df[f"{gname}_max"]  = df[gcols].max(axis=1)
        df[f"{gname}_min"]  = df[gcols].min(axis=1)
        df[f"{gname}_sum"]  = df[gcols].sum(axis=1)

    # ---------- top-feature pairwise interactions ----------
    top_feats = ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17",
                 "F18", "F19", "F30", "F31", "F32"]
    for a, b in itertools.combinations(top_feats, 2):
        df[f"{a}x{b}"]  = df[a] * df[b]
        df[f"{a}d{b}"]  = df[a] / (df[b].abs() + 1e-8)

    # ---------- log1p transforms of large-range features ----------
    large_range = ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17",
                   "F18", "F19", "F30", "F31", "F32", "F33", "F34", "F35",
                   "F36", "F37", "F38"]
    for c in large_range:
        df[f"{c}_log1p"] = np.log1p(df[c].abs()) * np.sign(df[c])
        df[f"{c}_sq"]    = df[c] ** 2

    # ---------- difference / ratio features for consecutive cols ----------
    for i in range(1, orig_n_features):
        c1, c2 = f"F{i:02d}", f"F{i+1:02d}"
        df[f"diff_{c1}_{c2}"] = df[c1] - df[c2]

    # ---------- absolute value features for F39-F47 (centered near 0) ----------
    near_zero = [f"F{i:02d}" for i in range(39, 48)]
    for c in near_zero:
        df[f"{c}_abs"] = df[c].abs()

    return df.values, list(df.columns)

X_train_fe, feat_names = build_features(X_train_raw)
X_test_fe, _           = build_features(X_test_raw)
print(f"      Features : {X_train_raw.shape[1]} → {X_train_fe.shape[1]}")

# ──────────────────────────────────────────────
# 4.  PREPROCESSING
# ──────────────────────────────────────────────
print("\n[3/8] Preprocessing ...")
imputer = SimpleImputer(strategy="median")
scaler  = RobustScaler()

X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train_fe))
X_test_proc  = scaler.transform(imputer.transform(X_test_fe))

# Replace any remaining inf/nan
X_train_proc = np.nan_to_num(X_train_proc, nan=0.0, posinf=0.0, neginf=0.0)
X_test_proc  = np.nan_to_num(X_test_proc,  nan=0.0, posinf=0.0, neginf=0.0)

print(f"      Shape after processing : {X_train_proc.shape}")

# ──────────────────────────────────────────────
# 5.  FEATURE IMPORTANCE PRE-SCREEN (quick RF)
# ──────────────────────────────────────────────
print("\n[4/8] Feature importance pre-screening ...")
quick_rf = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
quick_rf.fit(X_train_proc, y_train)
importances = quick_rf.feature_importances_

# Keep top features (importance > 25th percentile)
threshold = np.percentile(importances, 25)
keep_mask = importances >= threshold
n_keep = keep_mask.sum()
print(f"      Keeping {n_keep} / {X_train_proc.shape[1]} features (threshold={threshold:.6f})")

X_train_sel = X_train_proc[:, keep_mask]
X_test_sel  = X_test_proc[:, keep_mask]

# ──────────────────────────────────────────────
# 6.  BUILD STACKING ENSEMBLE
# ──────────────────────────────────────────────
print("\n[5/8] Building stacking ensemble ...")

pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# --- Level-1 estimators ---
xgb1 = XGBClassifier(
    n_estimators=2000, max_depth=9, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.6,
    reg_alpha=0.05, reg_lambda=2.0, gamma=0.1,
    min_child_weight=3, scale_pos_weight=pos_weight,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)

xgb2 = XGBClassifier(
    n_estimators=1500, max_depth=7, learning_rate=0.03,
    subsample=0.75, colsample_bytree=0.7, colsample_bylevel=0.7,
    reg_alpha=0.1, reg_lambda=1.5, gamma=0.05,
    min_child_weight=5, scale_pos_weight=pos_weight,
    eval_metric="logloss", random_state=123, n_jobs=-1,
)

lgbm1 = LGBMClassifier(
    n_estimators=2000, max_depth=9, learning_rate=0.02, num_leaves=127,
    subsample=0.8, colsample_bytree=0.6,
    reg_alpha=0.05, reg_lambda=2.0, min_child_samples=10,
    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
)

lgbm2 = LGBMClassifier(
    n_estimators=1500, max_depth=7, learning_rate=0.03, num_leaves=63,
    subsample=0.75, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.5, min_child_samples=20,
    is_unbalance=True, random_state=123, n_jobs=-1, verbose=-1,
)

cat1 = CatBoostClassifier(
    iterations=2000, depth=9, learning_rate=0.02,
    l2_leaf_reg=3.0, random_strength=0.5, bagging_temperature=0.8,
    auto_class_weights="Balanced", random_seed=42,
    verbose=0, thread_count=-1,
)

cat2 = CatBoostClassifier(
    iterations=1500, depth=7, learning_rate=0.03,
    l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
    auto_class_weights="Balanced", random_seed=123,
    verbose=0, thread_count=-1,
)

rf = RandomForestClassifier(
    n_estimators=2000, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced_subsample",
    random_state=42, n_jobs=-1,
)

et = ExtraTreesClassifier(
    n_estimators=2000, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", class_weight="balanced_subsample",
    random_state=42, n_jobs=-1,
)

level1 = [
    ("xgb1",  xgb1),
    ("xgb2",  xgb2),
    ("lgbm1", lgbm1),
    ("lgbm2", lgbm2),
    ("cat1",  cat1),
    ("cat2",  cat2),
    ("rf",    rf),
    ("et",    et),
]

# --- Level-2 meta-learner ---
meta = LogisticRegression(
    C=1.0, max_iter=1000, solver="lbfgs", random_state=42, n_jobs=-1,
)

stacking_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stack = StackingClassifier(
    estimators=level1,
    final_estimator=meta,
    cv=stacking_cv,
    stack_method="predict_proba",
    passthrough=False,
    n_jobs=1,           # each base learner already uses n_jobs=-1
)

print(f"      Level-1 models : {[n for n, _ in level1]}")
print(f"      Level-2 model  : LogisticRegression")

# ──────────────────────────────────────────────
# 7.  MANUAL CROSS-VALIDATION  (detailed metrics)
# ──────────────────────────────────────────────
print("\n[6/8] Running 5-fold stratified CV with full metrics ...")
eval_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics = {"acc": [], "prec": [], "rec": [], "f1": [],
           "rmse": [], "fpr": [], "lat": []}

t0_cv = time.time()
for fold, (tr_idx, va_idx) in enumerate(eval_cv.split(X_train_sel, y_train), 1):
    Xtr, Xva = X_train_sel[tr_idx], X_train_sel[va_idx]
    ytr, yva = y_train[tr_idx], y_train[va_idx]

    fold_t0 = time.time()
    stack_fold = StackingClassifier(
        estimators=level1,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                           solver="lbfgs", random_state=42,
                                           n_jobs=-1),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )
    stack_fold.fit(Xtr, ytr)
    fold_fit_time = time.time() - fold_t0

    # --- predict & threshold search ---
    t_inf = time.time()
    probs = stack_fold.predict_proba(Xva)[:, 1]
    inf_time = time.time() - t_inf

    # Optimise threshold for F1
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.70, 0.005):
        prd = (probs >= thr).astype(int)
        f = f1_score(yva, prd, average="macro")
        if f > best_f1:
            best_f1, best_thr = f, thr

    preds = (probs >= best_thr).astype(int)

    acc  = accuracy_score(yva, preds)
    prec = precision_score(yva, preds, zero_division=0)
    rec  = recall_score(yva, preds, zero_division=0)
    f1   = f1_score(yva, preds, average="macro")
    rmse = np.sqrt(mean_squared_error(yva, probs))
    tn, fp, fn, tp = confusion_matrix(yva, preds).ravel()
    fpr  = fp / (fp + tn)
    lat  = (inf_time / len(Xva)) * 1000

    metrics["acc"].append(acc)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["f1"].append(f1)
    metrics["rmse"].append(rmse)
    metrics["fpr"].append(fpr)
    metrics["lat"].append(lat)

    print(f"  Fold {fold}: Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  "
          f"F1={f1:.4f}  RMSE={rmse:.4f}  FPR={fpr:.4f}  "
          f"thr={best_thr:.3f}  fit={fold_fit_time:.0f}s")

elapsed_cv = time.time() - t0_cv
print(f"\n  --- CV Summary (mean ± std) ---")
for m in ["acc", "prec", "rec", "f1", "rmse", "fpr", "lat"]:
    vals = np.array(metrics[m])
    print(f"  {m.upper():>5s}: {vals.mean():.4f} ± {vals.std():.4f}")
print(f"  Total CV time: {elapsed_cv:.0f}s")

if np.mean(metrics["acc"]) >= 0.9980:
    print("\n  >>> TARGET ACHIEVED  (>= 99.80%) <<<")
else:
    print(f"\n  Current accuracy: {np.mean(metrics['acc']):.4f} — continuing with best model")

# ──────────────────────────────────────────────
# 8.  TRAIN FINAL MODEL ON ALL DATA
# ──────────────────────────────────────────────
print("\n[7/8] Training final stacking model on full training set ...")
t0 = time.time()
stack.fit(X_train_sel, y_train)
print(f"      Fit time : {time.time()-t0:.0f}s")

# Final threshold from CV: use the mean best threshold
# (but on full training, use a quick scan too)
train_probs = stack.predict_proba(X_train_sel)[:, 1]
best_thr_final, best_f1_final = 0.5, 0.0
for thr in np.arange(0.30, 0.70, 0.005):
    prd = (train_probs >= thr).astype(int)
    f = f1_score(y_train, prd, average="macro")
    if f > best_f1_final:
        best_f1_final, best_thr_final = f, thr

train_preds = (train_probs >= best_thr_final).astype(int)
train_acc = accuracy_score(y_train, train_preds)
print(f"      Best threshold   : {best_thr_final:.3f}")
print(f"      Train Accuracy   : {train_acc:.4f}")
print(f"      Train F1 (macro) : {f1_score(y_train, train_preds, average='macro'):.4f}")
print("\n      Classification Report (train):")
print(classification_report(y_train, train_preds,
                            target_names=["Normal (0)", "Faulty (1)"]))

# ──────────────────────────────────────────────
# 9.  PREDICT TEST & SAVE
# ──────────────────────────────────────────────
print("[8/8] Generating predictions on TEST set ...")
test_probs = stack.predict_proba(X_test_sel)[:, 1]
test_preds = (test_probs >= best_thr_final).astype(int)

submission = pd.DataFrame({"ID": test_ids, "CLASS": test_preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\n[DONE] Submission saved to : {OUTPUT_PATH}")
print(f"       Rows    : {len(submission)}")
print(f"       Class 0 : {(test_preds==0).sum()}")
print(f"       Class 1 : {(test_preds==1).sum()}")
print(f"       Preview :")
print(submission.head(10).to_string(index=False))
print("\n" + "=" * 70)
print("  DONE — submit FINAL.csv for evaluation")
print("=" * 70)
