# -*- coding: utf-8 -*-
"""
Sentinel-63 - High-Accuracy Fault Detection Pipeline  (Target >= 98%)
======================================================================
Uses a Voting Ensemble of:
  - XGBoost
  - LightGBM
  - Random Forest
  - Extra Trees
with feature engineering, StandardScaler, and stratified cross-validation.
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------------------------
# 1.  PATHS
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "TRAIN.csv")
TEST_PATH   = os.path.join(BASE_DIR, "TEST.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "FINAL.csv")

# -----------------------------------------------
# 2.  LOAD DATA
# -----------------------------------------------
print("=" * 60)
print("  Sentinel-63 - High-Accuracy Fault Detection Pipeline")
print("=" * 60)
print("\n[1/7] Loading data ...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"      TRAIN shape : {train_df.shape}")
print(f"      TEST  shape : {test_df.shape}")

feature_cols = [c for c in train_df.columns if c != "Class"]
X_train = train_df[feature_cols].values
y_train = train_df["Class"].values

test_ids = test_df["ID"].values
X_test   = test_df[feature_cols].values

# -----------------------------------------------
# 3.  DATA INSPECTION
# -----------------------------------------------
print("\n[2/7] Data inspection ...")
n_missing = np.isnan(X_train).sum() + np.isnan(X_test).sum()
print(f"      Total missing values : {n_missing}")
class_counts = pd.Series(y_train).value_counts()
print(f"      Class distribution:")
for cls, cnt in class_counts.items():
    pct = cnt / len(y_train) * 100
    print(f"        Class {cls}: {cnt:,}  ({pct:.1f}%)")

# -----------------------------------------------
# 4.  FEATURE ENGINEERING
# -----------------------------------------------
print("\n[3/7] Feature engineering ...")

def add_features(X):
    """Add statistical aggregate features to boost model performance."""
    orig_cols = [f"F{i:02d}" for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=orig_cols)

    # Row-level statistics
    df["row_mean"]   = df.mean(axis=1)
    df["row_std"]    = df.std(axis=1)
    df["row_min"]    = df.min(axis=1)
    df["row_max"]    = df.max(axis=1)
    df["row_range"]  = df["row_max"] - df["row_min"]
    df["row_median"] = df.median(axis=1)
    df["row_skew"]   = df.skew(axis=1)
    df["row_kurt"]   = df.kurtosis(axis=1)
    df["row_sum"]    = df[orig_cols].sum(axis=1)
    df["row_q25"]    = df[orig_cols].quantile(0.25, axis=1)
    df["row_q75"]    = df[orig_cols].quantile(0.75, axis=1)
    df["row_iqr"]    = df["row_q75"] - df["row_q25"]

    # Pairwise interactions for high-variance features
    df["F10_x_F11"] = df["F10"] * df["F11"]
    df["F10_x_F30"] = df["F10"] * df["F30"]
    df["F19_x_F20"] = df["F19"] * df["F20"]
    df["F31_x_F32"] = df["F31"] * df["F32"]

    return df.values

X_train_fe = add_features(X_train)
X_test_fe  = add_features(X_test)
print(f"      Features after engineering : {X_train_fe.shape[1]}  (was {X_train.shape[1]})")

# -----------------------------------------------
# 5.  PREPROCESSING
# -----------------------------------------------
print("\n[4/7] Preprocessing (impute + scale) ...")
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train_fe))
X_test_proc  = scaler.transform(imputer.transform(X_test_fe))

# -----------------------------------------------
# 6.  BUILD ENSEMBLE
# -----------------------------------------------
print("\n[5/7] Building ensemble models ...")

# Try to use XGBoost and LightGBM if available
try:
    from xgboost import XGBClassifier
    xgb_available = True
    print("      [OK] XGBoost available")
except ImportError:
    xgb_available = False
    print("      [--] XGBoost not installed - using GradientBoosting fallback")

try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
    print("      [OK] LightGBM available")
except ImportError:
    lgbm_available = False
    print("      [--] LightGBM not installed - using ExtraTrees fallback")

# -- Individual estimators --
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

et = ExtraTreesClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

estimators = [("rf", rf), ("et", et)]

if xgb_available:
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    estimators.append(("xgb", xgb))
else:
    gb = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    estimators.append(("gb", gb))

if lgbm_available:
    lgbm = LGBMClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    estimators.append(("lgbm", lgbm))

print(f"      Ensemble members: {[name for name, _ in estimators]}")

ensemble = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=-1,
)

# -----------------------------------------------
# 7.  CROSS-VALIDATION
# -----------------------------------------------
print("\n[6/7] Running 5-fold stratified cross-validation ...")
t0 = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X_train_proc, y_train, cv=cv,
                            scoring="accuracy", n_jobs=1)
elapsed = time.time() - t0

print(f"      CV Accuracy per fold : {np.round(cv_scores, 4)}")
print(f"      Mean CV Accuracy     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"      Time elapsed         : {elapsed:.1f}s")

if cv_scores.mean() >= 0.98:
    print("      >>> TARGET ACHIEVED  (>= 98%) <<<")
else:
    print(f"      WARNING: Below 98% target - consider further tuning")

# -----------------------------------------------
# 8.  TRAIN ON FULL DATA & PREDICT
# -----------------------------------------------
print("\n[7/7] Training final model on full training set ...")
t0 = time.time()
ensemble.fit(X_train_proc, y_train)
elapsed = time.time() - t0
print(f"      Fit time : {elapsed:.1f}s")

train_preds = ensemble.predict(X_train_proc)
train_acc = accuracy_score(y_train, train_preds)
print(f"      Train Accuracy : {train_acc:.4f}")
print("\n      Classification Report (train):")
print(classification_report(y_train, train_preds, target_names=["Normal (0)", "Faulty (1)"]))

print("Generating predictions on TEST set ...")
test_preds = ensemble.predict(X_test_proc)

# -----------------------------------------------
# 9.  SAVE SUBMISSION
# -----------------------------------------------
submission = pd.DataFrame({"ID": test_ids, "CLASS": test_preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\n[DONE] Submission saved to : {OUTPUT_PATH}")
print(f"       Rows   : {len(submission)}")
print(f"       Preview:")
print(submission.head(10).to_string(index=False))
print("\n" + "=" * 60)
print("  DONE - submit FINAL.csv for evaluation")
print("=" * 60)
