import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = r'c:\Users\Simar Singh Rayat\OneDrive\Desktop\ML_Arena\ML Challenge Dataset'

print("Loading Data...")
tr = pd.read_csv(os.path.join(BASE, 'TRAIN.csv'))
fcols = [c for c in tr.columns if c != 'Class']
X = tr[fcols].values
y = tr['Class'].values
cols = [f'F{i:02d}' for i in range(1, 48)]
df = pd.DataFrame(X, columns=cols)

print("Engineering Features...")
df['rm'] = df.mean(axis=1)
df['rs'] = df.std(axis=1)
df['rn'] = df.min(axis=1)
df['rx'] = df.max(axis=1)
df['rr'] = df['rx'] - df['rn']
df['rmed'] = df.median(axis=1)
df['rsk'] = df.skew(axis=1)
df['rku'] = df.kurtosis(axis=1)
df['rsu'] = df[cols].sum(axis=1)
df['q25'] = df[cols].quantile(0.25, axis=1)
df['q75'] = df[cols].quantile(0.75, axis=1)
df['iqr'] = df['q75'] - df['q25']

df['a'] = df['F10'] * df['F11']
df['b'] = df['F10'] * df['F30']
df['c'] = df['F19'] * df['F20']
df['d'] = df['F31'] * df['F32']

print("Preprocessing...")
Xp = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(df.values))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Building Ensemble...")
ens = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=800, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=800, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=800, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1.0, scale_pos_weight=(y==0).sum()/max((y==1).sum(),1), eval_metric='logloss', random_state=42, n_jobs=-1)),
    ('lgbm', LGBMClassifier(n_estimators=800, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1.0, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1))
], voting='soft', n_jobs=-1)

print("Cross-Evaluating Accuracy and F1-Macro...")
scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
scores = cross_validate(ens, Xp, y, cv=cv, scoring=scoring, n_jobs=1)

acc = np.mean(scores['test_accuracy'])
f1 = np.mean(scores['test_f1_macro'])
print(f"CV Accuracy: {acc:.4f}")
print(f"CV F1-Macro: {f1:.4f}")

with open(os.path.join(BASE, "metrics_f1.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nF1-Macro: {f1:.4f}\n")
print("Done.")
