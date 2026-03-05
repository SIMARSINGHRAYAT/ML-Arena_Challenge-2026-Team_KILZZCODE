import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = r'c:\Users\Simar Singh Rayat\OneDrive\Desktop\ML_Arena\ML Challenge Dataset'
tr = pd.read_csv(os.path.join(BASE, 'TRAIN.csv'))
fcols = [c for c in tr.columns if c != 'Class']
X = tr[fcols].values
y = tr['Class'].values
cols = [f'F{i:02d}' for i in range(1, 48)]
df = pd.DataFrame(X, columns=cols)

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

Xp = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(df.values))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accs, precs, recs, f1s, rmses, fprs, lats = [], [], [], [], [], [], []

for train_idx, test_idx in cv.split(Xp, y):
    X_tr, X_te = Xp[train_idx], Xp[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    ens = VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=300, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=300, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1.0, scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1), eval_metric='logloss', random_state=42, n_jobs=-1)),
        ('lgbm', LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1.0, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1))
    ], voting='soft', n_jobs=-1)
    
    ens.fit(X_tr, y_tr)
    
    t0 = time.time()
    preds = ens.predict(X_te)
    probs = ens.predict_proba(X_te)[:, 1]
    inference_time = time.time() - t0
    
    latency_ms_per_sample = (inference_time / len(X_te)) * 1000
    
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds)
    rec = recall_score(y_te, preds)
    f1 = f1_score(y_te, preds, average='macro')
    # RMSE on probabilities for smoother evaluation of regression aspect
    rmse = np.sqrt(mean_squared_error(y_te, probs))
    
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    fpr = fp / (fp + tn)
    
    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
    rmses.append(rmse)
    fprs.append(fpr)
    lats.append(latency_ms_per_sample)

print(f"Accuracy: {np.mean(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f}")
print(f"Recall: {np.mean(recs):.4f}")
print(f"F1-Score: {np.mean(f1s):.4f}")
print(f"RMSE: {np.mean(rmses):.4f}")
print(f"FPR: {np.mean(fprs):.4f}")
print(f"Latency: {np.mean(lats):.4f} ms/sample")
