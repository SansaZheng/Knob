"""Tree-ensemble baselines on IBM TabFormer with the Kronos_SupervisedAD temporal split.

Usage: python -m baselines.tabformer --work SCRATCH --variants raw,agg --methods lgbm,xgb,rf,ada

Feature sets
  raw  - fields of the current transaction only (amount, time of day, chip, MCC, state, errors)
  agg  - raw + causal per-card / per-user / per-merchant history aggregates
  seq  - embeddings + surprise scores of the self-supervised sequence model (baselines.tabformer_seq)
  agg+seq - both
Training rows: every training-period fraud plus a row-ID-hash sample of training-period
normal transactions (label-dependent down-sampling of the majority class only).
Validation and test periods are scored completely, at their natural fraud rate.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from .common import SEEDS, record
from .tabformer_prepare import AGG, RAW

warnings.filterwarnings('ignore')
SEQ_DIM = 64
SEQ = [f'h{i}' for i in range(SEQ_DIM)] + ['nll_mcc', 'nll_amount', 'nll_gap', 'nll_chip', 'nll_hour']


def hash01(rows, seed=42):
    with np.errstate(over='ignore'):
        x = np.asarray(rows, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / (2. ** 53)


class Data:
    def __init__(self, work, variant):
        self.work = Path(work)
        self.y = np.load(self.work / 'y.npy'); self.split = np.load(self.work / 'split.npy')
        parts = variant.split('+')
        self.names = (RAW if 'raw' in parts else []) + (RAW + AGG if 'agg' in parts else [])
        self.seq = 'seq' in parts
        self.width = len(self.names) + (len(SEQ) if self.seq else 0)

    def matrix(self, rows):
        """rows must be sorted; columns are memory-mapped one at a time."""
        X = np.empty((len(rows), self.width), dtype=np.float32)
        for j, name in enumerate(self.names):
            X[:, j] = np.load(self.work / 'cols' / f'{name}.npy', mmap_mode='r')[rows]
        if self.seq:
            emb = np.load(self.work / 'seq_features.npy', mmap_mode='r')
            X[:, len(self.names):] = emb[rows]
        return X

    def sample(self, split, negatives, seed=42):
        rows = np.flatnonzero(self.split == split)
        y = self.y[rows]
        keep = (y == 1) | (hash01(rows, seed) < negatives / max((y == 0).sum(), 1))
        return rows[keep]

    def predict(self, model, split, chunk=1_000_000):
        rows = np.flatnonzero(self.split == split)
        out = np.empty(len(rows), dtype=np.float32)
        for b in range(0, len(rows), chunk):
            out[b:b + chunk] = model.predict_proba(self.matrix(rows[b:b + chunk]))[:, 1]
        return self.y[rows], out


def fit(method, seed, Xtr, ytr, Xva, yva):
    pos_weight = float(np.sqrt((ytr == 0).sum() / max(ytr.sum(), 1)))
    if method == 'lgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.1, num_leaves=63, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                                   scale_pos_weight=pos_weight, random_state=seed, n_jobs=2, verbose=-1)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(60, verbose=False)])
        return model, int(model.best_iteration_)
    if method == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.1, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=pos_weight,
                                  tree_method='hist', max_bin=128, eval_metric='aucpr',
                                  early_stopping_rounds=60, random_state=seed, n_jobs=2)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return model, int(model.best_iteration)
    if method == 'rf':     # same settings as the market tasks (La Morgia et al. 2020)
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed, n_jobs=2).fit(Xtr, ytr), 200
    if method == 'ada':    # same settings as the market tasks (La Morgia et al. 2023)
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100, learning_rate=0.5,
                                  random_state=seed).fit(Xtr, ytr), 100
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', required=True)
    p.add_argument('--variants', default='raw,agg')
    p.add_argument('--methods', default='lgbm,xgb,rf,ada')
    p.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    p.add_argument('--negatives', type=int, default=2_000_000)
    p.add_argument('--small-negatives', type=int, default=400_000, help='for the slower sklearn models')
    args = p.parse_args()
    for variant in args.variants.split(','):
        data = Data(args.work, variant)
        early = data.sample(1, 400_000)               # validation subset used only for early stopping
        Xes, yes = data.matrix(early), data.y[early]
        for method in args.methods.split(','):
            negatives = args.negatives if method in ('lgbm', 'xgb') else args.small_negatives
            rows = data.sample(0, negatives)
            Xtr, ytr = data.matrix(rows), data.y[rows]
            for seed in map(int, args.seeds.split(',')):
                start = time.time()
                model, rounds = fit(method, seed, Xtr, ytr, Xes, yes)
                yva, sva = data.predict(model, 1)
                yte, ste = data.predict(model, 2)
                record('tabformer', method, variant, seed, yva, sva, yte, ste, save_scores=False,
                       extra=dict(features=data.width, train_rows=int(len(ytr)), train_positives=int(ytr.sum()),
                                  train_negative_sample=negatives, rounds=rounds,
                                  fit_and_score_seconds=round(time.time() - start, 1)))
                keep = Path(args.work) / 'scores'; keep.mkdir(exist_ok=True)
                np.save(keep / f'{method}__{variant}__seed{seed}.npy', ste.astype(np.float32))
            del Xtr


if __name__ == '__main__':
    main()
