"""Collect run records, apply the validation-only selection rule, write tables and figures.

Usage: python -m baselines.aggregate

Selection rule (fixed in this file before the summary was produced; test labels are not used):
  1. dataset x method: keep the feature variant with the highest mean validation AP over seeds.
  2. per dataset: the method with the highest mean validation AP.
  3. one overall method: among methods run on all four tasks, the lowest mean rank of
     validation AP across the four tasks (ties: higher mean validation AP).
Kronos rows are copied from Kronos_SupervisedAD records on the same test rows; they were
not re-run here and take no part in the selection.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import RESULTS

DATASETS = ['sapienza', 'sd-stock', 'sd-crypto', 'tabformer']
TITLES = {'sapienza': 'Sapienza 15S', 'sd-stock': 'SD-FMM stock', 'sd-crypto': 'SD-FMM crypto', 'tabformer': 'IBM TabFormer'}
GENERAL = ['rf', 'ada', 'xgb', 'lgbm']
KRONOS = {'sapienza': ('Sapienza · 15S', 'sapienza'), 'sd-stock': ('SD-FMM · stock', 'sd-stock'),
          'sd-crypto': ('SD-FMM · crypto', 'sd-crypto'), 'tabformer': ('IBM TabFormer', 'tabformer')}


def main():
    rows = []
    for path in sorted((RESULTS / 'runs').glob('*/*.json')):
        r = json.loads(path.read_text())
        rows.append(dict(dataset=r['dataset'], method=r['method'], variant=r['variant'], seed=r['seed'],
                         val_AP=r['validation']['AP'], val_F1=r['validation']['F1'],
                         **{f'test_{k}': r['test'][k] for k in ('AP', 'ROC_AUC', 'precision', 'recall', 'F1', 'TP', 'FP', 'FN', 'n', 'positives')}))
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / 'all_runs.csv', index=False)
    g = runs.groupby(['dataset', 'method', 'variant'])
    summary = g.agg(seeds=('seed', 'count'), val_AP=('val_AP', 'mean'), val_AP_sd=('val_AP', 'std'),
                    test_AP=('test_AP', 'mean'), test_AP_sd=('test_AP', 'std'), test_F1=('test_F1', 'mean'),
                    test_F1_sd=('test_F1', 'std'), test_precision=('test_precision', 'mean'),
                    test_recall=('test_recall', 'mean'), test_TP=('test_TP', 'mean'), test_FP=('test_FP', 'mean'),
                    test_n=('test_n', 'first'), test_positives=('test_positives', 'first')).reset_index()
    summary.to_csv(RESULTS / 'summary_by_variant.csv', index=False)
    best = summary.loc[summary.groupby(['dataset', 'method']).val_AP.idxmax()].copy()       # rule 1
    best['val_rank'] = best.groupby('dataset').val_AP.rank(ascending=False)

    reference = []
    old = pd.read_csv(RESULTS / 'kronos_reference/downloaded_20261103_metrics.csv')
    new = pd.read_csv(RESULTS / 'kronos_reference/repair_rounds_20261103_comparison.csv')
    for d, (old_name, new_name) in KRONOS.items():
        o = old[old.dataset == old_name].iloc[0]; n = new[new.dataset == new_name].iloc[0]
        reference.append(dict(dataset=d, method='kronos-mini (R0, recorded)', variant='reference', seeds=1, test_AP=o.AP,
                              test_F1=o.F1, test_precision=o.precision, test_recall=o.recall, test_TP=o.TP, test_FP=o.FP,
                              test_n=o.test_n, test_positives=o.positives))
        reference.append(dict(dataset=d, method='kronos-mini (repaired, recorded)', variant=n.selected_round, seeds=1,
                              test_AP=n.new_AP, test_F1=n.new_F1, test_precision=n.new_precision, test_recall=n.new_recall,
                              test_TP=n.new_TP, test_FP=n.new_FP, test_n=n.n, test_positives=n.positives))
    table = pd.concat([best, pd.DataFrame(reference)], ignore_index=True)
    table['dataset'] = pd.Categorical(table.dataset, DATASETS, ordered=True)
    table = table.sort_values(['dataset', 'test_AP'], ascending=[True, False])
    table.to_csv(RESULTS / 'summary.csv', index=False)

    per_dataset = {d: best[best.dataset == d].sort_values('val_AP', ascending=False).iloc[0] for d in DATASETS}   # rule 2
    general = best[best.method.isin(GENERAL)]
    ranks = general.assign(r=general.groupby('dataset').val_AP.rank(ascending=False)).groupby('method').agg(
        tasks=('dataset', 'count'), mean_val_rank=('r', 'mean'), mean_val_AP=('val_AP', 'mean'),
        mean_test_AP=('test_AP', 'mean'), mean_test_F1=('test_F1', 'mean')).reset_index()
    ranks = ranks[ranks.tasks == len(DATASETS)].sort_values(['mean_val_rank', 'mean_val_AP'], ascending=[True, False])
    ranks.to_csv(RESULTS / 'overall_ranking.csv', index=False)
    chosen = ranks.iloc[0].method                                                                             # rule 3
    selection = dict(rule=__doc__.split('Selection rule')[1].split('Kronos rows')[0].strip(),
                     overall_method=chosen, overall_ranking=ranks.to_dict('records'),
                     per_dataset_best_by_validation={d: dict(method=r.method, variant=r.variant, val_AP=r.val_AP,
                                                             test_AP=r.test_AP, test_F1=r.test_F1) for d, r in per_dataset.items()},
                     chosen_method_per_dataset={d: best[(best.dataset == d) & (best.method == chosen)].iloc[0][
                         ['variant', 'val_AP', 'test_AP', 'test_AP_sd', 'test_F1', 'test_F1_sd', 'test_precision', 'test_recall',
                          'test_TP', 'test_FP']].to_dict() for d in DATASETS})
    (RESULTS / 'selection.json').write_text(json.dumps(selection, indent=2, default=float) + '\n')

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, d in zip(axes, DATASETS):
        t = table[table.dataset == d].sort_values('test_AP')
        colors = ['#9a9a9a' if 'kronos' in m else '#c0392b' if m == chosen else '#2c6fbb' for m in t.method]
        labels = [f'{m} [{v}]' if 'kronos' not in m else m for m, v in zip(t.method, t.variant)]
        ax.barh(labels, t.test_AP, xerr=t.test_AP_sd.fillna(0), color=colors)
        for y, (ap, f1) in enumerate(zip(t.test_AP, t.test_F1)):
            ax.text(ap + 0.01, y, f'{ap:.3f} / F1 {f1:.3f}', va='center', fontsize=7.5)
        ax.set_xlim(0, 1.18); ax.set_title(f'{TITLES[d]}  (test AP, mean of seeds)', fontsize=10)
        ax.tick_params(axis='y', labelsize=7.5); ax.axvline(float(t.test_positives.iloc[0]) / float(t.test_n.iloc[0]), color='orange', ls='--', lw=1)
    fig.suptitle(f'Supervised baselines vs recorded Kronos-mini — red: method chosen by validation rule ({chosen}); grey: Kronos records; orange dashes: anomaly rate', fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS / 'test_AP_by_method.png', dpi=140)
    print(table[['dataset', 'method', 'variant', 'seeds', 'val_AP', 'test_AP', 'test_AP_sd', 'test_F1', 'test_precision', 'test_recall', 'test_TP', 'test_FP']].to_string(index=False))
    print(ranks.to_string(index=False)); print('chosen:', chosen)


if __name__ == '__main__':
    main()
