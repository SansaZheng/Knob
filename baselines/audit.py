"""Recompute test metrics from the saved test scores, independently of the training code path.

Usage: python -m baselines.audit [--work SCRATCH]     (SCRATCH is needed only for TabFormer labels)

Market tasks: every run has its test scores in results/test_scores (float32).
TabFormer: scores of the chosen method in results/test_scores/tabformer are checked
when the scratch folder with the prepared labels is given.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from .common import RESULTS
from .market import load


def main():
    p = argparse.ArgumentParser(); p.add_argument('--work')
    args = p.parse_args()
    labels = {d: np.concatenate([s[1] for s in load(d, 'test')]) for d in ('sapienza', 'sd-stock', 'sd-crypto')}
    if args.work:
        work = Path(args.work)
        labels['tabformer'] = np.load(work / 'y.npy')[np.load(work / 'split.npy') == 2]
    checked, worst_ap, count_mismatch = 0, 0.0, []
    for dataset, y in labels.items():
        for path in sorted((RESULTS / 'test_scores' / dataset).glob('*.npz')):
            run = json.loads((RESULTS / 'runs' / dataset / (path.stem + '.json')).read_text())['test']
            with np.load(path) as z:
                stored, threshold = z['score'], float(z['threshold'])
            score = stored.astype(np.float64)
            assert len(score) == len(y) == run['n'] and int(y.sum()) == run['positives']
            ap = average_precision_score(y, score)
            # scores were rounded when stored, so the threshold is rounded the same way before comparing
            alert = stored >= stored.dtype.type(threshold)
            tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
            worst_ap = max(worst_ap, abs(ap - run['AP'])); checked += 1
            if (tp, fp) != (run['TP'], run['FP']):
                count_mismatch.append(dict(file=f'{dataset}/{path.name}', recorded=[run['TP'], run['FP']], recomputed=[tp, fp]))
    out = dict(files_checked=checked, max_abs_AP_difference=worst_ap, alert_count_mismatches=count_mismatch,
               note='alerts are recomputed from the stored scores with the stored threshold')
    (RESULTS / 'audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
