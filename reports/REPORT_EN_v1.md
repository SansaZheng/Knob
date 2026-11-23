# Supervised baselines on four financial-transaction tasks · English v1

Version: EN v1 (2026-11-23). Chinese version: [REPORT_CN_v1.md](REPORT_CN_v1.md).

## Conclusion

**The selected method is XGBoost.** Of the four general methods run on all four tasks (random forest, AdaBoost, XGBoost, LightGBM), it ranks first on validation AP in every task. Selection used validation data only.

- On Sapienza 15S, SD-FMM crypto and TabFormer, XGBoost's test AP is above the recorded Kronos-mini (repaired) result.
- On SD-FMM stock, Kronos-mini (repaired) has the higher AP (0.630 vs 0.602) and XGBoost the slightly higher F1 (0.617 vs 0.590). The test split has six sequences, so this gap is not stable.
- On TabFormer every method has a low test AP (best 0.078) because fraud in the test period looks different from the training period (Finding 1). This task is not solved.

## Results

Test metrics, mean of three seeds (42, 13, 2026); ± is the sample standard deviation. Thresholds are fixed by validation F1. Kronos rows are single-seed records copied from Kronos_SupervisedAD (R0 = first version, repaired = the recipe chosen after its repair rounds); they were not re-run here.

| Task (test size, anomalies) | Method | Features | Test AP | Test F1 | TP / FP |
| --- | --- | --- | ---: | ---: | ---: |
| **Sapienza 15S** (145,550 blocks, 77) | **XGBoost** | publisher features | **0.965 ± 0.003** | 0.910 | 67.3 / 3.7 |
| | AdaBoost (La Morgia 2023 settings) | + causal rolling | 0.954 ± 0.001 | 0.898 | 67.3 / 5.7 |
| | Random forest (La Morgia 2020 settings) | publisher features | 0.941 ± 0.002 | 0.898 | 68.7 / 7.3 |
| | Kronos-mini repaired (recorded) | — | 0.912 | 0.857 | 63 / 7 |
| | Kronos-mini R0 (recorded) | — | 0.752 | 0.701 | 48 / 12 |
| | LightGBM | + causal rolling | 0.677 ± 0.075 | 0.784 | 67.3 / 27.7 |
| | Anomaly Transformer (unsupervised) | publisher features | 0.019 ± 0.029 | 0.009 | 1.7 / 563 |
| **SD-FMM stock** (960 points, 99) | Kronos-mini repaired (recorded) | — | **0.630** | 0.590 | 51 / 23 |
| | SD-FMM public script + fine-tune | full window | 0.616 ± 0.052 | 0.606 | 52.7 / 19.7 |
| | Random forest | current value | 0.603 ± 0.002 | 0.615 | 57.0 / 29.3 |
| | **XGBoost** | current value | 0.602 ± 0.015 | **0.617** | 59.3 / 34.0 |
| | SD-FMM public script (synthetic only) | full window | 0.590 ± 0.037 | 0.550 | 44.3 / 17.7 |
| | Kronos-mini R0 (recorded) | — | 0.578 | 0.609 | 60 / 38 |
| | LightGBM | current value | 0.575 ± 0.005 | 0.615 | 57.3 / 30.7 |
| | AdaBoost | current value | 0.553 | 0.610 | 54 / 24 |
| **SD-FMM crypto** (10,880 points, 68) | SD-FMM public script + fine-tune | full window | **0.675 ± 0.022** | **0.614** | 46.3 / 37.0 |
| | AdaBoost | causal rolling | 0.669 ± 0.000 | 0.593 | 43 / 34 |
| | **XGBoost** | causal rolling | 0.657 ± 0.014 | 0.609 | 36.3 / 15.0 |
| | Random forest | causal rolling | 0.653 ± 0.010 | 0.598 | 35.7 / 16.0 |
| | LightGBM | causal rolling | 0.604 ± 0.040 | 0.607 | 35.0 / 12.3 |
| | Kronos-mini repaired (recorded) | — | 0.480 | 0.431 | 33 / 52 |
| | Kronos-mini R0 (recorded) | — | 0.357 | 0.403 | 26 / 35 |
| | Position-only control | no market data | 0.291 | 0.363 | 37 / 99 |
| | SD-FMM public script (synthetic only) | full window | 0.162 ± 0.014 | 0.234 | 13.0 / 30.7 |
| **IBM TabFormer** (3,658,953 transactions, 4,454) | **XGBoost** | aggregates + sequence embeddings | **0.078 ± 0.009** | **0.059** | 159 / 763 |
| | Random forest | aggregates | 0.061 ± 0.005 | 0.046 | 170 / 2,842 |
| | XGBoost | aggregates | 0.058 ± 0.006 | 0.045 | 124 / 905 |
| | AdaBoost (one seed) | aggregates | 0.040 | 0.042 | 129 / 1,498 |
| | LightGBM | aggregates | 0.030 ± 0.009 | 0.036 | 156 / 4,743 |
| | XGBoost | current transaction only | 0.017 ± 0.003 | 0.027 | 73 / 772 |
| | XGBoost | sequence embeddings only | 0.008 ± 0.001 | 0.019 | 63 / 2,194 |
| | Kronos-mini repaired (recorded) | global time stream | 0.007 | 0.009 | 21 / 31 |
| | Kronos-mini R0 (recorded) | global time stream | 0.001 | 0.002 | 18 / 13,559 |

Under rule 1 the LightGBM variant selected on TabFormer is "current transaction only" (validation AP 0.537, test AP 0.012); the aggregates row is listed for comparison. TabFormer validation AP: XGBoost 0.770 (aggregates + embeddings), 0.753 (aggregates); random forest 0.374; AdaBoost 0.566; Kronos-mini repaired 0.234.

## Selection rule

Fixed in `baselines/aggregate.py`; test labels are not read.

1. Per task and method, keep the feature variant with the highest seed-mean validation AP.
2. Per task, take the method with the highest validation AP.
3. One overall method: among methods run on all four tasks, the lowest mean rank of validation AP.

| Method | Mean validation rank | Mean validation AP | Mean test AP | Mean test F1 |
| --- | ---: | ---: | ---: | ---: |
| **XGBoost** | **1.00** | 0.788 | 0.576 | 0.549 |
| Random forest | 2.75 | 0.676 | 0.565 | 0.539 |
| AdaBoost | 3.00 | 0.681 | 0.554 | 0.536 |
| LightGBM | 3.25 | 0.632 | 0.467 | 0.509 |

Per-task winners by validation AP: XGBoost on Sapienza, SD-FMM crypto and TabFormer; the fine-tuned SD-FMM public script on SD-FMM stock (validation 0.704, test 0.616). The cross-task mean AP is shown only to explain the ranking; the tasks have different units and anomaly rates.

## What the baselines are

- **Random forest / AdaBoost**: the settings in the Sapienza authors' `classifier.py`, unchanged across tasks.
- **XGBoost / LightGBM**: one fixed parameter set, early stopping on validation.
- **Feature variants**: market tasks use publisher features, with or without causal rolling statistics (current and earlier blocks only, no absolute position). TabFormer uses current-transaction fields, per-card / per-user / per-merchant history aggregates, sequence-model embeddings, and combinations. User, card and merchant IDs and past labels are not features.
- **TabFormer sequence model**: a causal GRU with about 54,000 parameters, pretrained without labels on next-transaction prediction over training-period transactions (3 epochs). It exports a 64-d state and five surprise scores per transaction to XGBoost. It is a CPU-sized version of the public transaction-foundation-model recipe, **not** a reproduction of NVIDIA TFM, Revolut PRAGMA or NeoSpace Cortex.
- **SD-FMM public script**: network and losses from the authors' `train.py`. Only 999 synthetic sequences were released (the script slices 2,001), so 799/200 is used; a second variant fine-tunes on the real research-train split.
- **Anomaly Transformer**: a small model (d_model 64, 3 layers) with the original association discrepancy and minimax training, fitted without labels; validation labels set the threshold only; no point adjustment.

## Findings

1. **Fraud in the TabFormer test period is different.** In the training (1991–2015) and validation (2015-12 to 2018-01) periods about 71 % and 78 % of fraud is online. In the test period (2018-01 to 2020-02) online is 2.9 %, chip transactions are 87.7 %, and 96.6 % occur at non-US merchant locations (8.7 % in training). XGBoost therefore scores 0.77 AP on validation and 0.078 on test; the Kronos record (0.234 → 0.007) shows the same drop. This is a post-hoc diagnosis with test labels, used for explanation only. See `results/tabformer_meta/period_shift.json`.
2. **Per-card history is the most useful information on TabFormer.** XGBoost scores 0.017 with current fields, 0.058 with aggregates, 0.078 with embeddings added; embeddings alone give 0.008. The Kronos record used a global time stream rather than per-card history, which may partly explain its low score; this was not tested separately.
3. **SD-FMM crypto has a position shortcut.** All 337 start points sit at positions 98–101, so scoring by position alone gives 0.291 test AP. Kronos-mini R0 (0.357) is close to that floor. The tree models' causal features carry no position and reach 0.65–0.67.
4. **Methods are bunched at 0.55–0.63 on SD-FMM stock.** Trees on the current normalised value alone reach 0.60 because the released series are normalised over the whole window. Six test sequences cannot separate these methods.
5. **LightGBM is unstable with these fixed settings.** On Sapienza its test AP swings from 0.16 to 0.60 across seeds on publisher features (190 training positives). XGBoost does not show this. LightGBM was not tuned separately.
6. **The unsupervised Anomaly Transformer is near chance under point-wise scoring**: 0.019 on Sapienza (anomaly rate 0.0005) and at or below the anomaly rate on both SD-FMM tasks. This is not a reproduction of Chadalapaka et al., whose segmentation and evaluation differ.
7. **SD-FMM trained on synthetic data alone reaches only 0.162 on crypto**, below the position control; fine-tuning lifts it to 0.675. On stock the synthetic-only variant already reaches 0.590.

## Limitations

- Kronos numbers are copied single-seed records; a strict comparison needs Kronos retrained on the same seeds in the same environment.
- TabFormer trees used all training-period fraud plus about 2 million hash-sampled normal transactions (about 400,000 for random forest and AdaBoost); the Kronos record used about 500,000 label-independent training targets. Training volume differs.
- On TabFormer, LightGBM was not run with sequence embeddings and AdaBoost used one seed (it has no randomness). XGBoost with aggregates only (validation AP 0.753) still ranks first, so the overall ranking does not change.
- SD-FMM splits are research splits of files the publisher names `test`; scores are not comparable to the paper. The SD-FMM network and the Anomaly Transformer read the whole window (offline); the trees' causal features and the Kronos backbone see the past only.
- No hyperparameter search was done. The claim is that XGBoost is the most reliable under these fixed settings, not that this is its ceiling.
- Test metrics were visible in run logs before selection. The rule is mechanical and reads validation only, but this is not a blind test.

## Audit

`python -m baselines.audit` recomputed AP and alert counts for 99 saved score files: all alert counts match, largest AP difference 1.7e-4 (float32 storage). For TabFormer only the selected method's test scores are stored (XGBoost, aggregates + embeddings, three seeds); the other TabFormer runs have metric records only.
