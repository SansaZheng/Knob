# 实验数据 / Experiment data

本目录保存四项任务的**原始发布数据**(未改动数值和标签)及研究划分清单。数据由
[Kronos_SupervisedAD](https://github.com/SansaZheng/Kronos_SupervisedAD)(提交 `1f3f0aa`)迁入,划分与该仓库的 Kronos 实验完全相同,因此测试行一一对应。

| 任务 | 文件 | 规模 | 正类 | 来源与许可 |
| --- | --- | ---: | ---: | --- |
| Sapienza 15S | `sapienza/features_15S.csv.gz`、`sapienza/sapienza_groups_15S.csv` | 584,104 行,327 个事件组,85 个币种 | 317 个开始块 | La Morgia 等,ICCCN 2020;[作者仓库 d71250d](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset/tree/d71250d4cb055dde2d415c8cba38a0dcd6eb6e16);MIT |
| SD-FMM 股票 | `sd_fmm/stock/test_{data,label}_bs.npy` | 27 条序列 × 160 点 | 493 个点 | Dai 等,IPM 2026;[作者仓库 0d1a215](https://github.com/ys-Dai/SD-FMM/tree/0d1a2158ee849fdc31a127bd049e622174f5c893);MIT |
| SD-FMM 加密货币 | `sd_fmm/crypto/test_{data,label}.npy` | 337 条序列 × 160 点 | 337 个点 | 同上 |
| SD-FMM 合成异常 | `sd_fmm/synthetic/all_DTWsim_{data,label}.npy` | 999 条序列 × 160 点 | — | 同上(作者仓库 `systhesis/`);仅用于 SD-FMM 公开训练脚本 |
| IBM TabFormer | `tabformer/transactions.tgz.part000–006` | 24,386,900 笔交易,2,000 个用户,6,139 张卡 | 29,757 | Padhi 等,ICASSP 2021;[IBM/TabFormer ebb7cd6](https://github.com/IBM/TabFormer/tree/ebb7cd68ee1897599568107740bc452104bbbaf8);Apache-2.0 |

许可原文见 `notices/`。根目录代码的许可不替代各数据来源的许可。

## 划分

| 任务 | 划分方式 | train / validation / test | 正类 train / validation / test |
| --- | --- | --- | --- |
| Sapienza 15S | 按币种隔离(51/17/17 个币种),清单见 `sapienza_groups_15S.csv` | 327,103 / 111,451 / 145,550 行 | 190 / 50 / 77 |
| SD-FMM 股票 | 按序列内容分组,清单见 `sd_fmm/sd_fmm_sequences.csv` | 16 / 5 / 6 条序列 | 测试 99 个点(共 960 点) |
| SD-FMM 加密货币 | 同上 | 202 / 67 / 68 条序列 | 测试 68 个点(共 10,880 点) |
| IBM TabFormer | 按日历日时间划分,70% / 85% 行数分位处取整日边界 | 17,068,962 / 3,658,985 / 3,658,953 笔 | 20,886 / 4,417 / 4,454 |

TabFormer 的划分由 `baselines/tabformer_prepare.py` 从原始压缩包重算,测试段 3,658,953 笔、4,454 笔欺诈,与 Kronos_SupervisedAD 记录一致。

## 恢复 TabFormer 原始 CSV

```bash
cat data/tabformer/transactions.tgz.part* > transactions.tgz   # SHA-256 e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2
tar xzf transactions.tgz                                        # card_transaction.v1.csv,约 2.35 GB
```

实验脚本直接流式读取分卷,不需要先恢复。

## 使用时要知道的限制

- **SD-FMM 两个子集原文件都叫 `test`**,这里的 train/validation/test 是研究划分,不是论文的零样本测试,分数不能与论文表格直接比较。
- **SD-FMM 加密货币的标签位置几乎固定**:337 个开始点全部落在序列的第 98–101 个位置。只按位置打分、不看任何行情的对照,测试 AP 就有 0.29。带位置编码的模型在这个子集上的分数需要对照这个下限来读。
- **SD-FMM 序列已被发布者整段归一化到 [0,1]**,当前值本身就含有整段信息;“只用当前点”的特征在股票子集上已经有 0.6 的 AP。
- **Sapienza 标的是开始块**,不是操纵区间;12 个特征是作者的滚动工程特征,不是 OHLC。
- **TabFormer 是合成数据**,欺诈标签来自 IBM 的生成规则。
