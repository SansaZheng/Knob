"""Self-supervised transaction-sequence model for TabFormer (CPU-sized).

Usage: python -m baselines.tabformer_seq --work SCRATCH [--minutes 15]

A causal GRU reads each card's transactions in time order.  Pretraining task:
predict the next transaction's MCC, amount bin, time-gap bin, chip mode and hour
from the history.  It is fitted on training-period transactions only and never
sees a fraud label.  For every transaction t it then exports
  h_t      - the 64-d state after reading transactions <= t of that card
  nll_*    - how surprising transaction t was given transactions < t (five scores)
which feed a gradient-boosted head (baselines.tabformer, variants seq / agg+seq).

This follows the recipe of public transaction foundation models (tokenise ->
pretrain on next-event prediction -> embeddings into a boosted head) at a scale
that trains in minutes on two CPU cores.  It is NOT a reproduction of NVIDIA's
29M-parameter model, Revolut PRAGMA or NeoSpace Cortex.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIELDS = ['mcc', 'amount', 'gap', 'chip', 'hour', 'weekday', 'state', 'error', 'city_changed', 'seen_merchant']
TARGETS = ['mcc', 'amount', 'gap', 'chip', 'hour']
D, L = 64, 256


def tokens(work):
    col = lambda name: np.load(work / 'cols' / f'{name}.npy')
    split = np.load(work / 'split.npy')
    mcc = col('mcc').astype(np.int64)
    _, mcc_id = np.unique(mcc, return_inverse=True)
    amount = col('log_amount')
    edges = np.unique(np.quantile(amount[split == 0][::50], np.linspace(0, 1, 33)[1:-1]))   # train period only
    gap = col('card_gap_log')
    state = col('state_code').astype(np.int64)
    err = sum(col(f'err_{e}') for e in ('bad_pin', 'insufficient_balance', 'technical_glitch', 'bad_card_number',
                                        'bad_cvv', 'bad_expiration', 'bad_zipcode'))
    tok = np.stack([mcc_id, np.searchsorted(edges, amount),
                    np.where(gap < 0, 0, 1 + np.clip(gap.astype(np.int64), 0, 15)),
                    col('chip_code').astype(np.int64), np.clip(col('hour').astype(np.int64), 0, 23),
                    col('weekday').astype(np.int64), np.where(state < 0, 0, np.minimum(state + 1, 63)),
                    (err > 0).astype(np.int64), col('city_changed').astype(np.int64),
                    (col('um_prior_n_log') > 0).astype(np.int64)], axis=1).astype(np.int16)
    sizes = [int(tok[:, j].max()) + 1 for j in range(tok.shape[1])]
    return tok, sizes, split


class Model(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.embed = nn.ModuleList(nn.Embedding(s, D) for s in sizes)
        self.gru = nn.GRU(D, D, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(D, sizes[FIELDS.index(t)]) for t in TARGETS)

    def forward(self, x):
        z = sum(e(x[..., j]) for j, e in enumerate(self.embed))
        h, _ = self.gru(z)
        return h

    def nll(self, h, x):
        """Per-position loss of transaction t+1 under the state after transaction t: [B, T-1, targets]."""
        return torch.stack([F.cross_entropy(head(h[:, :-1]).transpose(1, 2), x[:, 1:, FIELDS.index(t)].long(),
                                            reduction='none') for head, t in zip(self.heads, TARGETS)], dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', required=True)
    p.add_argument('--minutes', type=float, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    args = p.parse_args()
    work = Path(args.work)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    tok, sizes, split = tokens(work)
    card = np.load(work / 'card.npy')
    n = len(card)
    starts = np.flatnonzero(np.r_[True, card[1:] != card[:-1]]); ends = np.r_[starts[1:], n]

    # training segments: consecutive blocks of L training-period transactions of one card
    train_end = np.array([s + int((split[s:e] == 0).sum()) for s, e in zip(starts, ends)])   # train rows are a prefix
    seg = np.concatenate([np.arange(s, te - L + 1, L) for s, te in zip(starts, train_end) if te - s >= L])
    model = Model(sizes)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    begin, step, log = time.time(), 0, []
    order = np.concatenate([np.random.permutation(len(seg)) for _ in range(args.epochs)])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=len(order) // 128 + 1, pct_start=0.1)
    while time.time() - begin < args.minutes * 60 and (step + 1) * 128 <= len(order):
        batch = seg[order[step * 128:(step + 1) * 128]]
        x = torch.from_numpy(np.stack([tok[b:b + L] for b in batch]).astype(np.int64))
        loss = model.nll(model(x), x).mean(dim=(0, 1))
        opt.zero_grad(); loss.sum().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 25 == 0:
            log.append(dict(step=step, seconds=round(time.time() - begin), **{t: round(float(v.detach()), 4) for t, v in zip(TARGETS, loss)}))
            print(log[-1], flush=True)
    torch.save(model.state_dict(), work / 'seq_model.pt')

    # export: 256-transaction blocks, each preceded by up to 256 earlier transactions as warm-up
    out = np.lib.format.open_memmap(work / 'seq_features.npy', mode='w+', dtype=np.float16, shape=(n, D + len(TARGETS)))
    windows = [(max(s, b - L), b, min(b + L, e)) for s, e in zip(starts, ends) for b in range(s, e, L)]
    uniform = np.log([sizes[FIELDS.index(t)] for t in TARGETS]).astype(np.float32)
    model.eval(); begin = time.time()
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            group = windows[i:i + 256]
            width = max(e - c for c, _, e in group)
            x = np.zeros((len(group), width, len(FIELDS)), dtype=np.int64)
            for j, (c, b, e) in enumerate(group):
                x[j, :e - c] = tok[c:e]                     # right padding never affects earlier causal states
            xt = torch.from_numpy(x)
            h = model(xt); nll = model.nll(h, xt).numpy(); h = h.numpy()
            for j, (c, b, e) in enumerate(group):
                out[b:e, :D] = h[j, b - c:e - c]
                if b == c:                                  # first transaction of a card has no history
                    out[b, D:] = uniform; out[b + 1:e, D:] = nll[j, 0:e - c - 1]
                else:
                    out[b:e, D:] = nll[j, b - c - 1:e - c - 1]
            if (i // 256) % 40 == 0:
                print(f'export {i}/{len(windows)} windows, {time.time() - begin:.0f}s', flush=True)
    out.flush()
    info = dict(parameters=int(sum(p.numel() for p in model.parameters())), hidden=D, block=L, vocabulary=dict(zip(FIELDS, sizes)),
                pretraining_targets=TARGETS, training_segments=int(len(seg)), steps=step, batch=128,
                training_transactions_seen=int(step * 128 * L), training_transactions_available=int((split == 0).sum()),
                wall_minutes_cap=args.minutes, epochs=args.epochs, seed=args.seed, labels_used=False, loss_log=log)
    (work / 'seq_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k != 'loss_log'}, indent=2))


if __name__ == '__main__':
    main()
