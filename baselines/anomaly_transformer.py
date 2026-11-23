"""Compact Anomaly Transformer (Xu et al., ICLR 2022) for the three market tasks.

Usage: python -m baselines.anomaly_transformer --dataset sapienza|sd-stock|sd-crypto

Unsupervised: fitted on the training sequences without labels, as in Chadalapaka
et al. (2022) on the Sapienza data.  Anomaly score = softmax(-association
discrepancy) x reconstruction error inside each window.  Labels are used only to
pick the alert threshold on validation.  CPU-sized model (d_model 64, 3 layers);
the association mechanism and minimax training follow the original paper.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SEEDS, record
from .market import load


class AnomalyAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.s, self.o = (nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d),
                                                  nn.Linear(d, heads), nn.Linear(d, d))

    def forward(self, x, dist):
        b, t, d = x.shape
        shape = lambda z: z.view(b, t, self.h, -1).transpose(1, 2)
        q, k, v = shape(self.q(x)), shape(self.k(x)), shape(self.v(x))
        series = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d / self.h), dim=-1)
        sigma = torch.pow(3, torch.sigmoid(self.s(x) * 5) + 1e-5) - 1          # [b,t,h]
        sigma = sigma.transpose(1, 2).unsqueeze(-1)                             # [b,h,t,1]
        prior = torch.exp(-dist ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
        out = (series @ v).transpose(1, 2).reshape(b, t, d)
        return self.o(out), series, prior


class Model(nn.Module):
    def __init__(self, features, window, d=64, heads=4, layers=3):
        super().__init__()
        self.embed = nn.Linear(features, d)
        pe = torch.zeros(window, d)
        pos = torch.arange(window).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)
        idx = torch.arange(window)
        self.register_buffer('dist', (idx[None] - idx[:, None]).abs().float())
        self.attn = nn.ModuleList(AnomalyAttention(d, heads) for _ in range(layers))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(layers))
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.out = nn.Linear(d, features)

    def forward(self, x):
        z = self.embed(x) + self.pe[:x.shape[1]]
        series, prior = [], []
        for attn, ff, n1, n2 in zip(self.attn, self.ff, self.n1, self.n2):
            a, s, p = attn(z, self.dist[:x.shape[1], :x.shape[1]])
            z = n1(z + a); z = n2(z + ff(z))
            series.append(s); prior.append(p)
        return self.out(z), series, prior


def kl(p, q):
    return (p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))).sum(-1).mean(1)   # [b,t]


def discrepancy(series, prior, detach_prior=False, detach_series=False):
    total = 0
    for s, p in zip(series, prior):
        p = p / p.sum(-1, keepdim=True)
        ps, ss = (p.detach() if detach_prior else p), (s.detach() if detach_series else s)
        total = total + kl(ps, ss) + kl(ss, ps)
    return total / len(series)


def windows(sequences, width, mean, scale):
    """Non-overlapping windows; the last one is right-aligned so every point is scored once."""
    xs, owners = [], []
    for i, (x, _) in enumerate(sequences):
        z = np.clip((x - mean) / scale, -5, 5).astype(np.float32)
        if len(z) < width:
            z = np.vstack([np.repeat(z[:1], width - len(z), axis=0), z]); starts = [0]; offset = width - len(x)
        else:
            starts = list(range(0, len(z) - width + 1, width)); offset = 0
            if starts[-1] + width < len(z):
                starts.append(len(z) - width)
        for s in starts:
            xs.append(z[s:s + width]); owners.append((i, s - offset))
    return np.stack(xs), owners


def score(model, sequences, width, mean, scale, temperature=50.):
    x, owners = windows(sequences, width, mean, scale)
    out = [np.full(len(s[1]), np.nan) for s in sequences]
    model.eval()
    with torch.no_grad():
        for b in range(0, len(x), 256):
            xb = torch.from_numpy(x[b:b + 256])
            rec, series, prior = model(xb)
            err = ((rec - xb) ** 2).mean(-1)
            # test-time criterion sums the layers, as in the reference implementation
            value = (torch.softmax(-discrepancy(series, prior) * len(series) * temperature, dim=-1) * err).numpy()
            for (i, start), v in zip(owners[b:b + 256], value):
                lo = max(start, 0)
                hi = min(start + width, len(out[i]))
                seg = v[lo - start:hi - start]
                # keep the first score a point received (right-aligned tail only fills the gap)
                gap = np.isnan(out[i][lo:hi])
                out[i][lo:hi][gap] = seg[gap]
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(1)
    splits = {s: load(args.dataset, s) for s in ('train', 'validation', 'test')}
    width = 100 if args.dataset == 'sapienza' else 160
    stacked = np.vstack([s[0] for s in splits['train']])
    mean, scale = stacked.mean(0), np.maximum(stacked.std(0), 1e-6)
    xtr, _ = windows(splits['train'], width, mean, scale)
    labels = {s: np.concatenate([q[1] for q in splits[s]]) for s in ('validation', 'test')}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Model(xtr.shape[-1], width)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(args.epochs):
            model.train()
            order = np.random.permutation(len(xtr)); losses = []
            for b in range(0, len(order), 64):
                xb = torch.from_numpy(xtr[order[b:b + 64]])
                rec, series, prior = model(xb)
                rec_loss = F.mse_loss(rec, xb)
                # minimise phase trains the prior, maximise phase trains the series association (k = 3)
                loss_min = rec_loss + 3 * discrepancy(series, prior, detach_series=True).mean()
                loss_max = rec_loss - 3 * discrepancy(series, prior, detach_prior=True).mean()
                opt.zero_grad()
                loss_min.backward(retain_graph=True)
                loss_max.backward()
                opt.step(); losses.append(rec_loss.item())
            print(f'{args.dataset} seed {seed} epoch {epoch + 1} reconstruction {np.mean(losses):.5f}', flush=True)
        sva = score(model, splits['validation'], width, mean, scale)
        ste = score(model, splits['test'], width, mean, scale)
        record(args.dataset, 'anomaly-transformer', 'unsupervised', seed, labels['validation'], sva,
               labels['test'], ste, extra=dict(window=width, epochs=args.epochs, d_model=64, layers=3,
                                               supervision='none for fitting; validation labels pick threshold only',
                                               causal=False))


if __name__ == '__main__':
    main()
