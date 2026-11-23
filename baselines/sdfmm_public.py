"""SD-FMM public training script (Dai et al., IPM 2026), adapted to the research splits.

Usage: python -m baselines.sdfmm_public --dataset sd-stock|sd-crypto

Model, losses and fragment extraction are taken from the authors' train.py
(https://github.com/ys-Dai/SD-FMM, commit 0d1a2158, MIT licence).  Two variants:

  synthetic  - the authors' recipe: train only on their released synthetic anomalies
               (data/sd_fmm/synthetic), no real label is seen in training.  The script
               slices [:1501]/[1501:2001] but 999 sequences were released, so 799/200 is used.
  finetune   - the same network, then supervised training on the real research
               train split; the epoch is chosen by validation AP.

The network attends over the full 160-point window in both directions, so it is an
offline detector, unlike the causal tree features and the causal Kronos backbone.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .common import ROOT, SEEDS, record
from .market import load

PATCH, LENGTH = 4, 160


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature, self.cos = temperature, nn.CosineSimilarity(dim=-1)

    def forward(self, normal, anomaly):
        return torch.mean(-torch.log(1 - torch.sigmoid(self.cos(normal, anomaly) / self.temperature) + 1e-8))


class Net(nn.Module):
    """ContrastiveTimeSeriesTransformer from the authors' train.py."""
    def __init__(self, d=64, heads=4, layers=2, ff=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(PATCH, d)
        pe = torch.zeros(5000, d)
        pos = torch.arange(5000, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True), layers)
        self.output_layer = nn.Linear(d, PATCH)
        self.fragment_expansion = nn.Linear(13, LENGTH)
        self.feature_extractor = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, d // 4))

    def forward(self, x, get_features=False):
        b = x.shape[0]
        if x.shape[1] == 13:
            x = self.fragment_expansion(x.transpose(1, 2)).transpose(1, 2)
        z = self.embedding(x.reshape(b, LENGTH // PATCH, PATCH))
        z = self.encoder(self.drop(z + self.pe[:, :z.size(1)]))
        if get_features:
            return self.feature_extractor(z.mean(1))
        return torch.sigmoid(self.output_layer(z).reshape(b, LENGTH))


def fragments(data, labels, width=13):
    """Paired anomaly / neighbouring normal fragments (authors' extract_fragments)."""
    normal, anomaly = np.zeros((len(data), width, 1)), np.zeros((len(data), width, 1))
    for i in range(len(data)):
        idx = np.where(labels[i] == 1)[0]
        if len(idx) == 0:
            a0 = np.random.randint(0, LENGTH - width); n0 = (a0 + 2 * width) % LENGTH
            n0 = min(n0, LENGTH - width)
        else:
            centre = (idx[0] + idx[-1] + 1) // 2
            a0 = min(max(0, centre - width // 2), LENGTH - width)
            n0 = a0 - width if a0 >= width else min(a0 + width, LENGTH - width)
        anomaly[i], normal[i] = data[i, a0:a0 + width], data[i, n0:n0 + width]
    return normal, anomaly


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def train(model, x, y, epochs, select, lr=1e-3, weight=0.5):
    """BCE + 0.5 x contrastive loss; keep the epoch with the best `select(model)` value."""
    bce, contrast = nn.BCELoss(), ContrastiveLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    normal, anomaly = fragments(x, y)
    main = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=16, shuffle=True)
    frag = DataLoader(TensorDataset(torch.FloatTensor(normal), torch.FloatTensor(anomaly)), batch_size=16, shuffle=True)
    best, state = -1., None
    for _ in range(epochs):
        model.train(); it = iter(frag)
        for xb, yb in main:
            try:
                nb, ab = next(it)
            except StopIteration:
                it = iter(frag); nb, ab = next(it)
            loss = bce(model(xb), yb) + weight * contrast(model(nb, True), model(ab, True))
            opt.zero_grad(); loss.backward(); opt.step()
        value = select(model)
        if value > best:
            best, state = value, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['sd-stock', 'sd-crypto'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--finetune-epochs', type=int, default=60)
    args = p.parse_args()
    torch.set_num_threads(1)
    syn_x = np.load(ROOT / 'data/sd_fmm/synthetic/all_DTWsim_data.npy').astype(np.float32)
    syn_y = np.load(ROOT / 'data/sd_fmm/synthetic/all_DTWsim_label.npy').astype(np.float32)
    real = {}
    for s in ('train', 'validation', 'test'):
        seqs = load(args.dataset, s)
        real[s] = (np.stack([q[0] for q in seqs]).astype(np.float32), np.stack([q[1] for q in seqs]).astype(np.float32))
    flat = lambda a: a.reshape(-1)
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Net()
        # authors' selection rule: synthetic hold-out F1 at threshold 0.1
        rule = lambda m: f1_score(flat(syn_y[799:]), flat(predict(m, syn_x[799:])) >= 0.1, zero_division=0)
        best = train(model, syn_x[:799], syn_y[:799], args.epochs, rule)
        record(args.dataset, 'sd-fmm-public', 'synthetic', seed, flat(real['validation'][1]),
               flat(predict(model, real['validation'][0])), flat(real['test'][1]), flat(predict(model, real['test'][0])),
               extra=dict(epochs=args.epochs, synthetic_holdout_F1=float(best), causal=False,
                          supervision='synthetic anomalies only; real validation labels pick threshold only'))
        rule = lambda m: average_precision_score(flat(real['validation'][1]), flat(predict(m, real['validation'][0])))
        best = train(model, real['train'][0], real['train'][1], args.finetune_epochs, rule, lr=3e-4)
        record(args.dataset, 'sd-fmm-public', 'finetune', seed, flat(real['validation'][1]),
               flat(predict(model, real['validation'][0])), flat(real['test'][1]), flat(predict(model, real['test'][0])),
               extra=dict(pretrain_epochs=args.epochs, finetune_epochs=args.finetune_epochs, causal=False,
                          supervision='synthetic pretraining, then real research-train labels'))


if __name__ == '__main__':
    main()
