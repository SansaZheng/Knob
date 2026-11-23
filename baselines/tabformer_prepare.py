"""Parse IBM TabFormer into compact arrays, the chronological split and causal features.

Usage: python -m baselines.tabformer_prepare --work /path/to/scratch

The split reproduces Kronos_SupervisedAD exactly: calendar days are ordered, the
70 % / 85 % row-count quantiles pick two day boundaries, ties stay together.
Every feature of a transaction uses only that transaction and earlier ones of the
same card / user / merchant.  Labels are never used as features.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARTS = sorted((ROOT / 'data/tabformer').glob('transactions.tgz.part*'))
ERRORS = ['Bad PIN', 'Insufficient Balance', 'Technical Glitch', 'Bad Card Number',
          'Bad CVV', 'Bad Expiration', 'Bad Zipcode']
RAW = ['log_amount', 'refund', 'hour', 'weekday', 'chip_code', 'mcc', 'state_code', 'online',
       'foreign', 'zip_missing'] + ['err_' + e.lower().replace(' ', '_') for e in ERRORS]
AGG = ['card_gap_log', 'user_gap_log', 'card_n_1h', 'card_n_24h', 'card_n_7d', 'card_amt_24h_log',
       'card_amt_7d_log', 'user_amt_ratio', 'user_amt_z', 'user_prior_n_log', 'um_prior_n_log',
       'um_gap_log', 'umcc_prior_frac', 'ustate_prior_frac', 'city_changed', 'state_changed',
       'user_online_frac', 'merchant_prior_n_log', 'user_err_24h', 'chip_changed']


def parse(work):
    cols = {}
    codes = {k: {} for k in ('merchant', 'city', 'state')}

    def code(name, values):
        table = codes[name]
        uniq, inv = np.unique(values, return_inverse=True)
        ids = np.empty(len(uniq), dtype=np.int32)
        for j, u in enumerate(uniq):
            ids[j] = table.setdefault(u, len(table))
        return ids[inv]

    # Stream the byte-split archive without restoring a 2.3 GB CSV on disk.
    cat = subprocess.Popen(['cat', *map(str, PARTS)], stdout=subprocess.PIPE)
    tar = subprocess.Popen(['tar', 'xzOf', '-'], stdin=cat.stdout, stdout=subprocess.PIPE)
    reader = pd.read_csv(tar.stdout, chunksize=1_000_000, dtype=str, keep_default_na=False)
    parts, n = [], 0
    for frame in reader:
        date = pd.to_datetime(dict(year=frame.Year.astype(int), month=frame.Month.astype(int),
                                   day=frame.Day.astype(int)))
        hm = frame.Time.str.split(':', expand=True).astype(int)
        day = ((date - pd.Timestamp('1970-01-01')) // pd.Timedelta(days=1)).to_numpy()   # unit-safe
        state = frame['Merchant State'].to_numpy()
        city = frame['Merchant City'].str.strip().to_numpy()     # online rows read ' ONLINE'
        err = frame['Errors?'].to_numpy()
        errbits = np.zeros(len(frame), dtype=np.uint8)
        for b, e in enumerate(ERRORS):
            errbits |= (np.char.find(err.astype(str), e) >= 0).astype(np.uint8) << b
        parts.append(dict(
            user=frame.User.astype(np.int32).to_numpy(), card=frame.Card.astype(np.int16).to_numpy(),
            day=day.astype(np.int32), minute=(hm[0] * 60 + hm[1]).to_numpy(dtype=np.int16),
            weekday=date.dt.dayofweek.to_numpy(dtype=np.int8),
            amount=frame.Amount.str.replace('$', '', regex=False).astype(np.float32).to_numpy(),
            chip=frame['Use Chip'].map({'Swipe Transaction': 0, 'Chip Transaction': 1,
                                        'Online Transaction': 2}).to_numpy(dtype=np.int8),
            merchant=code('merchant', frame['Merchant Name'].to_numpy()),
            city=code('city', city), state=code('state', state),
            online=(city == 'ONLINE'), foreign=np.array([len(s) > 2 for s in state]),
            zip_missing=(frame.Zip.to_numpy() == ''), mcc=frame.MCC.astype(np.int16).to_numpy(),
            errbits=errbits, y=(frame['Is Fraud?'].to_numpy() == 'Yes').astype(np.int8)))
        n += len(frame)
        print('parsed', n, flush=True)
    for k in parts[0]:
        cols[k] = np.concatenate([p[k] for p in parts])
    del parts
    np.savez(work / 'parsed.npz', **cols)
    return cols


def split_of(day):
    keys, counts = np.unique(day, return_counts=True)
    cumulative = np.cumsum(counts)
    total = len(day)
    bounds = [int(keys[min(int(np.searchsorted(cumulative, total * f)), len(keys) - 1)]) for f in (.7, .85)]
    return np.searchsorted(bounds, day, side='right').astype(np.int8), bounds


def group_start(*keys):
    """Boolean mask of first rows of each run in arrays already sorted by keys."""
    first = np.ones(len(keys[0]), dtype=bool)
    same = np.ones(len(keys[0]) - 1, dtype=bool)
    for k in keys:
        same &= k[1:] == k[:-1]
    first[1:] = ~same
    return first


def cumcount(first):
    idx = np.arange(len(first))
    return idx - np.maximum.accumulate(np.where(first, idx, 0))


def features(c, work):
    """Write one float32 file per feature (work/cols/<name>.npy) to keep memory bounded."""
    n = len(c['y'])
    cols = work / 'cols'; cols.mkdir(exist_ok=True)
    t = c['day'].astype(np.int64) * 86400 + c['minute'].astype(np.int64) * 60
    idx = np.arange(n)

    def put(name, values, order=None):
        assert name in RAW + AGG, name
        out = np.empty(n, dtype=np.float32)
        if order is None:
            out[:] = values
        else:
            out[order] = values
        np.save(cols / f'{name}.npy', out)

    amount = c['amount'].astype(np.float64)
    put('log_amount', np.sign(amount) * np.log1p(np.abs(amount)))
    put('refund', amount < 0); put('hour', c['minute'] / 60.); put('weekday', c['weekday'])
    put('chip_code', c['chip']); put('mcc', c['mcc']); put('online', c['online'])
    put('foreign', c['foreign']); put('zip_missing', c['zip_missing'])
    for b, e in enumerate(ERRORS):
        put('err_' + e.lower().replace(' ', '_'), (c['errbits'] >> b) & 1)
    split, bounds = split_of(c['day'])
    # State code: rank by training-period frequency (label-free); unseen states get -1.
    freq = np.bincount(c['state'][split == 0], minlength=c['state'].max() + 1)
    rank = np.full(len(freq), -1, dtype=np.int32)
    seen = np.flatnonzero(freq)
    rank[seen[np.argsort(-freq[seen], kind='stable')]] = np.arange(len(seen))
    put('state_code', rank[c['state']])

    card = c['user'].astype(np.int64) * 16 + c['card']
    np.save(work / 'split.npy', split); np.save(work / 'y.npy', c['y']); np.save(work / 't.npy', t)
    np.save(work / 'card.npy', card)
    BIG = np.int64(1) << 34
    absamt = np.abs(amount); del amount

    # --- card stream (lexsort is stable, so equal timestamps keep source-row order) ---
    o = np.lexsort((t, card)); ck, ct = card[o], t[o]
    first = group_start(ck)
    gap = np.r_[0, np.diff(ct)].astype(np.float64)
    put('card_gap_log', np.where(first, -1, np.log1p(np.maximum(gap, 0))), o)
    key = ck * BIG + ct; del ck, ct, gap
    cs = np.r_[0, np.cumsum(absamt[o])]
    for name, w in (('1h', 3600), ('24h', 86400), ('7d', 7 * 86400)):
        start = np.searchsorted(key, key - w, side='left')     # earlier rows of this card within w
        put('card_n_' + name, idx - start, o)
        if name != '1h':
            put(f'card_amt_{name}_log', np.log1p(cs[idx] - cs[start]), o)
    chip = c['chip'][o]
    put('chip_changed', np.where(first, 0, np.r_[0, chip[1:] != chip[:-1]]), o)
    del key, cs, start, chip, card
    print('card stream done', flush=True)

    # --- user stream ---
    user = c['user'].astype(np.int64)
    o = np.lexsort((t, user)); uk, ut = user[o], t[o]
    first = group_start(uk); k = cumcount(first).astype(np.float64)
    gap = np.r_[0, np.diff(ut)].astype(np.float64)
    put('user_gap_log', np.where(first, -1, np.log1p(np.maximum(gap, 0))), o)
    put('user_prior_n_log', np.log1p(k), o); del gap

    def prior_sum(v):
        s = np.cumsum(v) - v                       # exclusive running sum over the sorted array
        return s - np.maximum.accumulate(np.where(first, s, -np.inf))

    a = absamt[o]; s1 = prior_sum(a)
    mean = np.where(k > 0, s1 / np.maximum(k, 1), 0.); del s1
    put('user_amt_ratio', np.where(k > 0, a / (mean + 1.), -1), o)
    var = np.maximum(prior_sum(a * a) / np.maximum(k, 1) - mean ** 2, 0)
    put('user_amt_z', np.where(k > 1, (a - mean) / np.sqrt(var + 1.), 0), o); del var, mean, a
    put('user_online_frac', np.where(k > 0, prior_sum(c['online'][o].astype(np.float64)) / np.maximum(k, 1), -1), o)
    for name, v in (('city_changed', c['city'][o]), ('state_changed', c['state'][o])):
        put(name, np.where(first, 0, np.r_[0, v[1:] != v[:-1]]), o)
    key = uk * BIG + ut; del uk, ut
    ce = np.r_[0, np.cumsum(c['errbits'][o] > 0)]
    start = np.searchsorted(key, key - 86400, side='left')
    put('user_err_24h', ce[idx] - ce[start], o)
    user_k = np.empty(n, dtype=np.float32); user_k[o] = k
    del key, ce, start, k
    print('user stream done', flush=True)

    # --- user x merchant / user x mcc / user x state / merchant streams ---
    def prior_in(second):
        oo = np.lexsort((t, second, user))
        f = group_start(user[oo], second[oo])
        return oo, f, cumcount(f).astype(np.float64)

    oo, f, kk = prior_in(c['merchant'].astype(np.int64))
    put('um_prior_n_log', np.log1p(kk), oo)
    g = np.r_[0, np.diff(t[oo])].astype(np.float64)
    put('um_gap_log', np.where(f, -1, np.log1p(np.maximum(g, 0))), oo); del g
    for name, second in (('umcc_prior_frac', c['mcc']), ('ustate_prior_frac', c['state'])):
        oo, f, kk = prior_in(second.astype(np.int64))
        put(name, np.where(user_k[oo] > 0, kk / np.maximum(user_k[oo], 1), -1), oo)
    del user, user_k, f, kk
    m = c['merchant'].astype(np.int64)
    oo = np.lexsort((t, m))
    put('merchant_prior_n_log', np.log1p(cumcount(group_start(m[oo]))), oo)
    missing = [name for name in RAW + AGG if not (cols / f'{name}.npy').exists()]
    assert not missing, missing
    summary = dict(rows=n, positives=int(c['y'].sum()), day_boundaries=bounds, features_raw=RAW, features_agg=AGG,
                   splits={name: dict(rows=int((split == i).sum()), positives=int(c['y'][split == i].sum()))
                           for i, name in enumerate(('train', 'validation', 'test'))},
                   users=int(len(np.unique(c['user']))), cards=int(len(np.unique(np.load(work / 'card.npy')))))
    (work / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--work', required=True)
    work = Path(p.parse_args().work); work.mkdir(parents=True, exist_ok=True)
    if (work / 'parsed.npz').exists():
        with np.load(work / 'parsed.npz') as z:
            cols = {k: z[k] for k in z.files}
    else:
        cols = parse(work)
    features(cols, work)
