"""v4 features: the 12 ball features of train_possession.py PLUS team-shape features that need
no ball at all, so the model still has something to go on during long ball outages.

Shape features (per team, in the team's own attacking direction): where the block sits (centroid,
front line, back line), how wide / deep it is, how fast it moves, how many players sprint, and
how fast it moves forward; plus a few joint ones. They are smoothed over 1 s.

Attacking direction is found from the tracking itself, per period (team 0 is the one whose mean
x is lower, so it attacks +x). No match-specific constants, no labels.
"""
import warnings

import numpy as np

from train_possession import features as ball_features, PITCH, STEP

N_SHAPE = 23
N_FEATURES2 = 12 + N_SHAPE
SMOOTH = 5                     # steps (= 1 s at 5 fps)


def orient(players, team, period):
    """Copy of players (normalised 0-1) flipped per period so team 0 sits on the left."""
    out = players.copy()
    period = np.zeros(len(players), dtype=np.int8) if period is None else period
    for p in np.unique(period):
        m = period == p
        tm = team[m]
        x = players[m][..., 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x0 = np.nanmean(np.where(tm == 0, x, np.nan))
            x1 = np.nanmean(np.where(tm == 1, x, np.nan))
        if np.isfinite(x0) and np.isfinite(x1) and x0 > x1:
            out[m] = 1.0 - out[m]
    return out


def _smooth(a):
    """Moving average over SMOOTH steps along axis 0, ignoring NaN."""
    ok = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0)
    k = np.ones(SMOOTH, dtype=np.float32)
    num = np.stack([np.convolve(a0[:, j], k, "same") for j in range(a.shape[1])], 1)
    den = np.stack([np.convolve(ok[:, j], k, "same") for j in range(a.shape[1])], 1)
    return np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)


def shape_features(players, team, period):
    """(ceil(n/STEP), N_SHAPE) ball-free team-shape features at STEP resolution."""
    n = len(players)
    po = orient(players, team, period) * PITCH                   # metres, team 0 attacks +x
    idx = np.arange(0, n, STEP)
    prev = np.maximum(idx - STEP, 0)
    dt = STEP / 25.0
    cur, old = po[idx], po[prev]
    tm = team[idx]
    delta = cur - old
    spd = np.linalg.norm(delta, axis=2) / dt
    spd = np.where((spd > 12.0) | (idx[:, None] < STEP), np.nan, spd)
    cols = []
    sx_sy = []
    cx_abs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in (0, 1):
            m = (tm == t) & np.isfinite(cur[..., 0])
            sign = 1.0 if t == 0 else -1.0
            fx = np.where(m, cur[..., 0] if t == 0 else 105.0 - cur[..., 0], np.nan)
            y = np.where(m, cur[..., 1], np.nan)
            s = np.where(m, spd, np.nan)
            fv = np.where(m, sign * delta[..., 0] / dt, np.nan)
            fv = np.where(np.isfinite(s), fv, np.nan)
            sx, sy = np.nanstd(fx, axis=1) / 15.0, np.nanstd(y, axis=1) / 15.0
            cols += [(np.nanmean(fx, 1) - 52.5) / 30.0, sx, sy,
                     (np.nanpercentile(fx, 80, axis=1) - 52.5) / 30.0,
                     (np.nanpercentile(fx, 20, axis=1) - 52.5) / 30.0,
                     np.nanmean(s, 1) / 3.0, np.nanmean(s > 4.0, 1) if False else
                     np.nansum(np.where(np.isfinite(s), s > 4.0, 0), 1) / np.maximum(np.isfinite(s).sum(1), 1),
                     np.nanmean(fv, 1) / 2.0, m.sum(1) / 11.0]
            sx_sy.append(sx * sy)
            cx_abs.append(np.nanmean(np.where(m, cur[..., 0], np.nan), 1))
        alive = np.isfinite(spd)
        joint = [(cx_abs[0] - cx_abs[1]) / 30.0, sx_sy[0] - sx_sy[1],
                 np.nanmean(spd, 1) / 3.0, np.nanpercentile(spd, 90, axis=1) / 6.0,
                 np.nansum(np.where(alive, spd > 4.0, 0), 1) / np.maximum(alive.sum(1), 1)]
    out = np.stack(cols + joint, axis=1).astype(np.float32)
    out = _smooth(out)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def features2(ball, players, team, hidden, period=None):
    """(ceil(n/STEP), N_FEATURES2): ball features (subsampled) + shape features."""
    a = ball_features(ball, players, team, hidden)[::STEP]
    b = shape_features(players, team, period)
    assert len(a) == len(b), (len(a), len(b))
    return np.concatenate([a, b], axis=1)


def make_gru2(hidden=64, n_in=N_FEATURES2):
    import torch.nn as nn

    class GRU2(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_in, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hidden, 3)

        def forward(self, x):
            return self.head(self.rnn(x)[0])

    return GRU2()


def make_model(arch="gru", n_in=N_FEATURES2, hidden=64):
    """Sequence models with the same interface: (batch, time, n_in) -> (batch, time, 3) logits."""
    import torch, torch.nn as nn
    if arch == "gru":
        return make_gru2(hidden, n_in)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hidden, 3)
        def forward(self, x):
            return self.head(self.rnn(x)[0])

    class Block(nn.Module):
        def __init__(self, c, dil):
            super().__init__()
            self.conv = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
            self.norm = nn.GroupNorm(1, c)
            self.drop = nn.Dropout(0.1)
        def forward(self, x):
            return x + self.drop(torch.relu(self.norm(self.conv(x))))

    class TCN(nn.Module):          # dilated non-causal convolutions, receptive field about 125 steps (25 s)
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, hidden, 1)
            self.blocks = nn.Sequential(*[Block(hidden, d) for d in (1, 2, 4, 8, 16)])
            self.head = nn.Conv1d(hidden, 3, 1)
        def forward(self, x):
            return self.head(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)

    class ConvGRU(nn.Module):      # local conv smoothing in front of a bidirectional GRU
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(n_in, hidden, 5, padding=2), nn.ReLU())
            self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.head = nn.Linear(2 * hidden, 3)
        def forward(self, x):
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return self.head(self.rnn(h)[0])

    class Transformer(nn.Module):  # conv embedding + 3 self-attention layers over the 60 s window
        def __init__(self, win=300):
            super().__init__()
            self.emb = nn.Sequential(nn.Conv1d(n_in, hidden, 5, padding=2), nn.ReLU())
            self.pos = nn.Parameter(torch.zeros(1, win, hidden))
            layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, 0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 3)
            self.head = nn.Linear(hidden, 3)
        def forward(self, x):
            h = self.emb(x.transpose(1, 2)).transpose(1, 2)
            return self.head(self.enc(h + self.pos[:, :h.shape[1]]))

    return {"lstm": LSTM, "tcn": TCN, "convgru": ConvGRU, "transformer": Transformer}[arch]()
