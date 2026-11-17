"""Train a small sequence model that labels every moment H / A / D from tracking.

Trains ONLY on the external IDSSE matches built by build_idsse.py. Nothing from the competition
match is used here. One IDSSE match is held out to check that the model generalises.

Because IDSSE has the ball on 100% of frames and the competition tracking loses it on ~41%, every
training copy gets random ball dropouts applied so the model learns to cope with gaps.

    pip install torch numpy
    python training/train_possession.py                  # holds out J03WPY, 6 matches train
    python training/train_possession.py --epochs 30 --holdout J03WQQ

Runs on CPU in minutes; a GPU (Colab) just makes it faster. Saves training/possession_gru_v2.pt.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "external", "idsse")
PITCH = np.array([105.0, 68.0], dtype=np.float32)
STEP = 5                 # model runs at 5 fps; labels are expanded back to 25 fps afterwards
WIN, STRIDE = 300, 150   # 300 steps = 60 s windows
N_FEATURES = 12


def dropout_mask(n, rng, mean_gap=None):
    """True where the ball is hidden: a MIX of short flickers and long outages (4 s to 60 s).

    The target share of hidden frames is drawn per copy (30-50%) so the model sees a range.
    These are generic ranges, not fitted to the competition tracking.
    """
    target = rng.uniform(0.30, 0.50)
    hidden = np.zeros(n, dtype=bool)
    while hidden.mean() < target:
        start = rng.integers(0, n)
        if rng.random() < 0.4:
            length = int(rng.exponential(25)) + 1                       # short flicker
        else:
            length = int(np.exp(rng.uniform(np.log(100), np.log(1500))))  # 4 s .. 60 s outage
        hidden[start:start + length] = True
    return hidden


def features(ball, players, team, hidden):
    """(n, N_FEATURES) per-frame features that do not depend on which way a team attacks."""
    n = len(ball)
    seen = ~hidden & ~np.isnan(ball[:, 0])
    # carry the last seen ball forward through gaps; remember how stale it is
    idx = np.where(seen, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    b = ball[idx] * PITCH
    age = np.minimum(np.arange(n) - idx, 250) / 250.0
    age[~np.maximum.accumulate(seen)] = 1.0          # nothing seen yet

    d = np.linalg.norm(players * PITCH - b[:, None, :], axis=2)       # (n, K)
    d = np.where(np.isnan(d), np.inf, d)
    home = np.where(team == 0, d, np.inf)
    away = np.where(team == 1, d, np.inf)
    dh = np.minimum(home.min(axis=1), 40.0)
    da = np.minimum(away.min(axis=1), 40.0)
    near5h, near5a = (home < 5).sum(1), (away < 5).sum(1)
    near12h, near12a = (home < 12).sum(1), (away < 12).sum(1)

    step = 5
    speed = np.zeros(n, dtype=np.float32)
    speed[step:] = np.linalg.norm(b[step:] - b[:-step], axis=1) / (step / 25.0)
    pm = players * PITCH
    pspeed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    pspeed[step:] = np.linalg.norm(pm[step:] - pm[:-step], axis=2) / (step / 25.0)
    pspeed = np.where(pspeed > 12.0, np.nan, pspeed)       # ignore identity swaps / teleports
    with np.errstate(all="ignore"):
        mean_pspeed = np.nan_to_num(np.nanmean(pspeed, axis=1), nan=0.0)
        frac_still = np.nan_to_num(np.nanmean(pspeed < 1.0, axis=1), nan=0.0)
    out = np.stack([
        seen.astype(np.float32), age, dh / 10.0, da / 10.0,
        (da - dh) / 10.0,                       # >0 means home is closer
        near5h / 3.0, near5a / 3.0, near12h / 6.0, near12a / 6.0,
        np.minimum(speed, 40.0) / 20.0,
        mean_pspeed / 3.0, frac_still,
    ], axis=1)
    # A ball that is untracked from the very first frame of a half leaves NaN in the speed
    # feature; one NaN poisons a whole recurrent window. Missing means 0.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load(path, rng):
    z = np.load(path)
    sigma = rng.uniform(0.1, 0.5) / PITCH                  # metres of random tracking jitter
    ball = z["ball"] + rng.normal(0, 1, z["ball"].shape).astype(np.float32) * sigma
    players = z["players"] + rng.normal(0, 1, z["players"].shape).astype(np.float32) * sigma
    x = features(ball, players, z["team"], dropout_mask(len(z["ball"]), rng))
    return x[::STEP], z["label"][::STEP].astype(np.int64)


def windows(x, y):
    out = []
    for s in range(0, max(1, len(x) - WIN + 1), STRIDE):
        out.append((x[s:s + WIN], y[s:s + WIN]))
    return out


class GRU(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(N_FEATURES, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        return self.head(self.rnn(x)[0])


def macro_f1(y, p):
    f = []
    for c in range(3):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), f


def n_possessions(labels):
    """Runs of H or A (dead ball splits them), plus their mean length in seconds at 5 fps."""
    runs, length, prev = [], 0, None
    for v in labels:
        if v in (0, 1):
            if v == prev:
                length += 1
            else:
                if prev in (0, 1): runs.append(length)
                length = 1
        else:
            if prev in (0, 1): runs.append(length)
            length = 0
        prev = v
    if prev in (0, 1): runs.append(length)
    return len(runs), (np.mean(runs) / (25 / STEP) if runs else 0.0)


@torch.no_grad()
def predict(model, x):
    model.eval()
    out = np.zeros(len(x), dtype=np.int64)
    for s in range(0, len(x), WIN):                  # non-overlapping chunks are fine for a check
        chunk = torch.from_numpy(x[s:s + WIN])[None]
        out[s:s + WIN] = model(chunk)[0].argmax(-1).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="J03WPY")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--copies", type=int, default=3, help="dropout versions of each match")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    paths = sorted(glob.glob(os.path.join(DATA, "*.npz")))
    train_paths = [p for p in paths if args.holdout not in p]
    hold_path = [p for p in paths if args.holdout in p]
    if not train_paths or not hold_path:
        raise SystemExit("need built matches in {} (run build_idsse.py) incl. {}".format(DATA, args.holdout))

    print("building features ...")
    train = []
    for p in train_paths:
        for _ in range(args.copies):
            train += windows(*load(p, rng))
    xh, yh = load(hold_path[0], np.random.default_rng(123))
    print("{} training windows from {} matches; holding out {}".format(len(train), len(train_paths), args.holdout))

    model = GRU()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for s in range(0, len(order), 32):
            batch = [train[i] for i in order[s:s + 32]]
            xb = torch.from_numpy(np.stack([b[0] for b in batch]))
            yb = torch.from_numpy(np.stack([b[1] for b in batch]))
            loss = loss_fn(model(xb).reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(batch)
        pred = predict(model, xh)
        mf, per = macro_f1(yh, pred)
        pn, pd = n_possessions(pred); tn, td = n_possessions(yh)
        print("epoch {:>2}  loss {:.3f}  held-out macroF1 {:.3f} (H {:.2f} A {:.2f} D {:.2f})  "
              "possessions pred {} vs true {}  mean {:.1f}s vs {:.1f}s".format(
                  epoch + 1, total / len(train), mf, per[0], per[1], per[2], pn, tn, pd, td))

    out = os.path.join(HERE, "possession_gru_v2.pt")
    torch.save(model.state_dict(), out)
    print("saved", out)


if __name__ == "__main__":
    main()
