"""Final training: ALL external IDSSE matches, several random seeds, fixed number of epochs.

No match is held out here, so there is no validation score to print. The settings (epoch count)
were chosen earlier on external held-out matches by train_v3.py. Averaging several seeds makes
the final probabilities steadier than any single run.

External data only. Nothing from the competition match is used.

    python training/train_final.py                      # 3 seeds, 27 epochs each
    python training/train_final.py --seeds 5 --epochs 27

Writes training/possession_gru_final_s0.pt, _s1.pt, ... which possession_hybrid.py averages.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_possession import GRU, DATA, load, windows  # noqa: E402


def train_one(paths, seed, epochs, copies):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    print("\n=== seed {} ===  building features ...".format(seed))
    train = []
    for p in paths:
        for _ in range(copies):
            train += windows(*load(p, rng))
    print("{} training windows from {} matches".format(len(train), len(paths)))

    model = GRU()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for s in range(0, len(order), 32):
            batch = [train[i] for i in order[s:s + 32]]
            xb = torch.from_numpy(np.stack([b[0] for b in batch]))
            yb = torch.from_numpy(np.stack([b[1] for b in batch]))
            loss = loss_fn(model(xb).reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(batch)
        print("seed {} epoch {:>2}/{}  loss {:.3f}".format(seed, epoch, epochs, total / len(train)))
    out = os.path.join(HERE, "possession_gru_final_s{}.pt".format(seed))
    torch.save(model.state_dict(), out)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=27)
    ap.add_argument("--copies", type=int, default=3, help="noisy versions of each match")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(DATA, "*.npz")))
    if len(paths) < 7:
        print("warning: only {} matches found in {} (expected 7); run build_idsse.py".format(len(paths), DATA))
    if not paths:
        raise SystemExit("no matches built")
    for seed in range(args.seeds):
        train_one(paths, seed, args.epochs, args.copies)


if __name__ == "__main__":
    main()
