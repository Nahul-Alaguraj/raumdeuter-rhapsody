"""Train v3: pick the epoch by possession-spell F1 and tune the flip penalties, on EXTERNAL data.

    VAL  match  -> chooses the best epoch and the decoding penalties (lam, mu)
    TEST match  -> only reports; never used to choose anything
    the rest    -> training (with realistic ball outages and jitter, as in train_possession.py)

The score is computed with the competition's own metric code (evaluation/possession_metrics.py)
on the IDSSE labels. IDSSE possessions are shorter than the competition's, so the penalties
chosen here are on the cautious side -- say so in the technical document.

    python training/train_v3.py
    python training/train_v3.py --epochs 30 --val J03WPY --test J03WQQ
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from train_possession import GRU, DATA, load, windows  # noqa: E402
from decode import half_logprobs, viterbi  # noqa: E402

WEIGHTS = os.path.join(HERE, "possession_gru_v3.pt")
DECODE = os.path.join(HERE, "possession_decode_v3.json")
SEL_LAM, SEL_MU = 4.0, 1.0           # fixed penalties used only to choose the epoch
GRID_LAM = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
GRID_MU = [0.0, 1.0, 2.0, 4.0]


def spell(y, path):
    runs = [(0, len(y))]
    return pm.spell_scores(y.astype(np.int8), path.astype(np.int8), runs)["mean_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="J03WPY")
    ap.add_argument("--test", default="J03WQQ")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--copies", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    paths = sorted(glob.glob(os.path.join(DATA, "*.npz")))
    pick = lambda name: [p for p in paths if name in p]
    train_paths = [p for p in paths if args.val not in p and args.test not in p]
    if not train_paths or not pick(args.val) or not pick(args.test):
        raise SystemExit("need built matches incl. {} and {}".format(args.val, args.test))

    print("building features ...")
    train = []
    for p in train_paths:
        for _ in range(args.copies):
            train += windows(*load(p, rng))
    xv, yv = load(pick(args.val)[0], np.random.default_rng(123))
    xt, yt = load(pick(args.test)[0], np.random.default_rng(456))
    print("{} training windows from {} matches | val {} | test {}".format(
        len(train), len(train_paths), args.val, args.test))

    model = GRU()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
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
        f1 = spell(yv, viterbi(half_logprobs(model, xv), SEL_LAM, SEL_MU))
        mark = ""
        if f1 > best:
            best, best_epoch = f1, epoch
            torch.save(model.state_dict(), WEIGHTS)
            mark = "  <- best so far, saved"
        print("epoch {:>2}  loss {:.3f}  val spellF1 {:.3f}{}".format(
            epoch, total / len(train), f1, mark))

    print("\nbest epoch {} (val spellF1 {:.3f}). Tuning penalties on the val match ...".format(best_epoch, best))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    lv, lt = half_logprobs(model, xv), half_logprobs(model, xt)
    scores = {(l, m): spell(yv, viterbi(lv, l, m)) for l in GRID_LAM for m in GRID_MU}
    lam, mu = max(scores, key=scores.get)
    print("  best penalties on val: lam {} mu {}  (val spellF1 {:.3f})".format(lam, mu, scores[(lam, mu)]))
    for l in GRID_LAM:
        print("    lam {:>4}: ".format(l) + "  ".join("mu {} {:.3f}".format(m, scores[(l, m)]) for m in GRID_MU))

    raw = spell(yt, lv.argmax(1) if False else lt.argmax(1))
    dec = spell(yt, viterbi(lt, lam, mu))
    print("\nTEST match {} (never used to choose anything):".format(args.test))
    print("  spellF1 without decoding {:.3f}  |  with decoding {:.3f}".format(raw, dec))
    pn = lambda p: len(pm.possession_spells(p.astype(np.int8), [(0, len(p))]))
    print("  possessions: true {}  raw {}  decoded {}".format(pn(yt), pn(lt.argmax(1)), pn(viterbi(lt, lam, mu))))

    with open(DECODE, "w") as handle:
        json.dump({"lam": lam, "mu": mu, "best_epoch": best_epoch}, handle)
    print("saved", WEIGHTS, "and", DECODE)


if __name__ == "__main__":
    main()
