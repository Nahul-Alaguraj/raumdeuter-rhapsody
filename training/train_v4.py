"""Train v4: GRU on ball features + ball-free team-shape features, on EXTERNAL data only.

    IDSSE (7 Bundesliga matches, ball tracked -> we simulate outages)
    SoccerTrack v2 (10 matches, ball always hidden, labels may be -1 = unknown -> ignored)

Phase 1 (default): hold out one IDSSE match and one SoccerTrack match, pick the epoch by the
official spell F1 after Viterbi decoding (fixed lam/mu), report on a second IDSSE test match.
Phase 2 (--final): train on everything for --epochs epochs with several seeds, saving
training/possession_gru_v4_s{seed}.pt.

    python training/train_v4.py                       # phase 1
    python training/train_v4.py --final --epochs 25   # phase 2 (use the best epoch from phase 1)
"""
import argparse
import glob
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
from train_possession import features as ball_feats, DATA as IDSSE, PITCH, STEP, WIN, STRIDE, dropout_mask  # noqa: E402
from features2 import features2, make_model  # noqa: E402
from decode import half_logprobs, viterbi  # noqa: E402

ST_DIR = os.path.join(ROOT, "data", "external", "soccertrack")
LAM, MU = 2.0, 4.0           # decoding penalties (chosen on IDSSE in v3)
ARCH = "gru"
N_IN = 35                    # 35 = ball + team-shape features, 12 = ball features only (old model)


def load2(path, rng, soccertrack):
    z = np.load(path)
    sigma = rng.uniform(0.1, 0.5) / PITCH
    ball = z["ball"] + rng.normal(0, 1, z["ball"].shape).astype(np.float32) * sigma
    players = z["players"] + rng.normal(0, 1, z["players"].shape).astype(np.float32) * sigma
    team = z["team"]
    if team.ndim == 1:
        team = np.broadcast_to(team, z["players"].shape[:2])
    n = len(ball)
    hidden = np.ones(n, bool) if soccertrack else dropout_mask(n, rng)
    if N_IN <= 12:
        x = ball_feats(ball, players, team, hidden)[::STEP]
    else:
        x = features2(ball, players, team, hidden, z["period"])[:, :N_IN]
    return x, z["label"][::STEP].astype(np.int64)


def windows(x, y):
    out = []
    for s in range(0, max(1, len(x) - WIN + 1), STRIDE):
        xs, ys = x[s:s + WIN], y[s:s + WIN]
        if (ys >= 0).mean() > 0.3:
            out.append((xs, ys))
    return out


def spell(y, logp):
    path = viterbi(logp, LAM, MU).astype(np.int8)
    gt = np.where(y < 0, pm.X, y).astype(np.int8)
    runs = pm.valid_runs(gt)
    if len(runs) == 0:
        return float("nan")
    return pm.spell_scores(gt, path, runs)["mean_f1"]


def train_one(train, epochs, seed, evaluate=None, save_path=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(ARCH, n_in=N_IN)
    dev = torch.device("cuda" if (torch.cuda.is_available() and evaluate is None) else "cpu")
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    best, best_ep = -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for s in range(0, len(order), 32):
            batch = [train[i] for i in order[s:s + 32]]
            xb = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
            yb = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            loss = loss_fn(model(xb).reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(batch)
        msg = "epoch {:>2}  loss {:.3f}".format(ep, total / len(train))
        if evaluate:
            score, detail = evaluate(model)
            msg += "  val " + detail
            if score > best:
                best, best_ep = score, ep
                torch.save(model.state_dict(), save_path)
                msg += "  <- best, saved"
        print(msg, flush=True)
    model.cpu()
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-idsse", default="J03WPY")
    ap.add_argument("--test-idsse", default="J03WQQ")
    ap.add_argument("--val-st", default="132877")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-in", type=int, default=35)
    ap.add_argument("--no-st", action="store_true", help="train on IDSSE only")
    ap.add_argument("--st-copies", type=int, default=1)
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--archs", default=None, help="final mode only: arch:epochs,arch:epochs,... (features built once)")
    ap.add_argument("--arch", default="gru", choices=["gru", "lstm", "tcn", "convgru", "transformer"])
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    global N_IN, ARCH
    N_IN = args.n_in
    ARCH = args.arch
    os.makedirs(args.out_dir, exist_ok=True)

    idsse = sorted(glob.glob(os.path.join(IDSSE, "*.npz")))
    st = sorted(glob.glob(os.path.join(ST_DIR, "*.npz")))
    if not idsse or not st:
        raise SystemExit("need IDSSE npz in {} and SoccerTrack npz in {}".format(IDSSE, ST_DIR))
    held = [] if args.final else [args.val_idsse, args.test_idsse, args.val_st]
    rng = np.random.default_rng(0)

    print("building features (a few minutes) ...", flush=True)
    train = []
    for p in idsse:
        if any(h in p for h in held):
            continue
        for _ in range(args.copies):
            train += windows(*load2(p, rng, False))
    for p in st:
        if any(h in p for h in held):
            continue
        for _ in range(0 if args.no_st else args.st_copies):
            train += windows(*load2(p, rng, True))
    print("{} training windows".format(len(train)), flush=True)

    if args.final and args.archs:
        for item in args.archs.split(","):
            arch, ep = item.split(":")
            ARCH = arch
            for seed in range(args.seeds):
                out = os.path.join(args.out_dir, "possession_gru_{}_final_s{}.pt".format(arch, seed))
                print("\n=== {} seed {} ({} epochs) ===".format(arch, seed, ep), flush=True)
                model, _, _ = train_one(train, int(ep), seed)
                torch.save(model.state_dict(), out)
                print("saved", out, flush=True)
        return
    if args.final:
        for seed in range(args.seeds):
            out = os.path.join(args.out_dir, "possession_gru_{}_s{}.pt".format(args.tag, seed))
            model, _, _ = train_one(train, args.epochs, seed)
            torch.save(model.state_dict(), out)
            print("saved", out, flush=True)
        return

    vi = load2([p for p in idsse if args.val_idsse in p][0], np.random.default_rng(123), False)
    ti = load2([p for p in idsse if args.test_idsse in p][0], np.random.default_rng(456), False)
    vs = [load2(p, np.random.default_rng(7), True) for p in st if args.val_st in p]
    if not vs:
        raise SystemExit("no SoccerTrack match containing " + args.val_st)

    def score_all(model, with_test=False):
        a = spell(vi[1], half_logprobs(model, vi[0]))
        b = float(np.nanmean([spell(y, half_logprobs(model, x)) for x, y in vs]))
        return a, b

    def evaluate(model):
        a, b = score_all(model)
        return (a + b) / 2, "IDSSE {:.3f}  SoccerTrack(no ball) {:.3f}  mean {:.3f}".format(a, b, (a + b) / 2)

    path = os.path.join(args.out_dir, "possession_gru_{}_val.pt".format(args.tag))
    model, best, best_ep = train_one(train, args.epochs, 0, evaluate, path)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    print("\nbest epoch {} (val mean spellF1 {:.3f})".format(best_ep, best))
    print("TEST IDSSE {} (never used to choose anything): spellF1 {:.3f}".format(
        args.test_idsse, spell(ti[1], half_logprobs(model, ti[0]))))
    print("RESULT arch={} tag={} n_in={} no_st={} best_epoch={} val={:.3f} test_idsse={:.3f}".format(
        args.arch, args.tag, args.n_in, args.no_st, best_ep, best, spell(ti[1], half_logprobs(model, ti[0]))))


if __name__ == "__main__":
    main()
