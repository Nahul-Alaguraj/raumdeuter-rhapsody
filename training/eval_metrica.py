"""Score trained models on the Metrica sample games (external TEST data, never trained or tuned on).

Each model spec is  name:n_in:arch=glob[,arch=glob...]  -- every weight file matching a glob is loaded
and all of them are averaged (an ensemble; architectures can be mixed).
Two conditions per game: the ball exactly as tracked (about 40% already missing), and with an
extra simulated outage on top (the same generator used in training), averaged over a few seeds.
Scoring uses the official spell F1 (competition metric code) at 5 fps, per half, plus frame macro F1.

    python training/eval_metrica.py \
        v4:gru:35:training/possession_gru_v4_s*.pt \
        old:gru:12:training/possession_gru_final_s*.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from train_possession import features as ball_feats, dropout_mask, macro_f1, STEP, PITCH  # noqa: E402
from features2 import features2, make_model  # noqa: E402
from decode import half_logprobs, viterbi  # noqa: E402

DATA = os.path.join(ROOT, "data", "external", "metrica")
LAM, MU = 2.0, 4.0


def score(y, path):
    gt = np.where(y < 0, pm.X, y).astype(np.int8)
    runs = pm.valid_runs(gt)
    spell = pm.spell_scores(gt, path.astype(np.int8), runs)["mean_f1"] if len(runs) else float("nan")
    m = y >= 0
    return spell, macro_f1(y[m], path[m])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    specs = []
    for s in args.models:
        name, n_in, parts = s.split(":", 2)
        nets = []
        for part in parts.split(","):
            arch, pattern = part.split("=", 1)
            files = sorted(glob.glob(pattern))
            if not files:
                raise SystemExit("no weights match " + pattern)
            for f in files:
                m = make_model(arch, n_in=int(n_in)); m.load_state_dict(torch.load(f, map_location="cpu")); m.eval()
                nets.append(m)
        specs.append((name, int(n_in), nets))
        print("{}: {} model(s)".format(name, len(nets)))

    games = sorted(glob.glob(os.path.join(DATA, "game*.npz")))
    if not games:
        raise SystemExit("run training/build_metrica.py first")
    results = {(s[0], c): [] for s in specs for c in ("as tracked", "extra outage")}
    for g in games:
        z = np.load(g)
        ball, players, team, label, period = z["ball"], z["players"], z["team"], z["label"], z["period"]
        for cond, seeds in (("as tracked", [0]), ("extra outage", range(args.seeds))):
            for seed in seeds:
                rng = np.random.default_rng(seed)
                hidden = np.isnan(ball[:, 0])
                if cond == "extra outage":
                    hidden = hidden | dropout_mask(len(ball), rng)
                cache = {}
                for name, n_in, nets in specs:
                    key = n_in > 12
                    if key not in cache:
                        cache[key] = features2(ball, players, team, hidden, period) if key else \
                            ball_feats(ball, players, team, hidden)[::STEP]
                    x = cache[key][:, :n_in]
                    y = label[::STEP].astype(np.int64)
                    per = period[::STEP]
                    for p in np.unique(per):
                        sl = per == p
                        if (y[sl] >= 0).mean() < 0.2:
                            continue
                        lp = np.mean([half_logprobs(n, x[sl]) for n in nets], axis=0)
                        results[(name, cond)].append(score(y[sl], viterbi(lp, LAM, MU)))
    print("\n{:<10} {:<14} {:>9} {:>9}   (mean over games x halves x seeds)".format("model", "ball", "spellF1", "macroF1"))
    for (name, cond), v in results.items():
        a = np.array(v, dtype=float)
        print("{:<10} {:<14} {:>9.3f} {:>9.3f}   n={}".format(name, cond, np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), len(a)))


if __name__ == "__main__":
    main()
