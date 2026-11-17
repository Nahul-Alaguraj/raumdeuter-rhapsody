"""What does a dead ball look like in the competition tracking, versus in IDSSE?

This only DESCRIBES the data so we can understand the task. It changes no parameter and trains
nothing. The public ground truth is read here only to learn what "dead" means (RULES allow
inspecting data to understand possession); nothing is fitted to it.

    python training/diagnose_dead.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from predict_match import read_atd  # noqa: E402

PITCH = np.array([105.3, 68.0], dtype=np.float32)
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
IDSSE = os.path.join(ROOT, "data", "external", "idsse", "J03WPY.npz")


def describe(name, ball, players, mask):
    """ball visibility, ball speed and nearest-player distance over the frames in `mask`."""
    n = int(mask.sum())
    vis = ~np.isnan(ball[:, 0])
    idx = np.where(mask)[0]
    b = ball * PITCH
    sp = np.full(len(ball), np.nan, dtype=np.float32)
    sp[5:] = np.linalg.norm(b[5:] - b[:-5], axis=1) * 5.0       # m/s over 5 frames @25fps
    d = np.linalg.norm(players * PITCH - b[:, None, :], axis=2)
    near = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    near[~vis] = np.nan
    tracked = np.sum(~np.isnan(players[:, :, 0]), axis=1)
    f = lambda a: "{:6.2f}".format(np.nanmedian(a[idx])) if np.any(~np.isnan(a[idx])) else "   n/a"
    print("  {:<22} frames {:>6}  ball seen {:5.1f}%  speed med {} m/s  nearest-player med {} m  players {:4.1f}"
          .format(name, n, 100.0 * vis[idx].mean(), f(sp), f(near), tracked[idx].mean()))


def gap_stats(missing, label):
    runs, cur = [], 0
    for m in missing:
        if m: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    r = np.array(runs) / 25.0
    if len(r) == 0:
        print("  {}: no gaps".format(label)); return
    long2 = r[r > 2].sum() / r.sum(); long6 = r[r > 6].sum() / r.sum()
    print("  {}: {} gaps, median {:.2f}s p90 {:.1f}s p99 {:.1f}s | share of missing time in gaps >2s: {:.0%}, >6s: {:.0%}"
          .format(label, len(r), np.median(r), np.percentile(r, 90), np.percentile(r, 99), long2, long6))


def main():
    meta, raw = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    ball, players = read_atd(raw, n, len(team_of_each_track(meta)))

    graded = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    print("\n=== Competition tracking, ALL graded frames (no labels used) ===")
    describe("graded", ball, players, graded)
    for half, a, b in half_video_blocks(GAME):
        gap_stats(np.isnan(ball[a:b + 1, 0]), "ball-missing gaps, half {}".format(half))

    print("\n=== Competition tracking, public 5-minute window, split by the public labels ===")
    lab = np.array([l.strip() for l in np.loadtxt(GT, delimiter=",", skiprows=1, dtype=str, usecols=1)])
    for c, nm in (("H", "home in possession"), ("A", "away in possession"), ("D", "dead ball")):
        describe(nm, ball, players, lab == c)
    gap_stats(np.isnan(ball[lab == "D", 0]), "gaps inside D frames")
    alive = (lab == "H") | (lab == "A")
    gap_stats(np.isnan(ball[alive, 0]), "gaps inside alive frames")

    if os.path.exists(IDSSE):
        z = np.load(IDSSE)
        print("\n=== IDSSE match J03WPY (external), split by its labels ===")
        for c, nm in ((0, "home in possession"), (1, "away in possession"), (2, "dead ball")):
            describe(nm, z["ball"], z["players"], z["label"] == c)
    print()


if __name__ == "__main__":
    main()
