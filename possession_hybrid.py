"""Final submission pipeline: rule-based evidence (possession_rules.py) plus neural networks
trained ONLY on external data (training/, see SOLUTION.md).

Nothing is learned from the competition match here:
  * stoppage spotting, restart takers, Smart Tagging votes, last-controller rule and the
    minimum-duration decoder are fixed rules written from football logic;
  * the team and dead-ball probabilities come from an ensemble of a TCN and a ConvGRU
    (3 seeds each), trained on IDSSE and SoccerTrack v2 matches only;
  * the two "restart taker" weights are generic round numbers.

Regenerate the submission file (needs the competition assets in data/, see README.md):

    python possession_hybrid.py --final --name <team_name>      # writes evaluation/results/<team_name>.csv
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "training"))
sys.path.insert(0, os.path.join(HERE, "starter"))
import possession_rules as P  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from train_possession import features as gru_features, GRU, STEP  # noqa: E402
from decode import half_logprobs  # noqa: E402
from features2 import features2, make_gru2, make_model  # noqa: E402
from baseline import FIELD_SEP, GROUP_SEP, MISSING  # noqa: E402   (ATD file format constants)

import glob  # noqa: E402
V4 = sorted(glob.glob(os.path.join(HERE, "training", "possession_gru_v4_s*.pt")))
FINAL = sorted(glob.glob(os.path.join(HERE, "training", "possession_gru_final_s*.pt")))
USE_V4 = bool(V4) and os.environ.get("GRU_VERSION", "v4") == "v4"     # GRU_VERSION=v3 forces old model
WEIGHTS_LIST = V4 if USE_V4 else (FINAL or [os.path.join(HERE, "training", "possession_gru_v3.pt")])
FINAL_SPEC = ("tcn=training/possession_gru_tcn_final_s*.pt,"
              "convgru=training/possession_gru_convgru_final_s*.pt")
PARAMS = dict(P.PARAMS, taker_visible=2.0, taker_hidden=1.0)    # generic, not fitted to this match


def read_atd(raw_path, n_frames, n_slots):
    """ball (n,2) and players (n,n_slots,2), normalised 0-1, NaN where not tracked (input only)."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if parts[2][0] != MISSING:
                a = parts[2].split(",")
                ball[f] = (float(a[0]), float(a[1]))
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot < n_slots and group[0] != MISSING:
                    a = group.split(",")
                    players[f, slot] = (float(a[0]), float(a[1]))
    return ball, players


def load_spec(spec):
    """'arch=glob,arch=glob' -> list of loaded 35-feature networks (mixed architectures allowed)."""
    nets = []
    for part in spec.split(","):
        arch, pattern = part.split("=", 1)
        files = sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(HERE, pattern)))
        if not files:
            raise SystemExit("no weights match " + pattern)
        for f in files:
            m = make_model(arch, n_in=35)
            m.load_state_dict(torch.load(f, map_location="cpu"))
            m.eval()
            nets.append(m)
    return nets


def gru_probabilities(n, slot_team, spec=None):
    """P(home | alive) and P(dead) for every graded frame, from the externally trained GRU."""
    from games import require
    _, raw = require(["atd_meta", "atd_raw"])
    ball, players = read_atd(raw, n, len(slot_team))
    if spec:
        models = load_spec(spec)
        print("averaging {} model(s) from: {}".format(len(models), spec))
    else:
        models = []
        for w in WEIGHTS_LIST:
            m = make_gru2() if USE_V4 else GRU()
            m.load_state_dict(torch.load(w, map_location="cpu"))
            models.append(m)
        print("averaging {} GRU model(s): {}".format(len(models), [os.path.basename(w) for w in WEIGHTS_LIST]))
    p_home = np.full(n, np.nan)
    p_dead = np.full(n, np.nan)
    for _, a, b in half_video_blocks(GAME):
        sl = slice(a, b + 1)
        team = np.broadcast_to(slot_team, (b - a + 1, len(slot_team)))
        hid = np.isnan(ball[sl, 0])
        if USE_V4 or spec:
            x = features2(ball[sl], players[sl], team, hid, None)     # orientation found per half
        else:
            x = gru_features(ball[sl], players[sl], team, hid)[::STEP]
        lp = np.mean([half_logprobs(m, x) for m in models], axis=0)   # average over seeds
        prob = np.exp(lp)
        prob = prob / prob.sum(1, keepdims=True)                # (steps, 3): H, A, D
        up = lambda v: np.repeat(v, STEP)[:b - a + 1]
        p_home[sl] = up(prob[:, 0] / (prob[:, 0] + prob[:, 1] + 1e-9))
        p_dead[sl] = up(prob[:, 2])
    return p_home, p_dead


def build_evidence_hybrid(with_gru=True):
    ball, players, teams = P.load_tracking()
    tags = P.load_tags()
    n = len(ball)
    ball, players = P.rotate(ball, players, teams)

    ctrl, _, _ = P.ball_control(ball, players, teams)
    out, still, seen, ball_speed = P.ball_state(ball)
    tempo_2s = P.rolling_mean(np.nanmean(P.player_speed(players), 1), P.FPS)
    stopped, restart_frames = P.stoppages(ball, out, still, seen, ball_speed, tempo=tempo_2s)
    restart = np.zeros(n, dtype=bool)
    for r in restart_frames:
        restart[r:r + P.FPS] = True
    tag_vote = P.tag_evidence(tags, n)

    kickoffs = P.kickoff_frames(tags, ball)
    before_kickoff = np.zeros(n, dtype=bool)
    for kind, frame in kickoffs:
        if kind == "kick off (start)":
            half_start = max(a for a, _ in P.HALVES if a <= frame)
            before_kickoff[half_start:frame] = True

    taker_visible, taker_hidden = P.restart_takers(
        ball, players, teams, sorted(set(restart_frames) | {f for _, f in kickoffs}))
    for frame, side in P.conceding_kickoff(tags, kickoffs).items():
        taker_hidden[frame:frame + 2 * P.FPS] += side

    p_home, p_dead = gru_probabilities(n, teams) if with_gru else (None, None)
    return {"n": n, "ctrl": ctrl, "out": out, "still": still, "seen": seen,
            "tag_vote": tag_vote, "before_kickoff": before_kickoff, "stopped": stopped,
            "restart": restart, "taker_visible": taker_visible, "taker_hidden": taker_hidden,
            "p_home": p_home, "p_dead": p_dead, "teams": teams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hybrid_v1")
    ap.add_argument("--final", action="store_true", help="the submitted model set (TCN + ConvGRU, 3 seeds each)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="name::arch=glob,arch=glob ... one CSV per variant; rules are computed once")
    args = ap.parse_args()
    if args.final:
        args.variants = [args.name + "::" + FINAL_SPEC]
    if not args.variants:
        ev = build_evidence_hybrid()
        labels = P.decide(ev, PARAMS)
        P.describe(labels)
        path = os.path.join(HERE, "evaluation", "results", args.name + ".csv")
        P.write(path, labels)
        print("wrote", os.path.relpath(path, HERE))
        return
    ev = build_evidence_hybrid(with_gru=False)
    for v in args.variants:
        name, spec = v.split("::", 1)
        ev["p_home"], ev["p_dead"] = gru_probabilities(ev["n"], ev["teams"], spec)
        labels = P.decide(ev, PARAMS)
        path = os.path.join(HERE, "evaluation", "results", name + ".csv")
        P.write(path, labels)
        print("wrote", os.path.relpath(path, HERE), flush=True)


if __name__ == "__main__":
    main()
