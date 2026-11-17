"""Run the trained possession model on the competition match and write a submission CSV.

Runs on YOUR machine against the provided tracking. The model weights come from training on
external IDSSE data only (train_possession.py); nothing here is fitted to this match.

    python training/predict_match_v3.py                 # writes evaluation/results/gru_v3.csv
    python evaluation/validate_submission.py evaluation/results/gru_v3.csv
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, write_submission, FIELD_SEP, GROUP_SEP, MISSING  # noqa: E402
import json  # noqa: E402
from train_possession import features, GRU, STEP  # noqa: E402
from decode import half_logprobs, viterbi  # noqa: E402

WEIGHTS = os.path.join(HERE, "possession_gru_v3.pt")
OUT = os.path.join(ROOT, "evaluation", "results", "gru_v3.csv")
LETTERS = np.array(["H", "A", "D"])


def read_atd(raw_path, n_frames, n_slots):
    """ball (n,2) and players (n,n_slots,2), normalised 0-1, NaN where not tracked."""
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


def main():
    meta, raw = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    slot_team = team_of_each_track(meta)
    print("reading tracking ...")
    ball, players = read_atd(raw, n, len(slot_team))
    print("  ball tracked on {:.1f}% of all frames".format(100.0 * np.mean(~np.isnan(ball[:, 0]))))

    model = GRU()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    with open(os.path.join(HERE, "possession_decode_v3.json")) as handle:
        cfg = json.load(handle)
    print("decode penalties (chosen on external data): lam {} mu {}".format(cfg["lam"], cfg["mu"]))

    out = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        team = np.broadcast_to(slot_team, (last - first + 1, len(slot_team)))
        hidden = np.isnan(ball[sl, 0])                   # the REAL missing ball, not simulated
        x = features(ball[sl], players[sl], team, hidden)[::STEP]
        pred = np.repeat(viterbi(half_logprobs(model, x), cfg["lam"], cfg["mu"]), STEP)[:last - first + 1]
        out[sl] = LETTERS[pred]
        print("  half {} done".format(half))

    write_submission(OUT, out)
    for lab in "HADX":
        print(lab, int(np.sum(out == lab)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
