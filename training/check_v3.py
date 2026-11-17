"""Why does v3 say 'home' everywhere on the competition match?  Describes, changes nothing.

Prints, per half: NaNs in the features, the class shares of the RAW model output (v2 and v3
side by side), and the shares after decoding with a few penalty settings.

    python training/check_v3.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from predict_match import read_atd  # noqa: E402
from train_possession import features, GRU, STEP  # noqa: E402
from decode import half_logprobs, viterbi  # noqa: E402


def share(p):
    return "H {:4.0%} A {:4.0%} D {:4.0%}".format(*(np.mean(p == c) for c in (0, 1, 2)))


def main():
    meta, raw = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    slot_team = team_of_each_track(meta)
    ball, players = read_atd(raw, n, len(slot_team))
    models = {}
    for name in ("v2", "v3"):
        m = GRU(); m.load_state_dict(torch.load(os.path.join(HERE, "possession_gru_{}.pt".format(name)), map_location="cpu"))
        models[name] = m

    for half, a, b in half_video_blocks(GAME):
        sl = slice(a, b + 1)
        team = np.broadcast_to(slot_team, (b - a + 1, len(slot_team)))
        x = features(ball[sl], players[sl], team, np.isnan(ball[sl, 0]))[::STEP]
        print("\n=== half {} ===  steps {}  NaN in features: {}  inf: {}".format(
            half, len(x), int(np.isnan(x).sum()), int(np.isinf(x).sum())))
        print("  feature means:", np.round(np.nanmean(x, axis=0), 2))
        for name, m in models.items():
            lp = half_logprobs(m, x)
            print("  {} raw        : {}   (NaN in output: {})".format(name, share(lp.argmax(1)), int(np.isnan(lp).sum())))
            for lam, mu in ((0, 0), (2, 4), (4, 1)):
                print("  {} decode {},{}: {}".format(name, lam, mu, share(viterbi(lp, lam, mu))))
    print()


if __name__ == "__main__":
    main()
