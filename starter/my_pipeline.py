"""Version 1 of our possession pipeline (rule-based, no training, no ground truth used).

Idea, in plain words:
  1. On each frame where the ball is tracked, find the nearest player. If that player is close
     enough (CLOSE_M metres), that is a "vote" for their team. Otherwise: no vote.
  2. Keep the current possessor. Only hand the ball to the other team when the other team has
     been the one with the vote for HOLD frames in a row (ignoring frames with no vote).
     This stops the flicker that ruins spellF1.
  3. If the ball has been untracked for DEAD_GAP frames in a row, call it dead ball (D) until a
     vote shows up again.
  4. Frames outside the two graded halves are X.

All three numbers below are fixed, football-common-sense values. They are NOT tuned on the match
or on the public ground truth. Tune them later on external data only (see RULES.md).

    python starter/my_pipeline.py
    python evaluation/validate_submission.py evaluation/results/my_v1.csv
    python evaluation/run_evaluation.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, write_submission, FIELD_SEP, GROUP_SEP, MISSING  # noqa: E402

CLOSE_M = 3.0      # nearest player must be within this many metres to count as a vote
HOLD = 40          # frames (1.6 s) the other team must keep the votes before we switch
DEAD_GAP = 150     # frames (6 s) with no ball at all before we say dead ball

OUT = os.path.join(ROOT, "evaluation", "results", "my_v1.csv")


def votes(raw_path, team_of_track, n_frames, pitch):
    """Per frame: 0 = home vote, 1 = away vote, -1 = no vote, -2 = ball not tracked."""
    length, width = pitch
    out = np.full(n_frames, -2, dtype=np.int8)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            frame = int(parts[0]) - 1
            c = parts[2].index(",")
            bx = float(parts[2][:c]) * length
            by = float(parts[2][c + 1:parts[2].index(",", c + 1)]) * width
            out[frame] = -1
            best, team = None, -1
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                c = group.index(",")
                dx = float(group[:c]) * length - bx
                dy = float(group[c + 1:group.index(",", c + 1)]) * width - by
                d = dx * dx + dy * dy
                if best is None or d < best:
                    best, team = d, team_of_track[slot]
            if best is not None and best <= CLOSE_M ** 2:
                out[frame] = team
    return out


def state_machine(v, first, last):
    """Labels for frames first..last (inclusive) of one half."""
    labels = np.full(last - first + 1, "X", dtype="<U1")
    current = None            # 0 home, 1 away
    challenger, streak = None, 0
    gap = 0
    for i, f in enumerate(range(first, last + 1)):
        x = v[f]
        if x == -2:
            gap += 1
        else:
            gap = 0
        if x in (0, 1):
            if current is None:
                current = int(x)
            elif x != current:
                if challenger == x:
                    streak += 1
                else:
                    challenger, streak = int(x), 1
                if streak >= HOLD:
                    current, challenger, streak = int(x), None, 0
            else:
                challenger, streak = None, 0
        if gap >= DEAD_GAP:
            labels[i] = "D"
        elif current is None:
            labels[i] = "X"
        else:
            labels[i] = "H" if current == 0 else "A"
    return labels


def main():
    meta, raw = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    print("reading tracking ...")
    v = votes(raw, team_of_each_track(meta), n, GAMES[GAME]["pitch"])
    out = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        out[first:last + 1] = state_machine(v, first, last)
    write_submission(OUT, out)
    for lab in "HADX":
        print(lab, int(np.sum(out == lab)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
