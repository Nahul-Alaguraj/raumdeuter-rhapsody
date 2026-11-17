"""Download the open IDSSE / Sportec Bundesliga matches and turn each into a training file.

EXTERNAL data only (7 other matches, CC-BY 4.0, DFL / Sportec Solutions). Nothing from the
Barcelona-Athletic match is touched here, so it is fine to run this anywhere, Colab included.

For every frame we keep:
    ball    (n, 2)      ball x,y, normalised 0-1 (same convention as the competition ATD)
    players (n, K, 2)   up to K tracked players x,y, NaN where unused
    team    (n, K)      0 = home, 1 = away, -1 = unused slot
    label   (n,)        0 = H, 1 = A, 2 = D   <- from the tracking's own ball-state / owner fields
    period  (n,)        1 or 2

The label comes from fields that Sportec's tracking already carries (ball alive/dead and which
team owns the ball) -- the same kind of source the competition's ground truth is derived from.

    pip install kloppy numpy
    python training/build_idsse.py                  # all 7 matches, ~400 MB of XML each
    python training/build_idsse.py --matches J03WPY # just one, to test first

Each match takes a few minutes and a few GB of RAM to parse.
"""
import argparse
import os

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState, Team

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
K = 26                      # player slots per frame (22 + a few extras for subs/officials)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "data", "external", "idsse")   # data/ is gitignored


def convert(match_id):
    print("loading", match_id, "(downloads and parses a large XML, be patient) ...")
    ds = sportec.load_open_tracking_data(match_id=match_id, coordinates="metrica")
    n = len(ds.frames)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, K, 2), np.nan, dtype=np.float32)
    team = np.full((n, K), -1, dtype=np.int8)
    label = np.full(n, 2, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)

    for i, frame in enumerate(ds.frames):
        period[i] = frame.period.id
        if frame.ball_coordinates is not None:
            ball[i] = (frame.ball_coordinates.x, frame.ball_coordinates.y)
        owner = frame.ball_owning_team
        if frame.ball_state == BallState.ALIVE and owner is not None:
            label[i] = 0 if owner.ground.value == "home" else 1
        # else stays 2 = dead ball
        slot = 0
        for player, data in frame.players_data.items():
            if data.coordinates is None or slot >= K:
                continue
            players[i, slot] = (data.coordinates.x, data.coordinates.y)
            team[i, slot] = 0 if player.team.ground.value == "home" else 1
            slot += 1

    counts = {name: int(np.sum(label == code)) for name, code in (("H", 0), ("A", 1), ("D", 2))}
    print("  frames {}  labels {}  ball visible {:.1f}%".format(
        n, counts, 100.0 * np.mean(~np.isnan(ball[:, 0]))))
    return dict(ball=ball, players=players, team=team, label=label, period=period)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="+", default=MATCHES)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for match_id in args.matches:
        path = os.path.join(OUT_DIR, match_id + ".npz")
        if os.path.exists(path):
            print("skip", match_id, "(already built)")
            continue
        np.savez_compressed(path, **convert(match_id))
        print("  wrote", path)


if __name__ == "__main__":
    main()
