"""Metrica Sports public sample games 1 and 2 -> external TEST files (never trained on).

Source: https://github.com/metrica-sports/sample-data  (anonymised; "acknowledge the source").
Ball is tracked on essentially every frame; labels are derived from the hand-collected event data:

  * a team owns the ball from the start of its PASS / SHOT / RECOVERY / SET PIECE until the next
    event, if that gap is at most 10 s; longer gaps stay unknown (-1) and are not scored
  * after a BALL OUT, or after a goal, the ball is dead until the next event (at most 90 s)
  * the gap before a free kick is left unknown (the whistle is not an event)

    python training/build_metrica.py        # downloads about 130 MB, writes data/external/metrica/
"""
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "external", "metrica_raw")
OUT = os.path.join(ROOT, "data", "external", "metrica")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{g}/Sample_Game_{g}_{f}.csv"
FPS = 25
OWN = {"PASS", "SHOT", "RECOVERY", "SET PIECE"}
MAX_OWN_GAP, MAX_DEAD = 10 * FPS, 90 * FPS


def fetch(g, name):
    path = os.path.join(RAW, "g{}_{}.csv".format(g, name))
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(RAW, exist_ok=True)
        print("downloading", name, "game", g)
        urllib.request.urlretrieve(URL.format(g=g, f=name), path)
    return path


def read_tracking(path):
    d = pd.read_csv(path, skiprows=3, header=None)
    period = d[0].to_numpy(np.int8)
    xy = d.iloc[:, 3:].to_numpy(np.float32)          # player x,y pairs..., then ball x,y
    return period, xy[:, :-2].reshape(len(d), -1, 2), xy[:, -2:]


def build_labels(ev, n):
    """0 home, 1 away, 2 dead, -1 unknown, at tracking frame resolution."""
    ev = ev.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)
    lab = np.full(n, -1, dtype=np.int8)
    rows = list(ev.itertuples(index=False))
    use = [r for r in rows if r.Type in OWN or r.Type == "BALL OUT"]
    for k in range(len(use) - 1):
        r, nxt = use[k], use[k + 1]
        f, f2 = max(int(r._4) - 1, 0), min(int(nxt._4) - 1, n)
        if f2 <= f:
            continue
        team = 0 if r.Team == "Home" else 1
        goal = r.Type == "SHOT" and "GOAL" in str(r.Subtype) and "GOAL KICK" not in str(r.Subtype)
        if r.Type == "BALL OUT" or goal:
            if goal:
                f = max(int(r._6) - 1, f)        # the ball is still live until the shot ends
            if f2 - f <= MAX_DEAD:
                lab[f:f2] = 2
        elif nxt.Type == "SET PIECE" and str(nxt.Subtype) == "FREE KICK":
            continue                              # play stopped for a foul, exact moment unknown
        elif f2 - f <= MAX_OWN_GAP:
            lab[f:f2] = team
    return lab


def convert(g):
    ph, home, ball = read_tracking(fetch(g, "RawTrackingData_Home_Team"))
    _, away, _ = read_tracking(fetch(g, "RawTrackingData_Away_Team"))
    n = len(ph)
    players = np.concatenate([home, away], axis=1)
    team = np.concatenate([np.zeros(home.shape[1]), np.ones(away.shape[1])]).astype(np.int8)
    ev = pd.read_csv(fetch(g, "RawEventsData"))
    lab = build_labels(ev, n)
    kn = lab >= 0
    runs = np.diff(np.r_[0, (lab == 0) | (lab == 1), 0].astype(int))
    print("game {}: frames {} | labelled {:.0%} | home {:.0%} away {:.0%} dead {:.0%} of labelled | ball seen {:.1%}".format(
        g, n, kn.mean(), (lab == 0).sum() / kn.sum(), (lab == 1).sum() / kn.sum(), (lab == 2).sum() / kn.sum(),
        np.mean(~np.isnan(ball[:, 0]))))
    return dict(ball=ball, players=players, team=np.broadcast_to(team, players.shape[:2]).copy(), label=lab, period=ph)


def main():
    os.makedirs(OUT, exist_ok=True)
    for g in (1, 2):
        np.savez_compressed(os.path.join(OUT, "game{}.npz".format(g)), **convert(g))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
