"""Rule-based evidence and decoder used by the submission (possession_hybrid.py).

Everything here is fixed football logic written by hand: loading the tracking and Smart Tagging,
stoppage and restart spotting, last-controller rule, and the minimum-duration Viterbi decoder.
Nothing in this file is fitted to, or learns from, the competition match. The team and dead-ball
probabilities come from neural networks trained only on external data (see training/).
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1


def track_teams(meta_path):
    """Team (0 home, 1 away) of every track slot, from the ATD metadata."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def load_tracking(game=GAME):
    """ball (N,2) and players (N,S,2) in metres, NaN where missing, plus team of each slot."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    cache = os.path.join(ROOT, "data", "cache_tactical.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        z = np.load(cache)
        return z["ball"], z["players"], z["teams"]

    teams = track_teams(meta_path)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    slots = len(teams)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, slots, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1  # ATD frame_id counts from 1
            b = parts[2].split(",")
            ball[f] = (float(b[0]) * length, float(b[1]) * width)
            for s, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                g = group.split(",")
                players[f, s] = (float(g[0]) * length, float(g[1]) * width)

    np.savez_compressed(cache, ball=ball, players=players, teams=teams)
    return ball, players, teams


def load_tags(game=GAME):
    """Smart Tagging as a list of dicts: code, start, end (video frames), team, type."""
    path = require(["smart_tagging"], game)[0]
    fps = GAMES[game]["fps"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {}
        for lab in inst.iter("label"):
            group = (lab.findtext("group") or "").strip().lower()
            tags[group] = (lab.findtext("text") or "").strip()
        team = tags.get("team", "").lower()
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team": HOME if team == "barcelona" else AWAY if team.startswith("at") else None,
            "type": tags.get("type", "").lower(),
        })
    return out


# ==================== features ====================
"""Turn raw tracking into per-frame evidence.

Everything here is computed by code from the tracking and Smart Tagging. Nothing is hand labelled.

Coordinates are rotated per half so that the HOME team always attacks towards +x. That way a
feature like "home centroid x" means the same thing in both halves.
"""

import warnings

import numpy as np


warnings.filterwarnings("ignore", category=RuntimeWarning)

FPS = 25
PITCH = (105.3, 68.0)
HALVES = ((11185, 90652), (110808, 190198))  # graded video frames, end exclusive

# Tag codes that name the team WITH the ball, and the ones that name the team WITHOUT it.
ATTACK_CODES = {"BUILD UP", "PROGRESSION", "ATTACKING TRANSITION", "BALL IN FINAL THIRD",
                "CREATING CHANCES", "LONG BALL", "BALL IN THE BOX", "PLAYERS IN THE BOX",
                "GOALS", "SET PIECES"}
DEFEND_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}

# A player "controls" the ball when he is this close and nobody from the other team is near.
CONTROL_RADIUS = 1.0      # metres
CONTEST_RADIUS = 2.0      # metres, nearest opponent must be further than this


def half_of(n):
    """Array: 1 or 2 for graded frames of each half, 0 for ungraded frames."""
    h = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(HALVES, 1):
        h[a:b] = i
    return h


def rotate(ball, players, teams):
    """Rotate each half so home attacks +x. Returns copies."""
    ball, players = ball.copy(), players.copy()
    L, W = PITCH
    for a, b in HALVES:
        home_x = np.nanmean(players[a:b, teams == HOME, 0])
        away_x = np.nanmean(players[a:b, teams == AWAY, 0])
        if home_x > away_x:  # home defends the right goal in this half, so flip it
            for arr in (ball[a:b], players[a:b]):
                arr[..., 0] = L - arr[..., 0]
                arr[..., 1] = W - arr[..., 1]
    return ball, players


def rolling_mean(x, half_width):
    """Centred moving average along axis 0 that ignores NaN."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    k = 2 * half_width + 1
    pad = [(half_width, half_width)] + [(0, 0)] * (x.ndim - 1)
    cs = np.cumsum(np.pad(filled, pad), axis=0)
    cn = np.cumsum(np.pad(valid.astype(np.float64), pad), axis=0)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:]), cs])
    cn = np.concatenate([np.zeros((1,) + cn.shape[1:]), cn])
    s = cs[k:] - cs[:-k]
    c = cn[k:] - cn[:-k]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def ball_control(ball, players, teams):
    """Per frame: HOME or AWAY if one player clearly has the ball, else -1. Also distances."""
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    dh = d[:, teams == HOME].min(1)
    da = d[:, teams == AWAY].min(1)
    ctrl = np.full(len(ball), -1, dtype=np.int8)
    ctrl[(dh < CONTROL_RADIUS) & (da > CONTEST_RADIUS)] = HOME
    ctrl[(da < CONTROL_RADIUS) & (dh > CONTEST_RADIUS)] = AWAY
    return ctrl, dh, da








def tag_evidence(tags, n):
    """Per frame team vote from Smart Tagging in [-1, 1]: +1 home has it, -1 away has it."""
    votes = np.zeros(n)
    count = np.zeros(n)
    for t in tags:
        if t["team"] is None:
            continue
        if t["code"] in ATTACK_CODES:
            side = 1.0 if t["team"] == HOME else -1.0
        elif t["code"] in DEFEND_CODES:
            side = -1.0 if t["team"] == HOME else 1.0
        else:
            continue
        votes[t["start"]:t["end"]] += side
        count[t["start"]:t["end"]] += 1
    return np.where(count > 0, votes / np.maximum(count, 1), 0.0)


def kickoff_frames(tags, ball):
    """Estimated restart frame of each 'Kick Off' tag: when the ball first moves inside it."""
    v = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    out = []
    for t in tags:
        if not t["type"].startswith("kick off"):
            continue
        a, b = t["start"], t["end"]
        moving = np.where(v[a:b] > 3.0)[0]
        moving = moving[moving > 0]
        out.append((t["type"], a + int(moving[0]) if len(moving) else (a + b) // 2))
    return out


def stoppages(ball, out, still, seen, ball_speed, min_out=8, max_back=15 * FPS,
              kick_speed=4.0, kick_frames=3, max_hole=FPS, max_wait=60 * FPS,
              tempo=None, live_tempo=2.2, grace=10 * FPS):
    """Stoppages we can SEE: [start, end) spans where the ball is surely dead.

    * Ball out of the pitch for at least `min_out` frames. Dead from the first out frame.
    * Ball lying still for 2 s (placed for a free kick, goal kick, corner, kickoff). Dead back to
      when the ball was last moving (the foul or the ball going out), at most 15 s back.

    A stoppage lasts until the RESTART: a clear kick or throw (ball faster than `kick_speed` m/s
    for `kick_frames` frames in a row), never taken from behind a goal line. A keeper walking
    with the ball, or the ball being rolled back onto the pitch over the goal line, is not a
    restart. If the tracker loses
    the ball for more than `max_hole` frames, the visible stoppage stops there (the bridge below
    and the dead ball model take over). Safety brake: if `tempo` (players' mean speed, 2 s
    average) is given, a stoppage also ends once players move at live play pace (`live_tempo`), after a `grace`
    period at its start while players slow down.
    Returns a boolean mask and the list of restart frames.
    """
    n = len(ball)
    dead = np.zeros(n, dtype=bool)
    restarts = []
    v = np.nan_to_num(ball_speed, nan=0.0)
    slow = seen & (v < 2.0)
    fast = seen & (v > kick_speed)
    # A restart is the START of a clear kick or throw (fast for `kick_frames` frames in a row,
    # so jitter while a keeper walks with the ball does not count), never taken from behind a
    # goal line (only throw ins come from outside, over a touchline).
    L, _ = PITCH
    kicked = np.zeros(n, dtype=bool)
    edges = np.diff(np.r_[0, fast.astype(np.int8), 0])
    for r, r_end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if r_end - r < kick_frames or r < 10:
            continue
        x = ball[r, 0]
        behind_goal_line = x < -0.5 or x > L + 0.5
        if not behind_goal_line:
            kicked[r] = True

    def runs(mask):
        d = np.diff(np.r_[0, mask.astype(np.int8), 0])
        return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))

    busy = np.zeros(n, dtype=bool) if tempo is None else np.nan_to_num(tempo) > live_tempo

    def until_restart(e, origin):
        """From frame e, walk forward to the restart. Returns (end, restarted)."""
        start, hole = e, 0
        while e < n and e - start < max_wait:
            if kicked[e] and not out[e]:
                return e, True
            if busy[e] and e - origin > grace:      # players need a few seconds to slow down
                return e, False
            hole = hole + 1 if not seen[e] else 0
            if hole > max_hole:
                return e - hole + 1, False
            e += 1
        return e, False

    for a, b in runs(out):
        if b - a < min_out:
            continue
        e, ok = until_restart(b, a)
        dead[a:e] = True
        if ok:
            restarts.append(e)

    for a, b in runs(still):
        s = a
        while s > max(0, a - max_back) and slow[s - 1] and not out[s - 1]:
            s -= 1
        e, ok = until_restart(b, s)
        dead[s:e] = True
        if ok:
            restarts.append(e)

    restarts = [r for r in restarts if not dead[r]]   # a restart inside another stoppage is not one

    # Bridge: dead just before the tracker lost the ball, and when it finds the ball again it is
    # lying still or being placed -> nothing restarted in between, so the gap is dead too.
    for a, b in runs(~seen):
        if a < FPS or b >= n - 5 or not dead[a - FPS:a].any():   # dead within 1 s before
            continue
        if dead[b] or slow[b:b + 5].all():
            e = b
            while e < n and slow[e]:
                e += 1
            last = a - FPS + int(np.flatnonzero(dead[a - FPS:a])[-1])  # last dead frame
            dead[last:e] = True
            restarts = [r for r in restarts if not last <= r < e]
            if e < n and seen[e]:
                restarts.append(e)

    return dead, sorted(set(restarts))


def ball_state(ball):
    """Ball out of pitch, ball still (stationary for 2 s), ball seen, ball speed."""
    L, W = PITCH
    seen = ~np.isnan(ball[:, 0])
    margin = 0.5
    out = seen & ((ball[:, 0] < -margin) | (ball[:, 0] > L + margin) |
                  (ball[:, 1] < -margin) | (ball[:, 1] > W + margin))
    v = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    slow = (v < 0.5).astype(float)
    slow[np.isnan(v)] = np.nan
    still = rolling_mean(slow, 25) > 0.95
    still &= seen
    return out, still, seen, v


def player_speed(players):
    """Speed of every track slot in m/s, NaN where unknown or impossible (identity swap)."""
    vel = np.empty_like(players)
    vel[1:-1] = (players[2:] - players[:-2]) * (FPS / 2.0)
    vel[0] = vel[-1] = np.nan
    speed = np.linalg.norm(vel, axis=2)
    speed[speed > 12.0] = np.nan
    return speed


def restart_takers(ball, players, teams, restarts, after=2 * FPS):
    """Who takes each restart, as a per frame vote: +1 home, -1 away, 0 nothing.

    visible : the restart is seen, so the taker is the player on the ball (within 2.5 m) just
              before it moves. (In the original experiment this was compared with ball-proximity labels on the
    competition match; the submission does NOT use that measurement.)
    hidden  : the ball is often lost during a stoppage, but the taker still shows up in the
              players: during a slow moment one team has a player standing just off the pitch
              (throw in) or at a corner flag. Checked the same way: right about 80% of the time.
    The vote covers the taker moment and the next `after` frames, when his team has the ball.
    """
    L, W = PITCH
    n = len(ball)
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    dh, da = d[:, teams == HOME].min(1), d[:, teams == AWAY].min(1)
    near_home = dh < da
    nearest = np.minimum(dh, da)

    visible = np.zeros(n)
    for r in restarts:
        w = slice(max(0, r - 10), min(n, r + 3))
        ok = nearest[w] < 2.5
        if ok.any():
            side = 1.0 if near_home[w][ok].mean() >= 0.5 else -1.0
            visible[r:r + after] = side

    speed = player_speed(players)
    tempo = rolling_mean(np.nanmean(speed, 1), FPS)
    x, y = players[..., 0], players[..., 1]
    margin = 0.3
    off = (y < -margin) | (y > W + margin) | (x < -margin) | (x > L + margin)
    off |= (np.minimum(x, L - x) < 3) & (np.minimum(y, W - y) < 3)      # at a corner flag
    off &= speed < 1.5                                                  # standing, not running
    off_home = off[:, teams == HOME].any(1)
    off_away = off[:, teams == AWAY].any(1)
    one = (off_home ^ off_away) & (np.nan_to_num(tempo, nan=9.0) < 1.3)
    hidden = np.zeros(n)
    edges = np.diff(np.r_[0, one.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < FPS:
            continue
        side = 1.0 if off_home[a:b].mean() >= 0.5 else -1.0
        hidden[a:b + after] = side
    return visible, hidden


def conceding_kickoff(tags, kickoffs, after=2 * FPS):
    """After a goal the team that conceded kicks off. GOALS tags name the scoring team.

    Vote +1 home / -1 away from each 'kick off (after goal)' restart for `after` frames.
    """
    n_vote = {}
    goals = [t for t in tags if t["code"] == "GOALS" and t["team"] is not None]
    for kind, frame in kickoffs:
        if kind != "kick off (after goal)":
            continue
        before = [g for g in goals if g["start"] <= frame <= g["end"] + 10 * FPS]
        if before:
            scorer = before[-1]["team"]
            n_vote[frame] = 1.0 if scorer == AWAY else -1.0     # the other team kicks off
    return n_vote


def last_controller(ctrl, stopped, min_run=3, max_hole=3, min_keep=12, fade=2 * FPS):
    """The rule from the challenge: a loose ball stays with the last team that controlled it.

    ctrl    : per frame HOME / AWAY / -1 from ball_control().
    A touch is a run of at least `min_run` control frames (holes of up to `max_hole` allowed).
    A team WINS the ball when its consecutive touches (before the other team touches it) add up
    to `min_keep` frames (half a second), or come in two or more touches (it played on, e.g.
    intercepted and passed). A single short touch, like a header won but not kept, is ignored
    and the ball stays with the team that had it. Ownership starts at the first touch.

    Returns
      side      : +1 home, -1 away, 0 nobody yet (or reset by a stoppage)
      freshness : 1 while the team is touching the ball, fading to 0 over `fade` frames after
    """
    n = len(ctrl)
    side = np.zeros(n)
    fresh = np.zeros(n)
    idx = np.flatnonzero(ctrl >= 0)
    runs = []
    if len(idx):
        start = prev = idx[0]
        for i in idx[1:]:
            if ctrl[i] != ctrl[prev] or i - prev > max_hole + 1:
                runs.append((start, prev + 1, int(ctrl[prev])))
                start = i
            prev = i
        runs.append((start, prev + 1, int(ctrl[prev])))
    runs = [r for r in runs if r[1] - r[0] >= min_run]

    # group consecutive touches by the same team, keep only groups that really won the ball
    groups = []
    for a, b, team in runs:
        if groups and groups[-1][2] == team:
            groups[-1][1] = b
            groups[-1][3] += b - a
            groups[-1][4] += 1
        else:
            groups.append([a, b, team, b - a, 1])
    won = [g for g in groups if g[3] >= min_keep or g[4] >= 2]
    # after dropping a lost touch, neighbours of the same team join up again
    runs = []
    for a, b, team, _, _ in won:
        if runs and runs[-1][2] == team:
            runs[-1] = (runs[-1][0], b, team)
        else:
            runs.append((a, b, team))

    # merge neighbouring runs of the same team, so the ownership spans the gap between them
    current = None
    for k, (a, b, team) in enumerate(runs):
        nxt = runs[k + 1][0] if k + 1 < len(runs) else n
        s = 1.0 if team == HOME else -1.0
        side[a:nxt] = s
        age = np.arange(nxt - b)
        fresh[a:b] = 1.0
        fresh[b:nxt] = np.exp(-age / fade)

    # a stoppage resets ownership: after it, the restart decides
    side[stopped] = 0.0
    fresh[stopped] = 0.0
    edges = np.flatnonzero(np.diff(stopped.astype(np.int8)) == -1) + 1   # first live frame
    for e in edges:
        nxt = np.flatnonzero(ctrl[e:] >= 0)
        stop = e + (nxt[0] if len(nxt) else n - e)
        side[e:stop] = 0.0
        fresh[e:stop] = 0.0
    return side, fresh




# ==================== decode ====================
"""Viterbi decoding: pick the smoothest H / A / D path that fits the evidence.

Each frame gives a score for each state. Changing state costs a fixed penalty. The decoder finds
the path with the best total score, so it only switches team when the evidence for the new team
is strong enough, for long enough, to pay the penalty. That gives clean possession chunks
instead of flicker, which is what spellF1 rewards.
"""

import numpy as np

H, A, D = 0, 1, 2


def viterbi(emis, switch_team, switch_dead):
    """emis: (T, 3) log scores per frame. Returns the best state per frame."""
    T = len(emis)
    trans = np.array([
        [0.0, -switch_team, -switch_dead],
        [-switch_team, 0.0, -switch_dead],
        [-switch_dead, -switch_dead, 0.0],
    ])
    back = np.zeros((T, 3), dtype=np.int8)
    score = emis[0].copy()
    for t in range(1, T):
        cand = score[:, None] + trans          # from state (row) to state (col)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + emis[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_duration(emis, switch_team, switch_dead, min_len):
    """Viterbi where each state must last at least min_len[state] frames.

    Each label gets a chain of sub states that count frames up to its minimum. The label can only
    be left from the last sub state, so nothing shorter than the minimum can appear.
    """
    T = len(emis)
    m = [max(1, int(k)) for k in min_len]
    cost = np.array([
        [0.0, switch_team, switch_dead],
        [switch_team, 0.0, switch_dead],
        [switch_dead, switch_dead, 0.0],
    ])
    neg = -1e18
    S = [np.full(m[l], neg) for l in range(3)]
    for l in range(3):
        S[l][-1] = emis[0, l]           # a half may start in any state
    back0 = np.zeros((T, 3), dtype=np.int8)       # which label entered sub state 0
    backlast = np.zeros((T, 3), dtype=bool)       # last sub state came from itself
    for t in range(1, T):
        ends = np.array([S[l][-1] for l in range(3)])
        new = []
        for l in range(3):
            enter = ends - cost[:, l]
            enter[l] = neg
            src = int(np.argmax(enter))
            back0[t, l] = src
            e = emis[t, l]
            s = np.empty(m[l])
            if m[l] == 1:
                stay = S[l][0]
                backlast[t, l] = stay >= enter[src]
                s[0] = max(stay, enter[src]) + e
            else:
                s[0] = enter[src] + e
                s[1:] = S[l][:-1] + e
                if S[l][-1] >= S[l][-2]:
                    s[-1] = S[l][-1] + e
                    backlast[t, l] = True
            new.append(s)
        S = new
    # backtrack
    l = int(np.argmax([S[k][-1] for k in range(3)]))
    i = m[l] - 1
    path = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = l
        if t == 0:
            break
        if m[l] == 1:
            if not backlast[t, l]:
                l = int(back0[t, l]); i = m[l] - 1
        elif i == 0:
            l = int(back0[t, l]); i = m[l] - 1
        elif i == m[l] - 1 and backlast[t, l]:
            pass
        else:
            i -= 1
    return path


def emissions(team_logodds, dead_logodds, scale):
    """Per frame log probabilities of H, A, D from two log odds, scaled to per second units."""
    log_sig = lambda z: -np.logaddexp(0.0, -z)   # log of the logistic function
    alive = log_sig(-dead_logodds)
    e = np.column_stack([
        alive + log_sig(team_logodds),
        alive + log_sig(-team_logodds),
        log_sig(dead_logodds),
    ])
    return e * scale



# ==================== decision ====================
import numpy as np

PARAMS = {
    # how much each piece of evidence counts (log odds units)
    "shape_clip": 3.0,       # team model, from player shape only
    "control": 3.0,          # last team in control (loose ball stays with it), fades after
    "control_fade": 2.0,     # seconds for that to fade so shape and tags take over
    "tag": 1.0,              # Smart Tagging team phase
    # restart taker weights: overridden by generic round numbers in possession_hybrid.py (the values here
    # came from a check on the competition match and are not used by the submission)
    "taker_visible": 2.9,    # ~95% right
    "taker_hidden": 1.4,     # ~80% right
    "dead_clip": 4.0,        # dead ball model
    "stoppage": 8.0,         # inside a stoppage we can see (ball out / placed), or just after
    "ball_missing": 0.5,     # tracker lost the ball (mild hint of a stoppage)
    # decoder: evidence is counted per second, a switch costs this many units
    "scale": 1.0 / FPS,
    "switch_team": 1.0,
    "switch_dead": 3.0,
    # shortest allowed run, in seconds: a possession needs real control; even the quickest
    # restart (a quick throw in) takes about 4 s
    "min_possession": 1.0,
    "min_dead": 4.0,
}


def logit(p, clip):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def decide(ev, P=PARAMS):
    n = ev["n"]
    # Direct ball evidence (who controlled it last) leads; the slow phase evidence (team shape,
    # tags) only fills in once that has faded, e.g. deep inside a ball gap.
    side, fresh = last_controller(ev["ctrl"], ev["stopped"], fade=P["control_fade"] * FPS)
    phase = logit(ev["p_home"], P["shape_clip"]) + P["tag"] * ev["tag_vote"]
    team = P["control"] * side * fresh + (1.0 - fresh) * phase
    team += P["taker_visible"] * ev["taker_visible"] + P["taker_hidden"] * ev["taker_hidden"]

    dead = logit(ev["p_dead"], P["dead_clip"])
    dead += P["ball_missing"] * (~ev["seen"])
    dead[ev["stopped"]] = P["stoppage"]
    dead[ev["restart"] & ~ev["stopped"]] = -P["stoppage"]
    dead[ev["before_kickoff"]] = 20.0     # certain

    team = np.nan_to_num(team)
    dead = np.nan_to_num(dead)
    labels = np.full(n, "X", dtype="<U1")
    for a, b in HALVES:
        e = emissions(team[a:b], dead[a:b], P["scale"])
        min_len = [P["min_possession"] * FPS] * 2 + [P["min_dead"] * FPS]
        path = viterbi_min_duration(e, P["switch_team"], P["switch_dead"], min_len)
        labels[a:b] = np.array(["H", "A", "D"])[path]
    return labels


def describe(labels):
    """Possession counts and durations, the numbers a coach would sanity check."""
    g = half_of(len(labels)) > 0
    runs = []
    for a, b in HALVES:
        seg = labels[a:b]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(seg)]
        runs += [(seg[s], (e - s) / FPS) for s, e in zip(starts, ends)]
    poss = [d for lab, d in runs if lab in "HA"]
    dead = [d for lab, d in runs if lab == "D"]
    share = {k: np.mean(labels[g] == k) for k in "HAD"}
    print("possessions {}  mean {:.1f} s  median {:.1f} s".format(
        len(poss), np.mean(poss), np.median(poss)))
    print("dead spells {}  mean {:.1f} s".format(len(dead), np.mean(dead) if dead else 0))
    print("share  H {H:.3f}  A {A:.3f}  D {D:.3f}".format(**share))


def write(path, labels):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

