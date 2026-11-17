"""Turn the model's per-step probabilities into a clean H/A/D sequence.

A Viterbi pass with switch penalties: changing team (H<->A) costs `lam`, entering or leaving
dead ball costs `mu`. Bigger penalties give fewer, longer possessions. The penalties are chosen on
EXTERNAL held-out matches only (see train_v3.py), never on the competition match.
"""
import numpy as np
import torch

from train_possession import WIN


@torch.no_grad()
def half_logprobs(model, x):
    """(n, 3) log-probabilities from overlapping windows, averaged so edges don't decide a step."""
    model.eval()
    acc = np.zeros((len(x), 3), dtype=np.float32)
    cnt = np.zeros(len(x), dtype=np.float32)
    starts = list(range(0, max(1, len(x) - WIN + 1), WIN // 2))
    if starts[-1] + WIN < len(x):
        starts.append(len(x) - WIN)
    for s in starts:
        out = torch.log_softmax(model(torch.from_numpy(x[s:s + WIN])[None])[0], -1).numpy()
        acc[s:s + WIN] += out
        cnt[s:s + WIN] += 1
    out = acc / np.maximum(cnt, 1)[:, None]
    if np.isnan(out).any():
        raise ValueError("NaN in model output: {} values. Check the input features.".format(int(np.isnan(out).sum())))
    return out


def viterbi(logp, lam, mu):
    """Best state path (0=H, 1=A, 2=D) under switch penalties lam (H<->A) and mu (to/from D)."""
    n = len(logp)
    cost = np.array([[0, lam, mu],
                     [lam, 0, mu],
                     [mu, mu, 0]], dtype=np.float32)
    score = logp[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost                     # cand[i, j]: come from i, go to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
