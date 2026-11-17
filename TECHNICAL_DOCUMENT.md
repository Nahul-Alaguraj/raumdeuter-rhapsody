# Ball possession from tracking — team raumdeuter_rhapsody

More than a Hack 2027, Phase 1. Submission: `raumdeuter_rhapsody.csv`. Code: the public repository linked in the form (file `SOLUTION.md` is the entry point).

## 1. Approach in one paragraph

Every frame gets H, A or D from an algorithm; nothing is hand labelled and nothing is learned from the competition match. The pipeline has three parts. A **rule layer** of fixed football logic reads the automatic tracking (ATD) and Smart Tagging and spots stoppages, restart takers and which team last clearly controlled the ball. A **neural layer**, an ensemble of a temporal convolutional network (TCN) and a convolutional GRU (3 seeds each), outputs for every moment P(home has the ball) and P(dead ball); it was trained **only on external matches**. A **decoder** combines both as log-odds and picks the most plausible H/A/D path with a Viterbi pass that has switch penalties and minimum durations. We chose this hybrid because the rules are reliable where the ball is visible, while a model trained on other matches carries the decision through the long stretches (median real gap about 10 to 12 s, about 41% of graded frames) where the ball is missing.

## 2. Assets used

We used the automatic tactical tracking (ATD: 46 anonymous player slots and the ball, normalised coordinates, team of each slot from its metadata) and the Smart Tagging file (team phase tags and kick-off tags). Frame indices are aligned to video frames (ATD frame_id minus 1). We did not use the video, the broadcast-camera tracking or the homography. Early on we also tried a self-supervised variant that fitted models to the competition match's own tracking; we dropped it because it learns from the provided game, and it is not in the repository or the submission.

## 3. Training data (all external; none from the provided game)

| Dataset | Size | Source | Who labelled, and how |
|---|---|---|---|
| IDSSE / Sportec open Bundesliga tracking | 7 matches, about 1M frames, ball on every frame | DFL / Sportec Solutions, CC BY 4.0, loaded with kloppy | Ball state (alive/dead) and owning team are fields of the data provider's own tracking, produced by its data operations; we map them to H / A / D |
| SoccerTrack v2 | 10 university matches (20 halves), positions only; the ball is always treated as missing | `atomscott/soccertrack-v2` on Hugging Face, CC BY 4.0 | The dataset authors' 12-class event annotations. We derive a possession label by rule: the team of the last touch holds the ball until the next event if the gap is at most 10 s; ball out or goal means dead until the restart (at most 90 s); other gaps stay unknown and are ignored |
| Metrica Sports public sample games 1 and 2 | 2 matches | github.com/metrica-sports/sample-data (source acknowledged) | Same rule applied to its event files. Used **only for evaluation**, never for training |

Training uses simulated ball outages (a mix of short flickers and 4 to 60 s gaps hiding 30 to 50% of the ball) and random position jitter of 0.1 to 0.5 m. These are broad generic ranges, not fitted values; they were designed after looking at general statistics of the provided tracking (how long and how often the ball goes missing), which we disclose. Inputs per 0.2 s step: 12 ball-based features (distance to nearest player of each team, players near the ball, ball speed, how long since the ball was last seen) and 23 ball-free team-shape features (block position, width, depth, speed, forward movement of each team, with the attacking direction found from the tracking itself). The ball-free features are what keep the model useful during outages.

## 4. The possession rules we implemented

* **Controlled ball.** A team controls the ball when one of its players is within 1 m of it and no opponent is within 2 m. A contested ball (a challenge) gives no control information.
* **Loose balls.** The team that last clearly controlled the ball keeps it while the ball is loose or in the air; this evidence fades over 2 s, after which the neural layer's team estimate and the Smart Tagging phase (attacking vs defending phases of each team, weight 1) take over.
* **Stoppages and restarts.** The ball is dead when it is out of the pitch (at least 8 frames), or lies still for 2 s (placed for a free kick, goal kick, corner, kick-off), until a clear kick or throw (faster than 4 m/s for 3 frames). A safety brake ends a stoppage if players move at live-play pace. The team of the player taking the restart gets possession just after it. Kick-offs come from the Smart Tagging kick-off tags.
* **Minimum durations.** A possession lasts at least 1 s and a dead-ball spell at least 4 s (even the quickest throw-in takes about that long), so flicker cannot split a real possession.
* **Decoder constants** (evidence weights, switch penalties 1.0 between teams and 3.0 to or from dead ball, and the durations above) are generic round numbers. The two weights for restart takers (2.0 and 1.0) are deliberately round numbers.

## 5. Results and how choices were made

Architectures and the ensemble were chosen on held-out **external** matches before looking at the public window. On a held-out IDSSE match the new ball-free features raised spell F1 from 0.49 (ball features only) to 0.57 (GRU) and 0.64 (TCN). On the Metrica sample games, single models scored 0.65 to 0.71 spell F1 and ensembles 0.72, so we submit the TCN plus ConvGRU ensemble (6 networks), the simplest set tied for best. (Metrica's ball is missing mostly during dead ball, unlike our tracking, so we use those numbers only to compare models, not as expected performance.)

On the 5-minute public window (17 possessions, so noisy, roughly plus or minus 0.1) the submitted file scores **spell F1 0.63, macro F1 0.88**, dead-ball F1 0.89 (precision 0.999, recall 0.80), possession share error 0.01. It finds 14 possessions against 17 true ones. We viewed this window while debugging and when comparing candidates, but did not tune any parameter on it or use it to choose between models.

## 6. Limitations

Possessions are slightly over-merged (about 14 found against 17, mean 18 s against 14 s), and only about 40% of transitions land within 1 s of the true moment; boundary timing is the main thing that caps the score at strict overlap thresholds. About a fifth of dead-ball frames are labelled as possession. Long outages after a turnover are the hardest case. With more time we would tune the decoder penalties on a larger external set with a realistic ball-outage regime, add more external matches with a missing-ball pattern like ours, and improve restart-taker detection when the ball is hidden. Our rule layer also contains thresholds (distances, speeds) written from football reasoning while looking at this tracking format; none were fitted to labels, but they were not tuned on external data either.

## 7. Reproduction

Needs the competition data bundle in `data/` and `pip install -r requirements-solution.txt` (numpy, pandas, torch; exact versions pinned; CPU only). Then:

```
python possession_hybrid.py --final --name raumdeuter_rhapsody
python evaluation/validate_submission.py evaluation/results/raumdeuter_rhapsody.csv
```

This regenerates the submitted file byte for byte (`python tools/check_repro.py` runs it twice and compares). The network weights are committed in `training/`. The notebook `possession_final_training_kaggle.ipynb` rebuilds all external data from the public sources and retrains the models; GPU training is not bit-exactly repeatable, so the committed weights are the reference.
