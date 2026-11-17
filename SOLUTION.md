# Our solution: possession from tracking, trained only on external data

Team: `raumdeuter_rhapsody`. The submitted file is `raumdeuter_rhapsody.csv`, produced by the one command below.

## Regenerate the submission

1. Put the competition assets in `data/` as described in `README.md` / `DATA.md`.
2. `pip install -r requirements-solution.txt` (CPU only; Python and package versions are recorded in that file).
3. Run:

```
python possession_hybrid.py --final --name raumdeuter_rhapsody
python evaluation/validate_submission.py evaluation/results/raumdeuter_rhapsody.csv
```

`python tools/check_repro.py` runs step 3 twice and checks that the two CSVs are byte-identical
(add `--fresh` to also rebuild the tracking cache, as in a fresh clone).

## What the pipeline does

1. **Rule layer** (`possession_rules.py`): fixed football logic written by hand, with no fitting to the
   competition match. It reads the ATD tracking and Smart Tagging and finds stoppages (ball out, ball placed
   still, until a clear kick), restart takers, and which team last clearly controlled the ball.
2. **Neural layer** (`training/`): an ensemble of a TCN and a ConvGRU (3 seeds each, weights committed in
   `training/`) gives, for every moment, P(home has the ball) and P(dead ball). Inputs are the 12 ball
   features plus 23 ball-free team-shape features (block position, width and depth, speeds, forward
   movement), so the model still works in long ball outages. These networks are trained **only on external data**.
3. **Decoder**: the evidence is combined as log-odds and decoded with a Viterbi pass with switch penalties and
   minimum durations (possession at least 1 s, dead ball at least 4 s). The constants are generic round numbers.

Output labels are `H`, `A`, `D` for every frame; abstentions (`X`) are only written where the tracking has no data.

## External data (training and tuning)

| Data | Use | Source / licence |
|---|---|---|
| IDSSE / Sportec open Bundesliga tracking, 7 matches | training; labels from the tracking's own ball-state and owner fields | DFL / Sportec Solutions, CC BY 4.0, via kloppy |
| SoccerTrack v2, 10 matches | training (player positions only; ball treated as always missing); labels derived from its 12-class event annotations | `atomscott/soccertrack-v2` on Hugging Face, CC BY 4.0 |
| Metrica Sports public sample games 1 and 2 | evaluation only, never trained on; labels derived from the event files | github.com/metrica-sports/sample-data (source acknowledged) |

No data from the competition match, and no hand labels, were used for training or tuning. The labelling code
is in `training/build_idsse.py`, `training/build_metrica.py` and the SoccerTrack cell of the notebook.

## Training

`possession_final_training_kaggle.ipynb` rebuilds all external data from the public sources and trains the
networks end to end (5 architectures, 3 seeds each; TCN and ConvGRU are the ones submitted).
Training uses simulated ball outages (a mix of short flickers and 4 to 60 s gaps hiding 30 to 50% of the ball)
and random position jitter, with generic ranges. GPU training is not bit-exactly repeatable, so the committed
weights are the reference; the submission CSV is regenerated from them deterministically.

## How choices were made

Architectures and the ensemble were chosen on held-out external matches (IDSSE, SoccerTrack) and on the
Metrica sample games, **before** looking at public-window scores. Switch penalties for the GRU decoding were
chosen on a held-out IDSSE match with the official metric.

## Rule compliance: what touches the competition data

The competition files (ATD tracking, Smart Tagging) are read in exactly two places, both as **input** for
producing labels: `possession_rules.load_tracking` / `load_tags` and `possession_hybrid.read_atd`.
No code in this repository fits, trains, tunes or scores anything against the competition match or its
public ground truth: there are no fitting calls on it, no label files are read, and all training and
evaluation code (`training/`) reads only the external datasets listed above. Scoring against the public
window was done separately with the organisers' `evaluation/run_evaluation.py`.

## Disclosures

* The 5-minute public ground truth was viewed while debugging and to compare candidate models. It was not
  used to tune any parameter or to choose between models.
* The outage simulation ranges were designed after inspecting generic statistics of the provided tracking
  (how often and how long the ball is missing); they are broad ranges, not fitted values.
* An earlier experiment that fitted models to the competition match itself (self-supervised) was dropped
  as not allowed; it is not part of this repository or of the submission.
* Metrica's ball is missing mostly during dead ball, unlike the competition tracking, so its absolute scores
  are optimistic; we used it only to compare models with each other.

## Known limitations

Possessions are slightly over-merged (a few fewer, longer spells than the public ground truth) and about a
third of transitions land more than 1 s from the true moment. Dead-ball recall is about 0.8 on the public window.
