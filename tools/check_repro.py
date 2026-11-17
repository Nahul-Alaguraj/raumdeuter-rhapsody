"""Reproducibility check: regenerate the submission twice and compare the two CSVs byte for byte.

    python tools/check_repro.py                        # run twice, compare
    python tools/check_repro.py --reference evaluation/results/f_ens_tcn_conv.csv
    python tools/check_repro.py --fresh                # also delete data/cache_*.npz first

--fresh forces the tracking to be re-read from the raw ATD, like a fresh clone would.
Prints SHA-256 of both outputs and PASS / FAIL. Needs the competition assets in data/.
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "results")


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def run(name):
    path = os.path.join(RESULTS, name + ".csv")
    if os.path.exists(path):
        os.remove(path)
    subprocess.run([sys.executable, os.path.join(ROOT, "possession_hybrid.py"), "--final", "--name", name],
                   cwd=ROOT, check=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", help="an existing CSV the output should be identical to")
    ap.add_argument("--fresh", action="store_true", help="delete data/cache_*.npz first")
    args = ap.parse_args()
    if args.fresh:
        for name in ("cache_tactical.npz", "cache_evidence.npz"):
            p = os.path.join(ROOT, "data", name)
            if os.path.exists(p):
                os.remove(p)
                print("removed", p)
    a, b = run("repro_run_a"), run("repro_run_b")
    ha, hb = sha(a), sha(b)
    print("\nrun A", ha)
    print("run B", hb)
    ok = ha == hb
    if args.reference:
        hr = sha(args.reference)
        print("ref  ", hr)
        ok = ok and ha == hr
    print("PASS: byte-identical" if ok else "FAIL: outputs differ")
    for p in (a, b):
        os.remove(p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
