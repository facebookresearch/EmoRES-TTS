# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Aggregate rho and H-Rate from a scored results file.

Usage:
    PYTHONPATH=/path/to/EmoRES:/path/to/CoCoEmo \
    python scripts/mixed_metrics_emores.py \
        --results results/res/per_sample_results.json \
        --baseline results/nosteer/per_sample_results.json

Both formulas are imported from CoCoEmo (MIT):
``cocoemo/evaluation/mixed_metrics.py:81,128``.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        required=True,
        help="per_sample_results.json for the steered arm",
    )
    ap.add_argument(
        "--baseline",
        required=True,
        help="per_sample_results.json for the no-steer arm",
    )
    a = ap.parse_args()

    from cocoemo.evaluation.mixed_metrics import (
        compute_dominant_hit_rate,
        compute_rank_correlation,
        load_per_sample_results,
    )

    base = {
        s["sentence_id"]: s
        for s in load_per_sample_results(a.baseline)
        if s.get("sentence_id")
    }
    if not base:
        raise SystemExit(f"[FATAL] no sentence_id records in {a.baseline}")

    rho = compute_rank_correlation(a.results, base)
    hit = compute_dominant_hit_rate(a.results, base)
    n = len(load_per_sample_results(a.results))

    def show(name, v):
        if isinstance(v, (tuple, list)):
            print(f"  {name:<8} {float(v[0]) * 100:7.2f}   (n={v[1]})")
        elif v is None:
            print(f"  {name:<8}      --   (undefined: no qualifying clips)")
        else:
            print(f"  {name:<8} {float(v) * 100:7.2f}")

    print(f"=== {a.results}  vs baseline {a.baseline}   ({n} scored clips)")
    show("rho", rho)
    show("H-Rate", hit)


if __name__ == "__main__":
    main()
