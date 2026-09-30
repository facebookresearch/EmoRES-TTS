# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Score a directory of generated wavs with anchored E-SIM.

Usage:
    PYTHONPATH=/path/to/EmoRES:/path/to/CoCoEmo \
    python scripts/score_anchored_esim.py \
        --manifest iemocap_mixed.csv --wav-dir out/res \
        --esd-root "<ESD>/Emotion Speech Dataset" \
        --cache-dir .cache --out res_esim.json

The emotion2vec loader is CoCoEmo's (MIT) ``cocoemo/evaluation/metrics.py:321``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from emores.esim import embed_bank, embed_wavs, EMOS, esim, load_anchor_bank


def read_targets(manifest):
    """[(wav_filename, p over EMOS summing to 1, dominant target)], plus diagnostics."""
    keep, dropped_mass, no_target = [], [], 0
    with open(manifest) as fh:
        rows_in = list(csv.DictReader(fh))
    for r in rows_in:
        raw = np.array([float(r.get(f"p_{e}", 0) or 0) for e in EMOS])
        total = raw.sum()
        if total <= 0:
            no_target += 1
            continue
        dropped_mass.append(float(r.get("p_neutral", 0) or 0))
        p = raw / total
        keep.append((r["wav_filename"], p, EMOS[int(p.argmax())]))
    return keep, {
        "no_target": no_target,
        "mean_neutral_mass_dropped": float(np.mean(dropped_mass))
        if dropped_mass
        else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--wav-dir",
        required=True,
        help="directory of generated wavs named <wav_filename>.wav",
    )
    # Resolved against the repo, not the cwd: this script is run from job
    # directories and a bare relative default fails there with a
    # FileNotFoundError after the manifest has already been parsed.
    ap.add_argument(
        "--anchors",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "esd_anchors.txt",
        ),
    )
    ap.add_argument(
        "--esd-root",
        required=True,
        help="the dir holding ESD's speaker folders (0011/, ...)",
    )
    ap.add_argument(
        "--emotion-model",
        default="iic/emotion2vec_plus_large",
        help=("local emotion2vec directory, or a hub id if the machine has network"),
    )
    ap.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="1..3 relabels the bank; the falsifier control",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="reuse embeddings across arms; strongly recommended",
    )
    ap.add_argument("--out", default=None, help="write per-clip results here as JSON")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    rows, diag = read_targets(a.manifest)
    if not rows:
        raise SystemExit(f"[FATAL] no clip in {a.manifest} has any mass on {EMOS}")

    wavs, missing = [], []
    for sid, _, _ in rows:
        hit = glob.glob(os.path.join(a.wav_dir, f"{sid}.wav"))
        (wavs if hit else missing).append(hit[0] if hit else sid)
    if missing:
        # An arm that silently scores fewer clips than another is not comparable
        # to it, and the difference is invisible in a mean. Refuse rather than
        # quietly score the intersection.
        raise SystemExit(
            f"[FATAL] {len(missing)} of {len(rows)} wavs missing from {a.wav_dir}, "
            f"first: {missing[0]}.wav"
        )

    import torch
    from funasr import AutoModel

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Same construction as CoCoEmo's load_models, minus Whisper and WavLM which
    # this column does not use.
    model = AutoModel(device=device, model=a.emotion_model, disable_update=True)

    tag = f"rot{a.rotate}" if a.rotate else "curated"
    bank = load_anchor_bank(a.anchors, a.esd_root, rotate=a.rotate)
    A = embed_bank(
        model,
        bank,
        cache_dir=os.path.join(a.cache_dir, tag) if a.cache_dir else None,
    )
    # The generated wavs are cached under the ARM directory's name, so two arms
    # never collide.
    gcache = (
        os.path.join(a.cache_dir, f"gen_{os.path.basename(a.wav_dir.rstrip('/'))}.npz")
        if a.cache_dir
        else None
    )
    G = embed_wavs(model, wavs, cache=gcache)

    P = np.stack([p for _, p, _ in rows])
    res = esim(G, A, [t for _, _, t in rows], P)

    print(
        f"\n=== anchored E-SIM | {os.path.basename(a.wav_dir)} | bank={tag} | n="
        f"{len(rows)} ==="
    )
    if a.rotate:
        print("  FALSIFIER RUN: labels rotated. Separation here is noise, not signal.")
    print(f"  E-SIM (weighted) {res['mean_weighted']:.4f}")
    print(f"  E-SIM (raw)      {res['mean_raw']:.4f}")
    print(f"  SEP              {res['mean_sep']:.4f}")
    print(
        f"  top-1            {res['mean_top1']:.3f}   (chance {res['chance_top1']:.3f})"
    )
    print(f"  excluded, no mass on {EMOS}: {diag['no_target']}")
    print(
        f"  mean neutral mass dropped before renormalising: "
        f"{diag['mean_neutral_mass_dropped']:.3f}"
    )

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(
                {
                    "wav_dir": a.wav_dir,
                    "manifest": a.manifest,
                    "bank": tag,
                    "n": len(rows),
                    "emos": list(EMOS),
                    "diagnostics": diag,
                    "mean": {
                        k: res[k]
                        for k in (
                            "mean_weighted",
                            "mean_raw",
                            "mean_sep",
                            "mean_top1",
                            "chance_top1",
                        )
                    },
                    "per_clip": [
                        {
                            "wav_filename": sid,
                            "weighted": float(w),
                            "raw": float(r),
                            "sep": float(s),
                            "top1": bool(t),
                            "target": tg,
                        }
                        for (sid, _, tg), w, r, s, t in zip(
                            rows,
                            res["weighted"],
                            res["raw"],
                            res["sep"],
                            res["top1"],
                        )
                    ],
                },
                fh,
                indent=1,
            )
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
