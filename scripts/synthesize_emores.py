# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Run CoCoEmo's synthesis path with EmoRES composition substituted for Eq. 7.

Usage:
    python scripts/synthesize_emores.py --lambda-c 1 --lambda-r 3 \
        --backbone cosyvoice2 --model-dir <dir> \
        --steering-dir <CoCoEmo>/steering_vectors/<backbone> \
        --manifest iemocap_mixed.csv --alpha <A> --output-dir out/res

Adapted from CoCoEmo (MIT): ``scripts/synthesize.py``,
``cocoemo/steering/_core_cosyvoice.py:782`` and
``cocoemo/steering/_core_indextts.py:1034``.
"""

import argparse
import sys

DEFAULT_ALPHA = 5.0


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--lambda-r", type=float, default=3.0)
    ap.add_argument(
        "--raw",
        action="store_true",
        help="skip the rbar rescaling; injected norm then grows with lambda_r",
    )
    ap.add_argument(
        "--emo-vector",
        dest="emo_vector",
        action="store_true",
        default=True,
        help="IndexTTS-2: stack steering on the native emo_vector conditioning "
        "(CoCoEmo App. E.1). "
        "ON by default -- this is the paper's configuration, not their script's",
    )
    ap.add_argument(
        "--no-emo-vector",
        dest="emo_vector",
        action="store_false",
        help="steering alone, which is what CoCoEmo's synthesize.py does",
    )
    ap.add_argument(
        "--nosteer",
        action="store_true",
        help="the unsteered baseline: no injection at all. rho and H-Rate are "
        "defined as the "
        "increase over this arm, so they cannot be computed without it",
    )
    ap.add_argument(
        "--instruct",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="CosyVoice2: 0 off, 1 descriptive instruct2 prompt (default, the "
        "paper's Ins1), "
        "2 percentage-based",
    )
    ap.add_argument(
        "--emo-scale",
        type=float,
        default=0.6,
        help="scale for --emo-vector (CoCoEmo App. E.1 uses 0.6)",
    )
    ours, rest = ap.parse_known_args()

    helping = any(h in rest for h in ("-h", "--help"))
    if helping:
        print(__doc__.split("Why a patch")[0].rstrip())
        print("EmoRES options (everything else is CoCoEmo's, listed below):")
        print("  --lambda-c FLOAT   shared-axis weight (default 1.0)")
        print("  --lambda-r FLOAT   residual weight (default 3.0)")
        print(
            "  --raw              skip the rbar rescaling\n"
            f"  --alpha FLOAT      injected norm in rbar units (default {DEFAULT_ALPHA}"
            f")\n"
        )
    elif not any(x == "--alpha" or x.startswith("--alpha=") for x in rest):
        rest += ["--alpha", str(DEFAULT_ALPHA)]
        print(
            f"[res] --alpha not given, using {DEFAULT_ALPHA}"
            f" (injected norm = alpha * rbar). Comparable "
            f"across the knobs within a backbone, but not across backbones or corpora.",
            flush=True,
        )

    import cocoemo.steering._core_cosyvoice as core

    from emores.compose import compose, load_library

    try:
        import scripts.synthesize as syn
    except ImportError as e:
        raise SystemExit(
            f"cannot import CoCoEmo's scripts/synthesize.py ({e}).\n"
            f"Put the CoCoEmo checkout on PYTHONPATH, e.g.\n"
            f"PYTHONPATH=/path/to/EmoRES:/path/to/CoCoEmo python "
            f"scripts/synthesize_emores.py ..."
        ) from e

    cache = {}
    cur = {}
    skipped = []

    def res_compose(steering_files_dict, emotion_percentages):
        cur["p"] = emotion_percentages
        emotions = [e for e in syn.EMOTIONS if e in steering_files_dict]
        key = tuple(sorted(steering_files_dict.items()))
        if key not in cache:
            cache[key] = load_library(steering_files_dict, emotions)
        p = [float(emotion_percentages.get(f"p_{e}", 0.0)) for e in emotions]
        if sum(p) <= 0:
            skipped.append(1)
            lib = cache[key]
            return {
                op: {L: V[0] * 0.0 for L, V in layers.items()}
                for op, layers in lib.items()
            }
        return compose(
            cache[key], p, ours.lambda_c, ours.lambda_r, normalise=not ours.raw
        )

    core.create_sample_specific_mixed_vectors = res_compose

    if ours.emo_vector:
        import cocoemo.backbones.indextts2 as it2

        SLOTS = [
            "happy",
            "angry",
            "sad",
            "afraid",
            "disgusted",
            "melancholic",
            "surprised",
            "calm",
        ]
        TO_SLOT = {
            "happy": "happy",
            "angry": "angry",
            "sad": "sad",
            "surprise": "surprised",
        }

        _gen = it2.generate_steered_speech

        def gen_with_emov(*a, **kw):
            p = cur.get("p") or {}
            w = {e: float(p.get(f"p_{e}", 0) or 0) for e in TO_SLOT}
            tot = sum(w.values())
            if tot > 0:
                v = dict.fromkeys(SLOTS, 0.0)
                for e, sl in TO_SLOT.items():
                    v[sl] = ours.emo_scale * w[e] / tot
                kw["emo_vector"] = [round(v[s], 6) for s in SLOTS]
                kw["emo_alpha"] = 1.0
            return _gen(*a, **kw)

        it2.generate_steered_speech = gen_with_emov
        print(
            f"[res] native emo_vector ON, scale {ours.emo_scale} (IndexTTS-2 only)",
            flush=True,
        )

    if "cosyvoice2" in rest:
        import cocoemo.backbones.cosyvoice2 as cv2bb

        from emores.cosy_compat import patch_zero_shot_kwarg

        _orig_load = cv2bb.load_model

        def _load_and_patch(*a, **kw):
            m = _orig_load(*a, **kw)
            patch_zero_shot_kwarg(m)
            return m

        cv2bb.load_model = _load_and_patch

        if ours.instruct:
            PREFIX, EOP = "You are a helpful assistant. ", "<|endofprompt|>"

            def _oxford(x):
                if len(x) == 1:
                    return x[0]
                if len(x) == 2:
                    return f"{x[0]} and {x[1]}"
                return ", ".join(x[:-1]) + f", and {x[-1]}"

            def _ins(p):
                live = sorted(
                    ((e, float(p.get(f"p_{e}", 0) or 0)) for e in syn.EMOTIONS),
                    key=lambda kv: -kv[1],
                )
                live = [(e, w) for e, w in live if w > 0]
                if not live:
                    return ""
                names = [e for e, _ in live]
                if ours.instruct == 1:
                    body = (
                        f"Say it in a {names[0]} tone."
                        if len(names) == 1
                        else f"Say it in a blend of {_oxford(names)} emotions."
                    )
                else:
                    tot = sum(w for _, w in live)
                    body = (
                        "Say it in "
                        + _oxford([f"{round(100 * w / tot)}% {e}" for e, w in live])
                        + "."
                    )
                return f"{PREFIX}{body}{EOP}"

            _cvgen = cv2bb.generate_steered_speech

            def gen_with_instruct(*a, **kw):
                t = _ins(cur.get("p") or {})
                if t:
                    kw["prompt_text"] = t
                    kw["inference_type"] = "instruct2"
                return _cvgen(*a, **kw)

            cv2bb.generate_steered_speech = gen_with_instruct
            print(f"[res] CosyVoice2 instruct2 Ins{ours.instruct} ON", flush=True)

    if ours.nosteer and not helping:
        import cocoemo.steering._core_indextts as _core_it2

        for _m in (core, _core_it2):
            _m.prepare_steering_injection_config = lambda *a, **k: {}
        print(
            "[res] NOSTEER: no hooks registered, steering vectors loaded but unused",
            flush=True,
        )
    if not helping and not ours.nosteer:
        mode = "raw" if ours.raw else "normalised to rbar"
        print(
            f"[res] composition: v = {ours.lambda_c} * c "
            f"+ {ours.lambda_r} * R_p, {mode}",
            flush=True,
        )
        if ours.lambda_c == 1.0 and ours.lambda_r == 1.0 and ours.raw:
            print(
                "[res] note: lambda_c = lambda_r with --raw is CoCoEmo's Eq. 7 exactly",
                flush=True,
            )

    sys.argv = [sys.argv[0]] + rest
    syn.main()
    if skipped:
        print(
            f"[res] {len(skipped)}"
            f" clips had no mass on any steerable emotion and were left unsteered "
            f"(every metric excludes them)",
            flush=True,
        )


if __name__ == "__main__":
    main()
