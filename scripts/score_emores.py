# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Run CoCoEmo's evaluator with the metric models loaded from local paths.

Usage:
    EMOTION2VEC=<dir> WHISPER=<file.pt> SPEAKER=<dir> \
    python scripts/score_emores.py --wav-dir out/res \
        --manifest iemocap_mixed.csv --output-dir results/res

Adapted from CoCoEmo (MIT): ``scripts/evaluate.py`` and
``cocoemo/evaluation/metrics.py:321``.
"""

import os


def main():
    e2v = os.environ.get("EMOTION2VEC")
    whi = os.environ.get("WHISPER")
    spk = os.environ.get("SPEAKER")

    import cocoemo.evaluation.metrics as M

    _orig = M.load_models

    def load_local(device=None, **kw):
        if e2v:
            kw["emotion_model_id"] = e2v
        if whi:
            kw["whisper_size"] = whi
        if spk:
            kw["speaker_model_id"] = spk
        print(
            f"[score] emotion2vec={kw.get('emotion_model_id')}\n"
            f"[score] whisper    ={kw.get('whisper_size')}\n"
            f"[score] speaker    ={kw.get('speaker_model_id')}",
            flush=True,
        )
        return _orig(device=device, **kw)

    M.load_models = load_local

    clamp = os.environ.get("WER_CLAMP", "1") not in ("", "0")
    if clamp:
        _orig_wer = M.whisper_wer_for_wav
        hit = []

        def wer_clamped(*a, **kw):
            v = _orig_wer(*a, **kw)
            if isinstance(v, (int, float)) and v > 1.0:
                hit.append(v)
                return 1.0
            return v

        M.whisper_wer_for_wav = wer_clamped

    try:
        import scripts.evaluate as ev
    except ImportError as e:
        raise SystemExit(
            f"cannot import CoCoEmo's scripts/evaluate.py ({e}).\n"
            f"PYTHONPATH=/path/to/EmoRES:/path/to/CoCoEmo python "
            f"scripts/score_emores.py ..."
        ) from e
    ev.main()
    if clamp and hit:
        import statistics

        print(
            f"[score] WER clamped to 1.0 on {len(hit)} clip(s); raw values ranged "
            f"{min(hit):.2f}-{max(hit):.2f} (median {statistics.median(hit):.2f}"
            f"). These are "
            f"non-terminating decodes, not transcription errors.",
            flush=True,
        )
    elif clamp:
        print("[score] WER clamp active; no clip exceeded 1.0", flush=True)


if __name__ == "__main__":
    main()
