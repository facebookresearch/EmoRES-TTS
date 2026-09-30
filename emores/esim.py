# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Anchored E-SIM: does the generated speech arrive at the target emotion?

Two different metrics share the name E-SIM, and they ask opposite questions:

    CoCoEmo's  E-SIM = cos(gen, the source recording)        -> did you preserve
               the source's emotion?
    anchored   E-SIM = mean cos(gen, anchors of target e)    -> did you arrive
               at the target emotion?

On an emotion-conversion task only the second is aligned with the objective, and
the two can rank arms in opposite orders -- an unsteered clone scores well on
the first precisely because it changed nothing. Both are legitimate; they are
simply not interchangeable, so the name alone is never enough to say which was
reported.

attribution. The anchored formulation is EmoSteer-TTS's
(arxiv.org/abs/2508.03543, Sec. 4.2: "100 anchor emotional samples per emotion",
drawn from MSP-Podcast and ESD). The emotion2vec embedding
extraction is CoCoEmo's ``extract_emotion2vec_feats``, imported rather than
reimplemented so that this column and their E-SIM column are computed by the
same code. Ours are the curated anchor bank (``data/esd_anchors.txt``), the
baseline subtraction, the p-weighted form, and the label-rotation falsifier.


Four numbers, and why each is there:

  raw       mean cos(gen, anchors of the dominant target). EmoSteer's published
            form; kept so the comparison
            with their numbers is like-for-like.
  weighted  sum_e p_e * cos(gen, anchors of e). On a mixture corpus the
            dominant-target form discards every
            non-dominant component of the very label the arm was steered toward,
            so this is the form that
            matches the steering objective.
  sep       raw minus the mean cosine to the other emotions' anchors. Raw anchor
            cosine has no zero: two
            arbitrary speech clips already sit strongly positive in emotion2vec
            space, so an arm can read 0.6
            while carrying no target emotion at all. Subtracting removes
            whatever is common to all speech.
  top1      fraction of clips whose best-scoring anchor emotion IS the target,
            chance 1/len(EMOS). A mean
            cosine can rise while the argmax never moves, and that distinction
            is not visible in the mean.

The anchored formulation follows EmoSteer-TTS (arXiv:2508.03543). It is not
CoCoEmo's E-SIM, which is ``cosine_similarity_wavs`` in
``cocoemo/evaluation/metrics.py:77`` and compares against the source recording.
"""

from __future__ import annotations

import os

import numpy as np

EMOS = ("angry", "happy", "sad", "surprise")


def load_anchor_bank(path, esd_root, rotate=0):
    """Anchor wav paths per emotion, resolved against a local ESD checkout."""
    src = {e: [] for e in EMOS}
    with open(path) as fh:
        lines = fh.read().splitlines()
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) != 3:
            raise ValueError(f"expected <speaker>/<Emotion>/<clip>.wav, got {line!r}")
        emo = parts[1].lower()
        if emo not in src:
            raise ValueError(f"{line!r} has emotion {emo!r}, not one of {EMOS}")
        src[emo].append(line)

    out = {}
    for i, e in enumerate(EMOS):
        held = EMOS[(i + rotate) % len(EMOS)]
        anchors = src[held]
        if len(anchors) < 10:
            raise ValueError(
                f"{held}: {len(anchors)} anchors is too few to average over"
            )
        out[e] = [os.path.join(esd_root, w) for w in anchors]
    missing = [p for ps in out.values() for p in ps[:1] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"anchor wav not found: {missing[0]}\n"
            f"esd_root={esd_root!r}  should be the directory containing ESD's "
            f"per-speaker folders (0011/, ...)"
        )
    return out


def embed_wavs(emotion_model, paths, cache=None):
    """Unit-norm emotion2vec utterance embeddings, (n, d), in the order given."""
    from cocoemo.evaluation.metrics import extract_emotion2vec_feats

    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        have = {str(k): v for k, v in zip(z["paths"], z["vecs"])}
        if all(p in have for p in paths):
            return np.stack([have[p] for p in paths])

    V = np.stack(
        [
            np.asarray(extract_emotion2vec_feats(emotion_model, p), dtype=np.float64)
            for p in paths
        ]
    )
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, paths=np.array(paths, dtype=object), vecs=V)
    return V


def embed_bank(emotion_model, bank, cache_dir=None):
    """{emotion: (k, d)} unit embeddings for a bank from `load_anchor_bank`."""
    return {
        e: embed_wavs(
            emotion_model,
            ps,
            cache=os.path.join(cache_dir, f"anchors_{e}.npz") if cache_dir else None,
        )
        for e, ps in bank.items()
    }


def esim(G, anchors, targets, p=None):
    """Anchored E-SIM for generated embeddings G (n, d) against an embedded bank.

    targets  length-n dominant target emotion per clip.
    p        optional (n, len(EMOS)) mixture weights, rows summing to 1, column
             order EMOS. Required for the weighted column;
             without it `weighted` is None.

    Returns per-clip arrays plus their means, so a caller can re-cut by subset
    without a second GPU pass.
    """
    if len(G) != len(targets):
        raise ValueError(f"{len(G)} embeddings but {len(targets)} targets")

    S = np.stack([(G @ anchors[e].T).mean(1) for e in EMOS], axis=1)  # (n, len(EMOS))
    ti = np.array([EMOS.index(t) for t in targets])
    rows = np.arange(len(G))

    raw = S[rows, ti]
    other = (S.sum(1) - raw) / (len(EMOS) - 1)
    top1 = S.argmax(1) == ti

    weighted = None
    if p is not None:
        p = np.asarray(p, dtype=float)
        if p.shape != S.shape:
            raise ValueError(f"p must be {S.shape}, got {p.shape}")
        if not np.allclose(p.sum(1), 1.0, atol=1e-6):
            raise ValueError("each row of p must sum to 1 over EMOS")
        weighted = (p * S).sum(1)

    return {
        "raw": raw,
        "sep": raw - other,
        "top1": top1,
        "weighted": weighted,
        "scores": S,
        "emos": EMOS,
        "mean_raw": float(raw.mean()),
        "mean_sep": float((raw - other).mean()),
        "mean_top1": float(top1.mean()),
        "mean_weighted": None if weighted is None else float(weighted.mean()),
        "chance_top1": 1.0 / len(EMOS),
    }
