# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Build the CREMA-D mixed-emotion evaluation set.

Usage:
    python scripts/build_cremad_mixed.py --cremad-root <CREMA-D>/data \
        --dump-schema
    python scripts/build_cremad_mixed.py --cremad-root <CREMA-D>/data --report
    python scripts/build_cremad_mixed.py --cremad-root <CREMA-D>/data \
        --out cremad_mixed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
from collections import Counter, defaultdict

CREMAD = None
AUDIO = None

SENT = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

FILE_EMO = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

VOTE_EMO = {
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
    "s": "sad",
    "ang": "angry",
    "dis": "disgust",
    "fea": "fear",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}
KEEP = (
    "happy",
    "sad",
    "angry",
    "surprise",
    "neutral",
)
NONNEUTRAL = ("happy", "sad", "angry", "surprise")
VOTES_CANDIDATES = [
    "finishedResponses.csv",
    "tabulatedVotes.csv",
    "summaryTable.csv",
]


def find_votes(explicit=None):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for name in VOTES_CANDIDATES:
        for d in (
            CREMAD,
            f"{CREMAD}/processedResults",
            os.path.dirname(CREMAD.rstrip("/")),
        ):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def sniff(path):
    """Infer the file's layout. Returns (rows, mapping) where mapping['fmt'] is 'wide'
    or 'long'.

    Two real layouts exist in processedResults/:
      wide  -- tabulatedVotes.csv: one row per (clip, rating condition) with
            per-emotion vote count columns
               A,D,F,H,N,S, plus fileName / numResponses / emoVote / agreement.
               The three rating conditions are
               stacked and distinguished only by the leading digit of the
               unnamed index column (see modality_values).
      long  -- finishedResponses.csv: one row per individual rater response,
            with a categorical response column.
    Never guesses silently: main() prints the mapping it inferred.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], {}
    cols = list(rows[0].keys())
    low = {c: (c or "").lower() for c in cols}

    def pick(*needles, exclude=()):
        for c in cols:
            n = low[c]
            if n and any(x in n for x in needles) and not any(x in n for x in exclude):
                return c
        return None

    clip = pick("clipname", "filename", "file_name", "fname", "clip")
    # wide layout: single-letter vote-count columns, one per CREMA-D rater
    # option
    wide = [c for c in cols if c in ("A", "D", "F", "H", "N", "S")]
    if clip and len(wide) >= 5:
        idx = cols[0] if not cols[0] else pick("index")
        return rows, {
            "fmt": "wide",
            "clip": clip,
            "count_cols": wide,
            "index": idx,
            "modality": idx,
            "n_col": pick("numresponses"),
        }
    emo = pick(
        "respemo",
        "response",
        "vote",
        "emo",
        exclude=("level", "acted", "clip", "file", "num"),
    )
    mod = pick("querytype", "modality", "condition", exclude=("clip", "file"))
    if mod and len({r.get(mod) for r in rows}) > 6:
        mod = None
    return rows, {"fmt": "long", "clip": clip, "emo": emo, "modality": mod}


def modality_values(rows, m):
    """The distinct rating-condition keys present. For the wide layout these are the
    index column's leading digit."""
    if m.get("modality") is None:
        return [None]
    if m["fmt"] == "wide":
        return sorted(
            {str(r.get(m["modality"], ""))[:1] for r in rows if r.get(m["modality"])}
        )
    return sorted({str(r.get(m["modality"])) for r in rows})


def acc_acted(counts):
    """Fraction of clips whose majority vote matches the acted label in the filename.

    This identifies the rating condition without needing a key the CREMA-D
    release never documents: the corpus reports voice-only 40.9%, face-only
    58.2%, audio-visual 63.6%, so the audio-only block is simply the least
    accurate one.
    """
    hit = tot = 0
    for cid, cnt in counts.items():
        p = cid.split("_")
        if len(p) != 4 or p[2] not in FILE_EMO:
            continue
        tot += 1
        hit += majority(cnt) == FILE_EMO[p[2]]
    return hit / tot if tot else 0.0


def pick_audio_only(rows, m):
    """The modality key whose acc_acted is lowest, i.e. the audio-only condition."""
    scored = [
        (acc_acted(vote_counts(rows, m, mod)), mod) for mod in modality_values(rows, m)
    ]
    return min(scored)[1] if scored else None


def vote_counts(rows, m, modality=None):
    """{clip_id: Counter(canonical label -> votes)}, restricted to one rating condition
    when given."""
    out = defaultdict(Counter)
    for r in rows:
        if modality is not None and m.get("modality") is not None:
            key = str(r.get(m["modality"], ""))
            if (key[:1] if m["fmt"] == "wide" else key) != str(modality):
                continue
        cid = (
            str(r.get(m["clip"], ""))
            .strip()
            .strip('"')
            .replace(".wav", "")
            .replace(".mp4", "")
            .replace(".flv", "")
        )
        if not cid:
            continue
        if m["fmt"] == "wide":
            for c in m["count_cols"]:
                try:
                    v = float(str(r.get(c, "0")).strip() or 0)
                except ValueError:
                    continue
                if v:
                    out[cid][VOTE_EMO[c.lower()]] += v
            continue
        raw = str(r.get(m["emo"], "")).strip().strip('"').lower()
        if not raw:
            continue
        # a tie like "A:H" is one rater splitting a vote; credit each side 0.5
        # so totals stay per-rater
        parts = [p for p in raw.replace("/", ":").split(":") if p]
        labs = [VOTE_EMO.get(p) for p in parts]
        if not labs or any(lab is None for lab in labs):
            continue
        for lab in labs:
            out[cid][lab] += 1.0 / len(labs)
    return out


# --------------------------------------------------------------------------- #
# The paper's filters
# --------------------------------------------------------------------------- #
def majority(cnt):
    top = max(cnt.values())
    winners = sorted(k for k, v in cnt.items() if v == top)
    return winners[0] if len(winners) == 1 else None  # a tie has no majority-vote label


def apply_filters(counts, nonneutral="distinct3", outside_votes="drop"):
    """(kept rows, diagnostic Counter). Each kept row is (clip_id, provided_label,
    dominant, {emo: p})."""
    kept, diag = [], Counter()
    for cid, cnt in counts.items():
        parts = cid.split("_")
        if len(parts) != 4:
            continue
        actor, sent, ecode, _ = parts
        provided = FILE_EMO.get(ecode)
        if provided is None or sent not in SENT:
            continue
        diag["all"] += 1

        outside = {k: v for k, v in cnt.items() if k not in KEEP and v > 0}
        if outside:
            if outside_votes == "drop":
                continue
            cnt = Counter({k: v for k, v in cnt.items() if k in KEEP})
            if not cnt or sum(cnt.values()) <= 0:
                continue
        diag["within_5"] += 1

        dom = majority(cnt)
        if dom is None:
            diag["tie_no_majority"] += 1
            continue
        if dom == provided:
            # the set is defined by disagreement with the provided label
            continue
        diag["differs_from_provided"] += 1

        nn_distinct = sum(1 for e in NONNEUTRAL if cnt.get(e, 0) > 0)
        nn_votes = sum(cnt.get(e, 0) for e in NONNEUTRAL)
        diag["distinct3"] += nn_distinct >= 3
        diag["distinct2"] += nn_distinct >= 2
        diag["votes3"] += nn_votes > 2
        diag["none"] += 1
        # On the voice-only ratings the other filters alone yield 774 clips,
        # while every reading of the "more than two non-neutral labels" clause
        # cuts far more (distinct3 11, votes3 306, distinct2 185). The default
        # is therefore "none"; the stricter readings remain available via
        # --nonneutral.
        ok = {
            "distinct3": nn_distinct >= 3,
            "distinct2": nn_distinct >= 2,
            "votes3": nn_votes > 2,
            "none": True,
        }[nonneutral]
        if not ok:
            continue

        tot = sum(cnt.values())
        kept.append((cid, provided, dom, {e: cnt.get(e, 0) / tot for e in KEEP}))
    return kept, diag


# --------------------------------------------------------------------------- #
# Reference audio: same actor, neutral, preferably a different sentence
# --------------------------------------------------------------------------- #
def neutral_refs():
    by_actor = defaultdict(dict)
    for p in glob.glob(f"{AUDIO}/*_NEU_*.wav"):
        parts = os.path.basename(p)[:-4].split("_")
        if len(parts) == 4:
            by_actor[parts[0]][parts[1]] = p
    return by_actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--votes",
        default=None,
        help="path to finishedResponses.csv / tabulatedVotes.csv",
    )
    ap.add_argument(
        "--modality",
        default="auto",
        help=(
            "rater condition to use: 'auto' (audio-only if detectable), a "
            "literal column value, or 'all'"
        ),
    )
    ap.add_argument(
        "--nonneutral",
        default="none",
        choices=["none", "distinct3", "votes3", "distinct2"],
    )
    ap.add_argument("--outside-votes", default="drop", choices=["drop", "renormalise"])
    ap.add_argument(
        "--cremad-root",
        required=True,
        help="CREMA-D data directory: the one containing AudioWAV/",
    )
    ap.add_argument("--out", default="cremad_mixed.csv")
    ap.add_argument("--dump-schema", action="store_true")
    # Prompt-resample replicates. The neutral-reference draw was always one
    # sample from random.Random(0); it is a nuisance parameter, not part of the
    # method, so re-drawing it under the same rule (same actor, a different
    # sentence's NEU take) gives a replicate that leaves every other column
    # byte-identical. Mirrors --ref-seed in build_iemocap_mixed_manifest.py.
    # SEED itself is inert on these backbones, so this is the only replicable
    # factor.
    ap.add_argument(
        "--ref-seed",
        type=int,
        default=0,
        help=("seed for the neutral-reference draw; 0 reproduces the shipped manifest"),
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="print filter counts for every reading and exit",
    )
    a = ap.parse_args()

    global CREMAD, AUDIO
    CREMAD = a.cremad_root.rstrip("/")
    AUDIO = f"{CREMAD}/AudioWAV"
    if not os.path.isdir(AUDIO):
        raise SystemExit(
            f"[FATAL] {AUDIO!r}"
            f" not found; --cremad-root should be the dir containing AudioWAV/"
        )

    vp = find_votes(a.votes)
    if not vp:
        print(
            "[BLOCKED] no CREMA-D rater-vote file found. The mixed set is\n"
            "defined by rater votes, so it cannot be built from the audio alone.\n"
            "Fetch one of these into your --cremad-root:\n"
            "  finishedResponses.csv   (preferred)\n"
            "  tabulatedVotes.csv\n"
            "from processedResults/ at\n"
            "  https://github.com/CheyneyComputerScience/CREMA-D"
        )
        return 2

    rows, m = sniff(vp)
    print(f"votes file : {vp}  ({len(rows)} rows, layout={m.get('fmt')})")
    print(
        f"inferred   : clip={m.get('clip')!r} "
        + (
            f"count_cols={m.get('count_cols')}"
            if m.get("fmt") == "wide"
            else f"emotion={m.get('emo')!r}"
        )
        + f" modality={m.get('modality')!r}"
    )
    if a.dump_schema:
        for r in rows[:4]:
            print("  sample:", dict(list(r.items())[:11]))
        print("  modality keys:", modality_values(rows, m))
        return 0
    if not m.get("clip") or not (m.get("count_cols") or m.get("emo")):
        print(
            "[FATAL] could not identify the clip-id and vote columns.\n"
            "Run --dump-schema and pass the file explicitly;\n"
            "this script will not guess a mapping that silently changes\n"
            "the sample count."
        )
        return 2

    mods = (
        [None]
        if (a.modality == "all" or m.get("modality") is None)
        else ([a.modality] if a.modality != "auto" else modality_values(rows, m))
    )
    if a.report:
        print(
            "\nFilter counts. The paper reports 772; the reading whose count hits 772 "
            "is the paper's."
        )
        print(
            f"  {'modality':>9} {'acc_acted':>10} {'all':>6} {'in {5}':>7} {'ties':>5} "
            f"{'!=label':>8} "
            f"{'distinct3':>10} {'votes3':>7} {'distinct2':>10}"
        )
        for mod in mods:
            c = vote_counts(rows, m, mod)
            _, d = apply_filters(c, "distinct3", a.outside_votes)
            print(
                f"  {str(mod):>9} {acc_acted(c):>10.3f} {d['all']:>6} "
                f"{d['within_5']:>7} "
                f"{d['tie_no_majority']:>5} {d['differs_from_provided']:>8} "
                f"{d['distinct3']:>10} "
                f"{d['votes3']:>7} {d['distinct2']:>10}"
            )
        print(
            "\n  Raters scored each clip audio-only / visual-only / audio-visual."
            "Audio-only is the only defensible\n"
            "condition for a TTS paper, and is identified by acc_acted ~= 0.409"
            "(CREMA-D's reported voice-only\n"
            "accuracy) -- NOT by the index key, which the release does not document."
            "Pass it via --modality."
        )
        return 0

    mod = None if a.modality == "all" else a.modality
    if a.modality == "auto":
        # Pooling every condition would mix face ratings into the soft
        # proportions and silently change the sample count, so auto resolves to
        # the audio-only block instead of averaging over all of them.
        mod = pick_audio_only(rows, m) if m.get("modality") is not None else None
        print(
            f"[auto] rating modality = {mod!r}"
            f" (lowest acc_acted, i.e. the audio-only condition). "
            f"Override with --modality; --report lists them all."
        )
    counts = vote_counts(rows, m, mod)
    kept, diag = apply_filters(counts, a.nonneutral, a.outside_votes)
    refs, rng = neutral_refs(), random.Random(a.ref_seed)

    out_rows, no_ref, same_sent = [], 0, 0
    for cid, provided, dom, p in sorted(kept):
        actor, sent, _, _ = cid.split("_")
        wav = f"{AUDIO}/{cid}.wav"
        if not os.path.exists(wav):
            continue
        cand = {s: pth for s, pth in refs.get(actor, {}).items() if s != sent}
        if cand:
            rsent = sorted(cand)[rng.randrange(len(cand))]
        elif sent in refs.get(actor, {}):
            rsent, same_sent = sent, same_sent + 1
        else:
            no_ref += 1
            continue
        out_rows.append(
            {
                "wav_filename": cid,
                "filepath": wav,
                "text": SENT[sent],
                "emotion": dom,
                "speaker": actor,
                "split": "test",
                "reference_filepath": refs[actor][rsent],
                "reference_text": SENT[rsent],
                "reference_same_sentence": int(rsent == sent),
                "provided_label": provided,
                "dominant_emotion": dom,
                **{f"p_{e}": round(p[e], 6) for e in KEEP},
            }
        )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(out_rows[0].keys()) if out_rows else ["wav_filename"],
        )
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {a.out}  n={len(out_rows)}   (paper: 772)")
    print(
        f"  reading: nonneutral={a.nonneutral} outside_votes={a.outside_votes}"
        f" modality={mod}"
    )
    print(
        f"  dropped: no same-actor neutral ref {no_ref}"
        f"; reference reuses the target sentence for {same_sent}"
    )
    print(
        "  dominant-emotion histogram: "
        + str(Counter(r["dominant_emotion"] for r in out_rows).most_common())
    )
    if len(out_rows) != 772:
        print(
            f"  [MISMATCH] n={len(out_rows)} != 772. Run --report to see "
            "which reading "
            f"of 'more than two non-neutral\n"
            f"emotion labels' and which rater modality reproduces the paper's count "
            f"before generating."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
