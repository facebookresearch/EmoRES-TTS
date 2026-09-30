# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Build the IEMOCAP mixed-emotion evaluation set.

Usage:
    python scripts/build_iemocap_mixed.py --iemocap-root <IEMOCAP> --report
    python scripts/build_iemocap_mixed.py --iemocap-root <IEMOCAP> \
        --out iemocap_mixed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = None

VOTE_EMO = {
    "neutral": "neutral",
    "neutral state": "neutral",
    "happiness": "happy",
    "excited": "happy",
    "sadness": "sad",
    "anger": "angry",
    "frustration": "angry",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "other": "other",
}
KEEP = ("happy", "sad", "angry", "surprise", "neutral")

ANNOT = re.compile(r"\[[^\]]*\]")


def strip_annotations(s):
    """Remove [EVENT] spans, returning the input unchanged when there is none.

    The whitespace tidy-up below only closes the hole a removed span leaves.
    Applying it unconditionally would also rewrite rows whose text merely
    contains a double space, changing the manifest for no audible reason.
    """
    if not s or "[" not in s:
        return s
    return re.sub(r"\s{2,}", " ", ANNOT.sub(" ", s)).strip()


NONNEUTRAL = (
    "happy",
    "sad",
    "angry",
    "surprise",
)
HEADER = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\S+)")
RATER = re.compile(r"^C-([EFM])(\d+):\s*(.*?)\s*(?:\(|$)")
DUR_MIN, DUR_MAX = 2.0, 10.0


def parse_eval(path, include_self=False):
    """{turn_id: (start, end, Counter(label -> votes))} for one dialog's EmoEvaluation
    file."""
    out, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read().splitlines()
    for ln in content:
        h = HEADER.match(ln)
        if h:
            start, end, turn, _ = h.groups()
            cur = turn
            out[turn] = (float(start), float(end), Counter())
            continue
        if cur is None:
            continue
        m = RATER.match(ln)
        if not m:
            continue
        kind, _idx, labels = m.groups()
        if kind != "E" and not include_self:
            continue  # C-F1/C-M1 are the subjects rating themselves
        parts = [p.strip().lower() for p in labels.split(";") if p.strip()]
        labs = [VOTE_EMO.get(p) for p in parts]
        labs = [x for x in labs if x]
        if not labs:
            continue
        for x in labs:
            out[cur][2][x] += 1.0 / len(labs)
    return out


def transcripts(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read().splitlines()
    for ln in content:
        m = re.match(r"^(\S+)\s+\[[\d.\-]+\]:\s*(.*)$", ln.strip())
        if m and m.group(2):
            out[m.group(1)] = m.group(2).strip()
    return out


def collect(include_self=False):
    """All utterances with votes, duration, wav path, transcript and speaker id."""
    rows = {}
    for ses in sorted(glob.glob(f"{ROOT}/Session*")):
        tdir = f"{ses}/dialog/transcriptions"
        for ep in sorted(glob.glob(f"{ses}/dialog/EmoEvaluation/*.txt")):
            dialog = os.path.basename(ep)[:-4]
            tr = (
                transcripts(f"{tdir}/{dialog}.txt")
                if os.path.exists(f"{tdir}/{dialog}.txt")
                else {}
            )
            for turn, (st, en, cnt) in parse_eval(ep, include_self).items():
                wav = f"{ses}/sentences/wav/{dialog}/{turn}.wav"
                if not os.path.exists(wav) or not cnt:
                    continue
                g = re.search(r"_([FM])\d+$", turn)
                rows[turn] = {
                    "turn": turn,
                    "wav": wav,
                    "dur": en - st,
                    "text": tr.get(turn, ""),
                    "speaker": f"{os.path.basename(ses)}_{g.group(1) if g else '?'}",
                    "votes": cnt,
                }
    return rows


def majority(cnt):
    top = max(cnt.values())
    win = sorted(k for k, v in cnt.items() if v == top)
    return win[0] if len(win) == 1 else None


def apply_filters(rows, outside_votes="drop", min_nonneutral=0):
    """Apply the evaluation-set filters and return (kept, diagnostics).

    Clips are kept when their votes fall inside the five emotions, their
    duration is within bounds, their text survives annotation stripping, the
    votes give a majority, and at least `min_nonneutral` distinct non-neutral
    emotions receive votes. `outside_votes` selects whether a clip with a vote
    outside the five is dropped or has its distribution renormalised over them.

    The returned counter records how many clips survive each stage, so the
    effect of every filter choice is visible via --report.
    """
    kept, diag = [], Counter()
    for r in rows.values():
        diag["all"] += 1
        cnt = r["votes"]
        if any(v > 0 for k, v in cnt.items() if k not in KEEP):
            if outside_votes == "drop":
                continue
            cnt = Counter({k: v for k, v in cnt.items() if k in KEEP})
            if not cnt or sum(cnt.values()) <= 0:
                continue
        diag["within_5"] += 1
        if not (DUR_MIN <= r["dur"] <= DUR_MAX):
            continue
        diag["duration_ok"] += 1
        r["text"] = strip_annotations(r["text"])
        if not r["text"]:
            continue
        diag["has_text"] += 1
        dom = majority(cnt)
        if dom is None:
            diag["tie_no_majority"] += 1
            continue
        nnd = sum(1 for e in NONNEUTRAL if cnt.get(e, 0) > 0)
        diag["nn2"] += nnd >= 2
        if nnd < min_nonneutral:
            continue
        diag["kept"] += 1
        tot = sum(cnt.values())
        kept.append((r, dom, {e: cnt.get(e, 0) / tot for e in KEEP}))
    return kept, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exclude-self",
        dest="include_self",
        action="store_false",
        help="drop the C-F1/C-M1 self-evaluations; they are counted by default",
    )
    ap.set_defaults(include_self=True)
    ap.add_argument(
        "--outside-votes",
        default="renormalise",
        choices=["drop", "renormalise"],
    )
    ap.add_argument(
        "--min-nonneutral",
        type=int,
        default=2,
        help=(
            "minimum distinct non-neutral emotions receiving votes; 2 is what "
            "makes rho/H-Rate defined"
        ),
    )
    ap.add_argument(
        "--iemocap-root",
        required=True,
        help=(
            "root of your IEMOCAP release: the directory containing Session1/ "
            "... Session5/"
        ),
    )
    ap.add_argument("--out", default="iemocap_mixed.csv")
    ap.add_argument("--report", action="store_true")
    ap.add_argument(
        "--ref-seed",
        type=int,
        default=0,
        help=(
            "seed for the neutral-reference draw; change it to get paired replicates"
        ),
    )
    a = ap.parse_args()

    global ROOT
    ROOT = a.iemocap_root
    if not glob.glob(f"{ROOT}/Session*"):
        raise SystemExit(
            f"[FATAL] no Session* under {ROOT!r}"
            f"; --iemocap-root should contain Session1..Session5"
        )

    if a.report:
        print(
            "Filter counts. The paper reports 1,055 (after a pyannote multi-speaker "
            "filter we cannot apply offline,"
        )
        print("so our counts should land at or slightly above theirs).")
        print(
            f"  {'raters':>12} {'outside':>12} {'all':>7} {'in {5}':>8} {'2-10s':>8} "
            f"{'has text':>9} "
            f"{'ties':>6} {'kept':>7}"
        )
        for slf in (False, True):
            rows = collect(slf)
            for ov in ("drop", "renormalise"):
                _, d = apply_filters(rows, ov, 2)
                print(
                    f"  {'E+self' if slf else 'E only':>12} {ov:>12} {d['all']:>7} "
                    f"{d['within_5']:>8} "
                    f"{d['duration_ok']:>8} {d['has_text']:>9} "
                    f"{d['tie_no_majority']:>6} {d['kept']:>7}"
                )
        return 0

    rows = collect(a.include_self)
    kept, diag = apply_filters(rows, a.outside_votes, a.min_nonneutral)

    refs = defaultdict(list)
    for r in rows.values():
        if (
            DUR_MIN <= r["dur"] <= DUR_MAX
            and majority(r["votes"]) == "neutral"
            and strip_annotations(r["text"]).strip()
        ):
            refs[r["speaker"]].append(r)
    rng = random.Random(a.ref_seed)

    out_rows, no_ref = [], 0
    for r, dom, p in sorted(kept, key=lambda x: x[0]["turn"]):
        cand = [x for x in refs.get(r["speaker"], []) if x["turn"] != r["turn"]]
        if not cand:
            no_ref += 1
            continue
        ref = cand[rng.randrange(len(cand))]
        out_rows.append(
            {
                "wav_filename": r["turn"],
                "filepath": r["wav"],
                "text": r["text"],
                "emotion": dom,
                "speaker": r["speaker"],
                "split": "test",
                "duration": round(r["dur"], 3),
                "reference_filepath": ref["wav"],
                "reference_text": strip_annotations(ref["text"]),
                "dominant_emotion": dom,
                "pyannote_filtered": 0,
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
    nn2 = sum(1 for r in out_rows if sum(1 for e in NONNEUTRAL if r[f"p_{e}"] > 0) >= 2)
    nn1 = sum(1 for r in out_rows if sum(1 for e in NONNEUTRAL if r[f"p_{e}"] > 0) >= 1)
    print(f"wrote {a.out}  n={len(out_rows)}   (paper: 1055)")
    print(
        f"  raters={'E+self' if a.include_self else 'E only'} outside_votes="
        f"{a.outside_votes}; "
        f"min_nonneutral={a.min_nonneutral}"
        f"; dropped for no same-speaker neutral reference: {no_ref}"
    )
    print(
        f"  TEP scoreable (>=1 non-neutral target): {nn1}"
        f"   rho/H-Rate scoreable (>=2): {nn2}"
    )
    print(
        "  dominant histogram: "
        + str(Counter(r["dominant_emotion"] for r in out_rows).most_common())
    )
    print(
        "Note: the paper's pyannote multi-speaker filter is not applied (model "
        "unavailable offline), so this set"
    )
    print(
        "        is a superset of theirs by however many cross-talk clips they removed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
