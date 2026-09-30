# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EmoRES: the shared/residual decomposition of an emotion steering vector, and its
norm-matched form.

The decomposition:
    c     = mean_e v_e                  the shared axis: the part common to
                                        every emotion in the library.
                                        One vector, identical whatever emotion
                                        is requested, so it cannot by itself
                                        encode which emotion was asked for.
    R_p   = coco - c                    the residual: what distinguishes the
                                        requested mixture from that common part.
                                        This is where emotion identity lives.
    v     = lambda_c * c + lambda_r * R_p

lambda_c = lambda_r = 1 recovers CoCoEmo exactly, because c cancels: 1*c +
1*(coco - c) = coco. That is not an approximation and `res()` asserts it,
because if it fails the decomposition is wrong and every other setting is
unreadable.

Why `abn` exists, and why it is not optional for a fair comparison. Raising
lambda_r also lengthens v, so a bare EmoRES arm beats CoCoEmo partly on dose
rather than on direction. `abn` rescales v to rbar = mean_e ||v_e|| -- the
energy the convex mixture itself spends -- so every (lambda_c, lambda_r) injects
the same norm and the knobs act only through their relative size. Comparisons in
the paper are all made in this form. Under `abn`, abn(1,1) is CoCoEmo rescaled
to rbar, which is the rule's free correctness check.

Adapted from CoCoEmo (MIT). The per-emotion vector library and the convex
mixture ``sum_e p_e v_e`` are theirs, implemented in
``cocoemo/steering/_core_cosyvoice.py:718``
(``create_sample_specific_mixed_vectors``). What this file adds is the
decomposition of that mixture into a shared axis and a residual, the
reweighting of the two, and the norm-matched form ``abn``.
"""

from __future__ import annotations


def shared_axis(V):
    """c = mean_e v_e over the active emotion set. V: (n_emotions, d)."""
    return V.mean(0)


def mixture(V, p):
    """CoCoEmo's convex mixture, sum_e p_e v_e. `p` must already be renormalised over
    the emotions in V
    (drop neutral mass and renormalise before calling; this function does not do
    it for you, because whether neutral participates is a property of the
    manifest, not of the rule)."""
    if abs(float(p.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"p must sum to 1 over the active emotions; got {float(p.sum()):.6f}"
        )
    return (p[:, None] * V).sum(0)


def res(V, p, lambda_c=1.0, lambda_r=3.0, check=True):
    """v = lambda_c * c + lambda_r * (coco - c). Unscaled -- see `abn` for the
    norm-matched form."""
    c = shared_axis(V)
    coco = mixture(V, p)
    v = lambda_c * c + lambda_r * (coco - c)
    if check and abs(lambda_c - 1.0) < 1e-12 and abs(lambda_r - 1.0) < 1e-12:
        rel = float((v - coco).norm() / (coco.norm() + 1e-12))
        if rel > 1e-6:
            raise AssertionError(
                f"(1,1) must reproduce the CoCoEmo mixture exactly, but the relative "
                f"deviation is "
                f"{rel:.3e}"
                f". The decomposition is wrong; do not trust any other setting."
            )
    return v


def rbar(V):
    """mean_e ||v_e||: the energy the convex mixture spends, and the target `abn`
    rescales to."""
    return float(V.norm(dim=-1).mean())


def abn(V, p, lambda_c=1.0, lambda_r=3.0):
    """EmoRES's direction at CoCoEmo's energy: res(...) rescaled to rbar.

    This is the form every comparison in the paper uses. At a fixed injection
    scale alpha the arms differ only in direction, so a difference between two
    (lambda_c, lambda_r) settings cannot be attributed to dose. The injected
    vector is alpha * abn(...), so ||injected|| = alpha * rbar.
    """
    v = res(V, p, lambda_c, lambda_r)
    return v * (rbar(V) / max(float(v.norm()), 1e-8))
