# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compose EmoRES steering vectors in the structure CoCoEmo's generation path expects.

CoCoEmo's mixed-emotion path composes Eq. 7, ``sum_e p_e v_e``, in
``create_sample_specific_mixed_vectors`` and hands the result to
``generate_steered_speech`` as ``{operation: {layer: tensor}}``. EmoRES replaces
the composition and nothing else: same vector library, same hooks, same layers,
same injection site. This module is that substitution, so the only thing
separating a CoCoEmo run from an EmoRES run is which function built the tensor.

Adapted from CoCoEmo (MIT): the output structure and the composition this
replaces are ``create_sample_specific_mixed_vectors`` in
``cocoemo/steering/_core_cosyvoice.py:718``.
"""

from __future__ import annotations

import torch

from .rule import abn, res


def load_library(steering_files, emotions):
    """{operation: {layer: (n_emotions, d) tensor}} from CoCoEmo's per-emotion .pt
    files.

    Row order follows `emotions` and is the contract every caller relies on: the
    p vector passed to `compose` must use that same order.
    """
    loaded = {}
    for e in emotions:
        if e not in steering_files:
            raise KeyError(
                f"no steering vector supplied for {e!r}"
                f"; the shared axis is the mean over the "
                f"full library, so a missing emotion changes c for every clip"
            )
        data = torch.load(steering_files[e], map_location="cpu", weights_only=False)
        loaded[e] = data.get("steering_vectors", data)

    first = loaded[emotions[0]]
    lib = {}
    for op in first:
        lib[op] = {}
        for layer in first[op]:
            rows = []
            for e in emotions:
                if op not in loaded[e] or layer not in loaded[e][op]:
                    raise KeyError(
                        f"{e} is missing {op} at layer {layer}"
                        f"; CoCoEmo's composer skips the "
                        f"emotion here, but that would drop a row from c and is not "
                        f"silent-safe"
                    )
                rows.append(torch.as_tensor(loaded[e][op][layer]).reshape(-1).float())
            lib[op][layer] = torch.stack(rows)
    return lib


def compose(lib, p, lambda_c=1.0, lambda_r=3.0, normalise=True):
    """{operation: {layer: vector}} for one clip, drop-in for
    create_sample_specific_mixed_vectors.

    p is ordered as `load_library`'s `emotions`. normalise=True applies `abn`,
    rescaling to rbar so that the injected magnitude is identical for every
    (lambda_c, lambda_r) and only the direction varies with the knobs;
    normalise=False injects the raw composed vector, whose length grows with
    lambda_r.
    """
    p = torch.as_tensor(p, dtype=torch.float32).reshape(-1)
    total = float(p.sum())
    if total <= 0:
        raise ValueError("p sums to zero: this clip requests no emotion at all")
    p = p / total
    fn = abn if normalise else res
    return {
        op: {layer: fn(V, p, lambda_c, lambda_r) for layer, V in layers.items()}
        for op, layers in lib.items()
    }
