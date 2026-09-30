# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility shim for CoCoEmo's CosyVoice2 adapter.

CoCoEmo calls ``inference_zero_shot`` and ``inference_instruct2`` with a
``prompt_speech_16k`` tensor (``cocoemo/steering/_core_cosyvoice.py:914,923``).
Current CosyVoice renamed that argument to ``prompt_wav`` and takes a path, so
the call is translated here rather than by editing their tree.
"""

import inspect
import logging
import os


def quiet_logging():
    for name in [
        "filelock",
        "modelscope",
        "deepspeed",
        "matplotlib",
        "numba",
        "urllib3",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    # force offline hubs (wetext / models are pre-cached; network is blocked)
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def patch_zero_shot_kwarg(
    cosyvoice, methods=("inference_zero_shot", "inference_instruct2")
):
    """Bridge inference_*(prompt_speech_16k=TENSOR) to inference_*(prompt_wav=PATH)."""
    for _m in methods:
        if hasattr(cosyvoice, _m):
            _patch_one(cosyvoice, _m)
    return cosyvoice


def _patch_one(cosyvoice, method):
    orig = getattr(cosyvoice, method)
    try:
        params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        params = {}
    if "prompt_speech_16k" in params:
        return  # already the API CoCoEmo expects
    if "prompt_wav" not in params:
        return  # unknown API; leave as-is

    def wrapper(*args, **kwargs):
        tmp_path = None
        if "prompt_speech_16k" in kwargs:
            val = kwargs.pop("prompt_speech_16k")
            if hasattr(val, "shape"):  # tensor -> temp 16k wav
                import tempfile

                import torchaudio

                wav = val.detach().cpu().float()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", dir=os.environ.get("TMPDIR"), delete=False
                ) as fd:
                    tmp_path = fd.name
                torchaudio.save(tmp_path, wav, 16000)
                kwargs["prompt_wav"] = tmp_path
            else:
                kwargs["prompt_wav"] = val

        def _gen():
            try:
                for out in orig(*args, **kwargs):
                    yield out
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        return _gen()

    setattr(cosyvoice, method, wrapper)


def load_cosyvoice2_compat(model_dir):
    """load CoCoEmo CosyVoice2 backbone + apply the kwarg shim."""
    quiet_logging()
    import cocoemo.backbones.cosyvoice2 as bb

    model = bb.load_model(model_dir)
    patch_zero_shot_kwarg(model)
    return bb, model
