# EmoRES-TTS

<p align="center">
  <a href="https://huggingface.co/papers/2609.38157"><img alt="Hugging Face Paper" src="https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2609.38157"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.38157-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://emorestts.github.io/"><img alt="Webpage" src="https://img.shields.io/badge/Website-Visit-orange?logo=googlechrome&logoColor=white"></a>
</p>

Official implementation for the paper [**EmoRES-TTS: Residual-Enhanced Vector Steering for Emotional Speech Generation**](https://arxiv.org/abs/2609.38157)<br>

Residual emotion steering for emotional TTS generation. EmoRES decomposes a mixed-emotion steering vector into a shared axis and an
emotion-carrying residual and weight the two independently.

## Installation

EmoRES-TTS is implemented on top of [CoCoEmo](https://github.com/wsssy/CoCoEmo), which is a required dependency and
is not vendored here.

```bash
git clone https://github.com/facebookresearch/EmoRES-TTS.git
git clone https://github.com/wsssy/CoCoEmo.git
export PYTHONPATH=$PWD/EmoRES-TTS:$PWD/CoCoEmo
```

You also need a backbone checkpoint (IndexTTS-2 or CosyVoice2), CoCoEmo's
[steering vectors](https://github.com/wsssy/CoCoEmo/tree/main/steering_vectors), `emotion2vec_plus_large`, and
[ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) for the anchored E-SIM anchors
(`data/esd_anchors.txt` lists the 400 clips by path).

## Usage

Build a manifest, synthesize, then score.

```bash
# 1. manifest (mixed-emotion targets, rebuilt from the corpus annotations)
python scripts/build_iemocap_mixed.py --iemocap-root <IEMOCAP> --out iemocap_mixed.csv

# 2. synthesize EmoRES, CoCoEmo, and no-steer
python scripts/synthesize_emores.py --lambda-c 1 --lambda-r 3 --alpha 5 \
    --backbone cosyvoice2 --model-dir <ckpt> --steering-dir $SD \
    --manifest iemocap_mixed.csv --output-dir out/res           # EmoRES
python scripts/synthesize_emores.py --lambda-c 1 --lambda-r 1 --raw --alpha 5 ... --output-dir out/cocoemo
python scripts/synthesize_emores.py --nosteer               ... --output-dir out/nosteer

# 3. TEP / rho / H-Rate / S-SIM / WER
python <CoCoEmo>/scripts/evaluate.py --wav-dir out/res --baseline-wav-dir out/nosteer \
    --manifest iemocap_mixed.csv --output-dir results/res

# 4. anchored E-SIM
python scripts/score_anchored_esim.py --manifest iemocap_mixed.csv --wav-dir out/res \
    --esd-root "<ESD>/Emotion Speech Dataset" --cache-dir .cache --out res_esim.json
```

## Evaluation dataset

No corpus audio, transcripts or embeddings are redistributed. Obtain each under its own terms and point the
scripts at your copy.

| corpus | used for | notes |
|---|---|---|
| [IEMOCAP](https://sail.usc.edu/iemocap/) | evaluation set | `data/iemocap_1083_ids.txt` lists the 1,083 ids |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | evaluation set | needs the rater votes as well, see below |

CREMA-D needs a second download: the rater votes that define the mixed-emotion targets are not in the audio
release. Fetch `processedResults/finishedResponses.csv` or `tabulatedVotes.csv` from the repo above into
your `--cremad-root`, then run

```bash
python scripts/build_cremad_mixed.py --cremad-root <CREMA-D>/data --out cremad_mixed.csv
```

## Citation

```bibtex
@article{huang2026emores,
  title         = {EmoRES-TTS: Residual-Enhanced Vector Steering for Emotional Speech Generation},
  author        = {Huang, Kuan-Po and Liu, Haohe and Peng, Puyuan and Wu, Haibin and
                   Ni, Zhaoheng and Lee, Hung-yi and Lee, Jinwon and Chachra, Neha},
  journal       = {arXiv preprint arXiv:2609.38157},
  year          = {2026},
  eprint        = {2609.38157},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2609.38157}
}
```

## License

CC BY-NC 4.0, see [LICENSE](LICENSE). Non-commercial use only.
Third-party models and corpora remain under their own licences.

## Acknowledgements

This work builds directly on [CoCoEmo](https://github.com/wsssy/CoCoEmo) (MIT). The steering vectors, the
injection hooks, the backbone adapters and the TEP / rho / H-Rate / S-SIM / WER implementations are theirs.
EmoRES imports them rather than reimplementing them, so the two produce numbers from the same code. The
anchored E-SIM formulation follows [EmoSteer-TTS](https://arxiv.org/abs/2508.03543).
