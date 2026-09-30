# Contributing to EmoRES

We want to make contributing to this project as easy and transparent as possible.

## What this repository does and does not contain

EmoRES ships a steering **rule** and an **evaluation**, and nothing else. It deliberately contains no model code, no model weights, no steering vectors and no corpus audio. Everything it operates on is supplied by the user, from sources they obtain under those sources' own terms. Please keep it that way -- see "Third-party material" below before opening a pull request that adds files.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Confirm your contribution adds no third-party material (see below).

## Third-party material

This is the one rule we ask you to read in full, because a mistake here is not something a later commit can undo.

**Do not add**, in any form, whole or partial:

- model code, weights, checkpoints or configs for any TTS backbone,
- steering vectors, emotion vector libraries, or other fitted artifacts you did not produce yourself,
- audio from any speech corpus, or transcripts of it,
- cached embeddings computed over corpus audio, which are derivatives of that corpus and carry its terms with them.

**Do add**, freely: code, identifiers, and values you derived. A list of filenames pointing into a corpus the user already has is a reference, not a redistribution, and is welcome. `data/esd_anchors_release.json` is the pattern to follow -- relative paths, our own selection diagnostics, no audio and no transcripts.

If a change requires third-party material to be useful, prefer a link and a documented setup step over vendoring it. If you are unsure which side of the line something falls on, open an issue before writing the code.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue. For reproduction reports, state which backbone, which vector library, and which `alpha` and `(lambda_c, lambda_r)` you used -- without those the numbers are not comparable to anything.

## License

By contributing to EmoRES, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
