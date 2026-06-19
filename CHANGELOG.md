# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Migrated the TensorFlow backend to TensorFlow 2.16+ / Keras 3.** The previous
  cap of `tensorflow>=2.13,<2.16` is replaced by `tensorflow>=2.16`, with an
  explicit `keras>=3.0.0` dependency for the TensorFlow extra.
  - Replaced removed `keras.backend` (`K.*`) ops with native `tf.*` equivalents
    in losses, evaluation metrics, and custom layers.
  - Dropped the removed `Embedding(input_length=...)` argument across models.
  - Renamed the custom metric `reset_states()` to `reset_state()` (Keras 3).
  - Updated `CyclicLR` to set the learning rate via
    `optimizer.learning_rate.assign(...)` instead of the removed
    `tf.keras.backend.set_value(optimizer.lr, ...)`.
  - Replaced the unserializable `Lambda` one-hot layer in the Detectability
    model with a direct op.
  - Added/repaired `get_config`/`from_config` on all TensorFlow models so full
    `.keras` save/load round-trips work, and added `build()` methods so
    `model.build(input_shape)` instantiates weights under Keras 3.
- Updated `run_scripts/` and the example notebooks for Keras 3 (optimizer
  `learning_rate=` argument; `.weights.h5` paths for `save_weights`).
- Modernized the documentation toolchain (`docs/requirements.txt`): Sphinx and
  `sphinx-book-theme` bumped to current releases; removed the unused
  `readthedocs-sphinx-search` dependency.
- CI now runs on pull requests and verifies formatting with `black --check`
  (via `make format-check`) instead of reformatting in place.

### Added
- `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.

## [0.2.7]
- Maintenance release.

## [0.2.6]
- Fixes for alphabet learning and streamlined default behavior for the number of
  processes used during Hugging Face datasets processing.
- DeepLC model revamp.

## [0.2.0]
- Introduced multi-backend support for TensorFlow/Keras and PyTorch with a
  shared public API. Earlier versions supported TensorFlow/Keras only.

[Unreleased]: https://github.com/wilhelm-lab/dlomix/compare/v0.2.7...HEAD
