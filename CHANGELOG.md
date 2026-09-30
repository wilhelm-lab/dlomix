# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


### Added
- **`tf-cuda` / `tensorflow-cuda` extra** for Linux GPU machines, wrapping
  `tensorflow[and-cuda]` (CUDA and cuDNN as pip wheels, so no system CUDA toolkit is
  needed). It is gated to Linux: TensorFlow's own extra carries no platform
  markers, and NVIDIA's CUDA wheels exist only for Linux, so ungated it fails to
  resolve on macOS and Windows. Elsewhere it installs the regular build. PyTorch
  needs no equivalent — its Linux wheels on PyPI are already CUDA builds.
- **`python -m dlomix`** prints an environment report: DLOmix, Python, framework
  versions, the active backend, and the GPUs visible to it, with hints for the
  common Linux GPU failure modes.
- A `resolve-extras` CI job resolves every extra on Linux x86_64/aarch64, macOS and
  Windows, for Python 3.11 and 3.12, using `uv pip compile --python-platform`. It
  reads only package metadata — no GPU runner, no wheel downloads — and asserts that
  `tf-cuda` brings the CUDA libraries on Linux and only there.
- `CITATION.cff`, `CONTRIBUTING.md`, and this changelog.

### Changed
- **Migrated the TensorFlow backend to TensorFlow 2.18+ / Keras 3.** The previous
  cap of `tensorflow>=2.13,<2.16` is replaced by `tensorflow>=2.18` (see *Fixed* for
  why not 2.16), with `keras>=3.0.0` as a core dependency.
  - Replaced the legacy `keras.backend` (`K.*`) API: tensor ops with native `tf.*`
    equivalents in the custom layers, and `epsilon()` / `floatx()` with
    `keras.config.*` (the losses and metrics now use `keras.ops` throughout).
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
- **Unified the losses and metrics across backends.** `masked_spectral_distance`,
  `masked_pearson_correlation_distance`, `adjusted_mean_absolute_error`,
  `adjusted_mean_squared_error`, `timedelta` and `TimeDeltaMetric` now have a
  single implementation written against `keras.ops`, replacing the two
  hand-written copies that had drifted apart. `MaskedIonmobLoss` stays
  PyTorch-only.
  - `keras>=3.0.0` moved from the TensorFlow extra into `install_requires`, and
    `dlomix.config` now derives `KERAS_BACKEND` from `DLOMIX_BACKEND` before
    `keras` is imported. Keras 3 is pure Python and does not pull in TensorFlow
    when running on the PyTorch backend. **`dlomix` must be imported before
    `keras`**; it warns when it was not.
  - *Breaking (PyTorch):* `masked_spectral_distance` now returns one value **per
    sample** instead of a pre-reduced scalar, matching the Keras convention and
    the previous TensorFlow behaviour. Hand-written PyTorch training loops must
    call `.mean()` before `.backward()`.
  - *Breaking (PyTorch):* `adjusted_mean_absolute_error` / `adjusted_mean_squared_error`
    took `(y_pred, y_true)` and masked components that were zero in **either**
    vector. They now take `(y_true, y_pred)` and discard only components that are
    zero in **both** — the documented behaviour, and what TensorFlow already did.
    Previously reported PyTorch values were too low.
  - *Breaking (PyTorch):* `TimeDeltaMetric` is now a `keras.metrics.Metric`
    (`percentage`, `name`, `double_delta`, `normalize`) rather than a plain
    callable, and is still directly callable. The `timedelta` function takes
    `(y_true, y_pred, normalize, percentage)`, the TensorFlow argument order.
  - `masked_spectral_distance` now clips the dot product to `[-1, 1]` before
    `arccos`, so rounding error can no longer produce `NaN`.
  - `tests/test_losses.py` and `tests/test_eval.py` are backend-neutral and assert
    fixed reference values; CI runs both files under both backends, so those runs
    agreeing is the cross-backend parity check.
- Standardized on `@keras.saving.register_keras_serializable(package="dlomix")`.
  It is the same function as `tf.keras.utils.register_keras_serializable` on
  Keras 3, but reaching it through `tf.keras` requires importing TensorFlow, which
  the backend-agnostic modules cannot do.
- Raised the minimum Python to 3.11 and NumPy to 2.0, matching what CI tests.
- CI now also runs on `feature/**` branches, reports the environment via
  `python -m dlomix`, and adds a `DLOMIX_BACKEND=pytorch` job so the shared code
  paths are exercised on both backends. It installs CPU-only PyTorch: PyPI's Linux
  torch wheels bring a full CUDA stack (~15 NVIDIA packages, several GB) that a
  GPU-less runner cannot use.
- TensorFlow is capped at `<2.22`, the newest version CI installs and tests.
  Previously any future release was accepted on Linux; raise the cap deliberately.

### Removed
- **`dlomix.losses.intensity_torch`, `dlomix.eval.chargestate_torch` and
  `dlomix.eval.rt_eval_torch`.** Their contents now live in the backend-agnostic
  `dlomix.losses.intensity`, `dlomix.eval.chargestate` and `dlomix.eval.rt_eval`.
  Import from `dlomix.losses` / `dlomix.eval` (or those modules) instead.
- Support for Python 3.10, which reaches end of life in October 2026. The CI
  matrix already only covered 3.11 and 3.12.

### Fixed
- **Packaging.** Three dependency declarations could resolve to broken installs:
  - The macOS `tensorflow<2.20` cap was computed with `platform.system()` inside
    `setup.py`, i.e. when the wheel was *built*. The wheel is `py3-none-any`, so the
    result applied to every platform: the Linux-built PyPI wheel carried no cap for
    macOS users, and a macOS-built one capped everyone. It is now a PEP 508
    environment marker, evaluated on the installing machine.
  - `tensorflow>=2.16` could never resolve to 2.16 or 2.17, which pin `numpy<2`
    against the core `numpy>=2.0`. The floor is now the honest `>=2.18`.
  - `torch` was unpinned. torch does not declare numpy as a dependency, so pip
    would pair a pre-2.3 torch — compiled against NumPy 1.x, and failing at import
    under NumPy 2 — with `numpy>=2.0`. Now `torch>=2.3`.
  - Dropped `torchvision`, which nothing imports.
- `dlomix.reports` could not be imported without TensorFlow, contradicting its
  backend-agnostic design: `Report.py` imported it for a single
  `isinstance(history, tf.keras.callbacks.History)` check (now duck-typed on a
  `history` dict attribute, which also accepts histories from PyTorch loops), and
  `postprocessing.py` for its opt-in legacy TF1 path (now a lazy import).
- The PyPI release workflow no longer invokes the deprecated `python setup.py
  --version`.
- **`DetectabilityModel.build(input_shape)` created no weights, making
  `load_weights` a silent no-op.** Keras 3's default `Model.build` only sets the
  `built` flag; it never instantiates sub-layers created in `__init__`. So
  `build(...)` followed by `load_weights(...)` raised nothing, loaded nothing, and
  the first forward pass produced fresh random weights — the detectability
  fine-tuning walkthrough appeared to resume from the base model while actually
  training from scratch (visible in its own saved outputs: the base model reaches
  `val_sparse_categorical_accuracy` 0.601, then "fine-tuning" restarts at 0.298).
  `DetectabilityModel` now has an explicit `build()`, and `tests/test_detectability_model.py`
  pins the contract.
- **The shipped Detectability checkpoints could not be loaded under Keras 3 at all.**
  `pretrained_models/` held TensorFlow-native checkpoints (`.index` +
  `.data-00000-of-00001`), which Keras 3 rejects with
  `File format not supported`. Each folder now also ships a converted
  `.weights.h5`, produced by the new `scripts/convert_detectability_checkpoints.py`
  (verified bit-identical to the original checkpoint scored under TensorFlow 2.15 /
  Keras 2). `Example_Detectability_Model_Walkthrough_prediction_colab.ipynb` loads
  the converted file and builds the model before loading, as Keras 3 requires.
- **`InferencePipeline` rejected correct `with_termini=True` pipelines.** Its
  consistency check compared the model's *padded* input width against the
  preprocessor's *configured* `max_seq_len`, which excludes the two terminal
  positions the preprocessor adds itself — so a model and preprocessor that both
  produced width 32 were reported as a mismatch at 32 vs 30. This fired for every
  `PrositIntensityPredictor` pipeline built with `with_termini=True`, the default on
  both the dataset and the model. `PeptidePreprocessor` now exposes `padded_seq_len`
  (the width it actually emits) and the check compares against that. The
  `max_seq_len + 2 if with_termini` rule, previously written out in four places, now
  lives only in `dlomix.data.processing.chain.padded_sequence_length`. The error
  message also reports both effective widths and names `with_termini` as the cause.
- `DetectabilityReport` used `np.round_`, removed in NumPy 2.0, so every report
  raised `AttributeError` at call time. The module has no test coverage, which is
  why the suite stayed green.
- `run_scripts/run_deeplc.py` imported `dlomix.data.RetentionTimeDataset` and
  `run_scripts/run_pipeline.py` imported `TimeDeltaMetric2`; neither has existed
  since the dataset refactor. Both scripts failed at import.
- Two notebooks called `save_weights` with a path lacking the `.weights.h5`
  suffix that Keras 3 requires, and two others pinned `dlomix[wandb]==0.1.0`.
- `TimeDeltaMetric.get_config` did not call `super().get_config()`, dropping
  `name` and `dtype` and breaking the `from_config` round-trip.
- Added `isort` to the `dev` extra; `make format-check`, which CI runs, invoked it
  without declaring it.

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
