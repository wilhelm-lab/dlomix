# Contributing to DLOmix

Thanks for your interest in contributing! This guide covers the workflow,
the multi-backend design you need to be aware of, and how to run the checks
that CI enforces.

## Getting started

```bash
# clone your fork, then from the repo root in a fresh virtualenv:
make install-dev      # editable install with dev deps (pytest, black, pylint, both backends)
```

Python >= 3.11 is required. The TensorFlow backend requires TensorFlow 2.18+
(Keras 3); the PyTorch backend requires PyTorch 2.3+. `python -m dlomix` prints
the versions and backend in use.

If you change dependencies or extras in `setup.py`, check that every extra still
resolves on every platform — this is what the `resolve-extras` CI job runs, and it
needs no GPU:

```bash
for p in x86_64-manylinux_2_28 aarch64-apple-darwin x86_64-pc-windows-msvc; do
  echo "dlomix[tf-cuda] @ file://$PWD" | uv pip compile - --python-version 3.11 --python-platform $p
done
```

## Backend selection (important)

DLOmix supports two backends, selected **at import time** from the
`DLOMIX_BACKEND` environment variable (`tensorflow`/`tf` or
`pytorch`/`torch`/`pt`, defaulting to TensorFlow). It must be set **before**
importing `dlomix`:

```bash
export DLOMIX_BACKEND=tensorflow   # or: pytorch
```

DLOmix also derives `KERAS_BACKEND` from `DLOMIX_BACKEND`, so **import
`dlomix` before `keras`** (it warns when you do not). Because `tf.keras` *is*
Keras 3 on TF 2.16+, only one backend can be live in a process.

**Models and layers** are backend-specific and live in parallel files:
TensorFlow in the bare name (`prosit.py`, `chargestate.py`), PyTorch in a
`_torch` suffix (`prosit_torch.py`, `chargestate_torch.py`). When you change a
public API, **update both backends**, keep the public class names identical,
and wire the new class into the correct `_BACKEND` branch of the relevant
subpackage `__init__.py`.

**Losses and metrics** (`losses/`, `eval/`) are *not* split. They are pure
tensor math written once against `keras.ops` and run on either backend, so add
new ones as backend-agnostic `keras.ops` code — do not write a `_torch` copy.
Convert inputs with `ops.convert_to_tensor` so raw numpy arrays work too, and
return per-sample values rather than a reduced scalar. The exception is
`MaskedIonmobLoss`, a stateful `nn.Module` for the PyTorch-only Ionmob model.

## Development workflow

1. Create a feature branch off `develop` (PRs target `main` or `develop`).
2. Make your change, with tests, in both backends where applicable.
3. Run the checks below locally — CI runs the same ones on your PR.
4. Open a pull request describing the change and the motivation.

## Checks (run these before pushing)

```bash
make format         # apply black + isort (profile black)
make format-check   # verify formatting without modifying files (what CI runs)
make lint-errors-only
make test           # full pytest suite with coverage (reinstalls the package)
```

For faster iteration against an existing editable install:

```bash
python -m pytest tests/test_models.py -v
```

Tests import backend-specific modules directly (e.g.
`from dlomix.models.chargestate import ...` vs `...chargestate_torch import ...`),
so the pure-PyTorch modules also run under the default TensorFlow backend.
Anything built on Keras follows the single process-wide `KERAS_BACKEND`, so
tests that compare a `tf.keras` model with a torch model skip unless it is
`tensorflow`. New behavior should be covered for both backends when it is
backend-specific. The suite downloads
small example datasets on first run, so a network connection is needed on a
cold cache.

`tests/test_losses.py` and `tests/test_eval.py` cover the shared losses and
metrics. They are backend-neutral — plain lists in, `keras.ops.convert_to_numpy`
out — and assert fixed reference values held in an `EXPECTED` table. CI runs them
under **both** backends, so those two runs agreeing on the same numbers is the
guard against the backends drifting apart. If you change one of these functions
and the reference values move, confirm the new numbers under both backends before
updating the table:

```bash
python -m pytest tests/test_losses.py tests/test_eval.py -v
DLOMIX_BACKEND=pytorch python -m pytest tests/test_losses.py tests/test_eval.py -v
```

## Code style

- Formatting is enforced by **black + isort (profile black)**; CI fails on
  unformatted code.
- Pre-commit hooks (`.pre-commit-config.yaml`) add autoflake (unused-import
  removal) and end-of-file/whitespace fixes. Note that
  `src/dlomix/_register_keras.py` is intentionally excluded from autoflake —
  do not let its imports be stripped.
- Keep public class and function names identical across the two backends.
- Register serializable classes and functions with
  **`@keras.saving.register_keras_serializable(package="dlomix")`**. On Keras 3
  this is literally the same function as `tf.keras.utils.register_keras_serializable`
  (`tf.keras` is a shim over Keras), but reaching it through `tf.keras` requires
  importing TensorFlow — which the backend-agnostic modules must not do. One
  spelling everywhere avoids implying there are two mechanisms.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example, the backend
in use (`DLOMIX_BACKEND`), and the versions of `dlomix`, `tensorflow`/`torch`,
and Python.
