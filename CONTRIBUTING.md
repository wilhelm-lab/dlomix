# Contributing to DLOmix

Thanks for your interest in contributing! This guide covers the workflow,
the multi-backend design you need to be aware of, and how to run the checks
that CI enforces.

## Getting started

```bash
# clone your fork, then from the repo root in a fresh virtualenv:
make install-dev      # editable install with dev deps (pytest, black, pylint, both backends)
```

Python >= 3.10 is required. The TensorFlow backend requires TensorFlow 2.16+
(Keras 3); the PyTorch backend requires `torch`/`torchvision`.

## Backend selection (important)

DLOmix supports two backends, selected **at import time** from the
`DLOMIX_BACKEND` environment variable (`tensorflow`/`tf` or
`pytorch`/`torch`/`pt`, defaulting to TensorFlow). It must be set **before**
importing `dlomix`:

```bash
export DLOMIX_BACKEND=tensorflow   # or: pytorch
```

Backend-specific implementations live in parallel files: TensorFlow in the
bare name (`prosit.py`, `chargestate.py`), PyTorch in a `_torch` suffix
(`prosit_torch.py`, `chargestate_torch.py`). When you change a public API,
**update both backends** and keep the public class/function names identical.
Wire new models/losses/layers into the correct `_BACKEND` branch of the
relevant subpackage `__init__.py`. See [CLAUDE.md](CLAUDE.md) for the full
architecture contract.

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
so both backends can be exercised in one process. New behavior should be
covered for both backends when it is backend-specific. The suite downloads
small example datasets on first run, so a network connection is needed on a
cold cache.

## Code style

- Formatting is enforced by **black + isort (profile black)**; CI fails on
  unformatted code.
- Pre-commit hooks (`.pre-commit-config.yaml`) add autoflake (unused-import
  removal) and end-of-file/whitespace fixes. Note that
  `src/dlomix/_register_keras.py` is intentionally excluded from autoflake —
  do not let its imports be stripped.
- Keep public class and function names identical across the two backends.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example, the backend
in use (`DLOMIX_BACKEND`), and the versions of `dlomix`, `tensorflow`/`torch`,
and Python.
