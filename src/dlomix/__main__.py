"""Environment report: ``python -m dlomix``.

Prints what a bug report needs -- versions, the active backend, and whether that
backend can see a GPU. Only the active backend is imported: DLOmix never has both
frameworks live in one process.
"""

import importlib.metadata
import os
import platform
import sys

import dlomix
from dlomix.config import _BACKEND, PYTORCH_BACKEND

PACKAGES = ["numpy", "keras", "tensorflow", "torch", "datasets", "pyarrow"]


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _tensorflow_devices():
    import tensorflow as tf

    build = tf.sysconfig.get_build_info()
    if build.get("is_cuda_build"):
        lines = [
            f"TensorFlow built with CUDA {build.get('cuda_version', '?')}, "
            f"cuDNN {build.get('cudnn_version', '?')}"
        ]
    else:
        lines = ["TensorFlow built without CUDA"]

    gpus = tf.config.list_physical_devices("GPU")
    lines.append(f"GPUs visible to TensorFlow: {len(gpus)}")
    lines += [f"  {gpu.name}" for gpu in gpus]

    if not gpus and platform.system() == "Linux":
        lines.append(
            'hint: for NVIDIA GPUs install `pip install "dlomix[tf-cuda]"`, which '
            "provides the CUDA libraries, and check that `nvidia-smi` shows a driver"
        )
    return lines


def _torch_devices():
    import torch

    if torch.version.cuda:
        lines = [f"PyTorch built with CUDA {torch.version.cuda}"]
    else:
        lines = ["PyTorch built without CUDA"]

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        lines.append(f"CUDA GPUs visible to PyTorch: {count}")
        lines += [f"  cuda:{i} {torch.cuda.get_device_name(i)}" for i in range(count)]
    else:
        lines.append("CUDA GPUs visible to PyTorch: 0")
        if torch.version.cuda and platform.system() == "Linux":
            lines.append(
                "hint: this build needs an NVIDIA driver new enough for CUDA "
                f"{torch.version.cuda}; check `nvidia-smi`"
            )

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        lines.append("Apple GPU (MPS) available: yes")
    return lines


def main():
    rows = [
        ("dlomix", dlomix.__version__),
        ("python", sys.version.split()[0]),
        ("platform", f"{platform.system()} {platform.machine()}"),
        ("backend", _BACKEND),
        # set by dlomix.config before keras is imported, so it is always accurate
        ("keras backend", os.environ.get("KERAS_BACKEND", "?")),
    ]
    rows += [(package, _version(package)) for package in PACKAGES]

    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")

    print()
    try:
        devices = (
            _torch_devices() if _BACKEND in PYTORCH_BACKEND else _tensorflow_devices()
        )
    except Exception as error:  # a diagnostic must report, never crash
        devices = [f"could not query devices: {type(error).__name__}: {error}"]
    print("\n".join(devices))


if __name__ == "__main__":
    main()
