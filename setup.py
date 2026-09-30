import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_metadata():
    metadata = {}
    with open("src/dlomix/_metadata.py") as f:
        exec(f.read(), metadata)
    return metadata


# Load metadata
META_DATA = get_metadata()

# Upper bound = the newest TensorFlow CI actually installs and tests. Raise it
# deliberately after CI passes on the new release, rather than letting users
# silently pick up an untested minor version.
# The floor: numpy>=2.0 (a core requirement) rules out TF 2.16/2.17, which pin numpy<2.
TENSORFLOW_VERSION = ">=2.18,<2.22"
# macOS: TF 2.20 collides with pyarrow, so stay below it there.
TENSORFLOW_VERSION_MACOS = ">=2.18,<2.20"

# Platform differences are expressed as environment markers, evaluated on the
# installing machine. A build-time platform check would be baked into the
# py3-none-any wheel for every platform.
tensorflow_extra_install = [
    f"tensorflow{TENSORFLOW_VERSION_MACOS}; platform_system == 'Darwin'",
    f"tensorflow{TENSORFLOW_VERSION}; platform_system != 'Darwin'",
]

# CUDA-enabled TensorFlow for Linux GPU machines. tensorflow[and-cuda] installs
# CUDA/cuDNN as pip wheels, which NVIDIA publishes only for Linux; TF's own extra
# carries no platform markers, so ungated it fails to resolve on macOS/Windows.
# Gated here, the extra degrades to the regular build on other platforms.
# Deliberately not part of `dev`: CI has no GPU and should not download CUDA.
tensorflow_cuda_extra_install = [
    f"tensorflow[and-cuda]{TENSORFLOW_VERSION}; platform_system == 'Linux'",
    f"tensorflow{TENSORFLOW_VERSION_MACOS}; platform_system == 'Darwin'",
    f"tensorflow{TENSORFLOW_VERSION}; "
    "platform_system != 'Linux' and platform_system != 'Darwin'",
]

# PyTorch needs no GPU extra: its Linux wheels on PyPI are already CUDA builds.
pytorch_extra_install = [
    # torch does not declare numpy as a dependency; builds before 2.3 were compiled
    # against NumPy 1.x and fail at import under NumPy 2.
    "torch>=2.3",
]

setuptools.setup(
    name=META_DATA["__package__"].lower(),
    version=META_DATA["__version__"],
    author=META_DATA["__author__"],
    author_email=META_DATA["__author_email__"],
    description=META_DATA["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=META_DATA["__github_url__"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"": ["data/processing/feature_dicts/*"]},
    python_requires=">=3.11",
    install_requires=[
        "datasets>=4.0.0",
        "huggingface_hub>=0.20.0",
        # Keras 3 is backend-agnostic and pure Python: it provides the single
        # `keras.ops` implementation of the losses and metrics shared by both
        # backends, and does not pull in TensorFlow when KERAS_BACKEND=torch.
        "keras>=3.0.0",
        "fpdf",
        "pandas",
        "numpy>=2.0",
        "matplotlib",
        "scikit-learn",
        "pyarrow",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest >= 7.0.0",
            "pytest-cov",
            "black",
            "isort",  # invoked by `make format` / `make format-check`
            "twine",
            "setuptools",
            "wheel",
            "pylint",
            *tensorflow_extra_install,
            *pytorch_extra_install,
        ],
        "wandb": [
            "wandb>=0.20.0",
        ],
        "tensorflow": tensorflow_extra_install,
        "tf": tensorflow_extra_install,
        "tensorflow-cuda": tensorflow_cuda_extra_install,
        "tf-cuda": tensorflow_cuda_extra_install,
        "torch": pytorch_extra_install,
        "pytorch": pytorch_extra_install,
        "lightning": [
            "lightning",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
)
