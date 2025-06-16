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

tensorflow_extra_install = [
    "tensorflow>=2.13,<2.16",  # 2.16 introduces breaking changes and has Keras 3 as default
]

pytorch_extra_install = [
    "torch",
    "torchvision",
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
    package_data={"": ["data/processing/pickled_feature_dicts/*"]},
    install_requires=[
        "datasets",
        "huggingface_hub",
        "fpdf",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pyarrow",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.9",
            "pytest-cov",
            "black",
            "twine",
            "setuptools",
            "wheel",
            "pylint",
            *tensorflow_extra_install,
            *pytorch_extra_install,
        ],
        "wandb": [
            "wandb >= 0.15",
        ],
        "tensorflow": tensorflow_extra_install,
        "tf": tensorflow_extra_install,
        "torch": pytorch_extra_install,
        "pytorch": pytorch_extra_install,
        "lightning": [
            "lightning",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
    ],
)
