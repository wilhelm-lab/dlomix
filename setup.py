import platform

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from dlomix import META_DATA, __version__

VERSION = __version__
tensorflow_version = "2.10.0"
tensorflow_version = "2.10.0"

os_name = platform.system().lower()

if os_name == "darwin":
    # Apple silicon
    tensorflow_requirement = "tensorflow-macos"
else:
    tensorflow_requirement = "tensorflow"

tensorflow_requirement = tensorflow_requirement + " == " + tensorflow_version

requirements = [
    "fpdf",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    # we install with the extra xml to ensure lxml is installed
    # more details about extras for pyteomics are here: https://pyteomics.readthedocs.io/en/latest/installation.html
    "pyteomics[XML]",
    "scikit-learn",
    "seaborn",
    tensorflow_requirement,
    "prospect-dataset @ git+https://github.com/wilhelm-lab/PROSPECT.git@develop",
]

dev_requirements = [
    "black",
    "pylint",
    "pytest >= 3.7",
    "pytest-cov",
    "setuptools",
    "twine",
    "wheel",
]

setuptools.setup(
    name=META_DATA["package_name"].lower(),
    version=VERSION,
    author=META_DATA["author"],
    author_email=META_DATA["author_email"],
    description=META_DATA["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=META_DATA["github_url"],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
    ],
)
