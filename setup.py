import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from dlomix import META_DATA, __version__

VERSION = __version__

setuptools.setup(
    name=META_DATA['package_name'].lower(),
    version=VERSION,
    author=META_DATA['author'],
    author_email=META_DATA['author_email'],
    description=META_DATA['description'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=META_DATA['github_url'],
    packages=setuptools.find_packages(),
    install_requires=[
        'fpdf',
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'pyarrow',
        'seaborn',
        ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            "pytest-cov",
            "black",
            "twine",
            "setuptools",
            "wheel",
            "pylint",
        ],
        "wandb": [
            "wandb >= 0.15",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"
    ],
)
