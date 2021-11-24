import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='dlomix',
    version='0.0.1',
    author="Omar Shouman",
    author_email="o.shouman@tum.de",
    description="Deep Learning for Proteomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wilhelm-lab/dlomix",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tensorflow'],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            "twine",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"
    ],
)
