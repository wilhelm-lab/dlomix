import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='mlomix',
     version='0.1',
     author="Omar Shouman",
     author_email="omar.shouman@tum.de",
     description="Deep Learning for Proteomics",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/omsh/mlomix",
     packages=setuptools.find_packages(),
    install_requires=[
          'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'tensorflow'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )