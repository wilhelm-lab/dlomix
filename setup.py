import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='dlpro',
     version='0.1',
     author="Omar Shouman",
     author_email="omar.shouman@gmail.com",
     description="Deep Learning in Proteomics",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/omsh/dlpro",
     packages=setuptools.find_packages(),
    install_requires=[
          'pandas', 'numpy', 'tensorflow'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )