
Datasets Guide
**************************

This guide provides an introduction to working with DLOmix dataset classes for various proteomics machine learning tasks.

.. contents:: Table of Contents
   :local:
   :depth: 1

Overview
========

Introduction to PeptideDataset
-------------------------------

``PeptideDataset`` is the foundational class for handling peptide data in DLOmix. It wraps HuggingFace's ``Dataset`` library and provides specialized preprocessing and other modules for proteomics tasks including:

* Loading data from various formats and sources with configurable parameters and sensible defaults (Hugging Face datasets on the Hugging Face Hub or in-memory, Parquet files, CSV files, etc..)
* Sequence parsing and encoding, including handling post-translational modifications (PTMs)
* Data splitting for model training and evaluation (Train/val/test)
* Feature extraction from the peptide sequences and/or modifications present in the sequence
* Automatic conversion to TensorFlow or PyTorch tensors
* Efficient batch processing with caching support (relies on Hugging Face datasets cache)
* Custom saving and loading the processed dataset object for faster experimentation and re-runs
* Improved reproducibility via logging saving/loading configurations and meta-data of the processed dataset class

Available Task-Specific Datasets
---------------------------------

DLOmix provides task-specific dataset classes that inherit from ``PeptideDataset`` with appropriate default values that work with the `PROSPECT dataset collection`_ on the HF hub.

.. _PROSPECT dataset collection: https://huggingface.co/collections/Wilhelmlab/prospect-ptms

* **RetentionTimeDataset** - Predicts peptide retention time with default label ``indexed_retention_time``
* **ChargeStateDataset** - Predicts charge state with default label ``most_abundant_charge_by_count``
* **DetectabilityDataset** - Predicts peptide detectability with default label ``Classes``
* **FragmentIonIntensityDataset** - Predicts fragment ion intensities with default label ``intensities_raw``

All classes share the same API but differ in their default parameters for common use cases, which are sensible for the PROSPECT datasets to provide a working example when used with the datasets hosted on the Hugging Face Hub.


Basic Usage
===========

Loading Data from Files
------------------------

Load datasets from local files (CSV, Parquet, etc.):

.. code-block:: python

   from dlomix.data import RetentionTimeDataset

   # Load from a single CSV file (auto-splits train/val with the provided ratio)
   dataset = RetentionTimeDataset(
       data_source="data/peptides.csv",
       data_format="csv",
       sequence_column="sequence",
       label_column="retention_time",
       val_ratio=0.2
   )

   # Load with explicit train/val/test splits
   dataset = RetentionTimeDataset(
       data_source="data/train.parquet",
       val_data_source="data/val.parquet",
       test_data_source="data/test.parquet",
       data_format="parquet"
   )

Loading Data from HuggingFace Hub
----------------------------------

Load directly from HuggingFace Hub datasets (hosted on the hub):

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="wilhelmlab/prospect-rt-dataset", # example dataset provided by Wilhelmlab, TU Munich
       data_format="hub",
       # column names match the provided example, replace if needed
       sequence_column="modified_sequence",
       label_column="indexed_retention_time"
   )

.. note::
   When using ``data_format="hub"``, ``val_data_source`` and ``test_data_source`` are ignored. The dataset must contain pre-defined splits.

Using In-Memory HuggingFace Datasets
-------------------------------------

Local data could be in other formats (e.g. pandas) and require some specific wrangling or formatting that the user would want to run before feeding the data into DLOmix and training a model. For this specific flow, the dataset class offers a way to feed in an in-memory hugging face dataset.

.. code-block:: python

   from datasets import load_dataset

   # ... user code, column renaming, wrangling, etc..

   # simulate an in-memory hugging face dataset
   hf_dataset = load_dataset("csv", data_files="data.csv")

   # Pass to DLOmix
   dataset = RetentionTimeDataset(
       data_source=hf_dataset,
       data_format="hf", # important to ensure that data is read and parsed correctly
       sequence_column="sequence",
       label_column="rt"
   )


Core Concepts
=============

Data Splits and Validation
---------------------------

DLOmix supports three splitting strategies:

1. **Single source with auto-split**: Set ``val_ratio`` to automatically split training data into train/val randomly
2. **Multiple sources**: Provide separate files for train/val/test
3. **Pre-split datasets**: Use HuggingFace Hub or ``DatasetDict`` with existing splits

.. code-block:: python

   # Strategy 1: Auto-split
   dataset = RetentionTimeDataset(
       data_source="train.csv",
       val_ratio=0.2,  # 20% for validation
       data_format="csv"
   )

   # Strategy 2: Separate files
   dataset = RetentionTimeDataset(
       data_source="train.csv",
       val_data_source="val.csv",
       test_data_source="test.csv",
       data_format="csv"
   )

   # Strategy 3: pre-split dataset (example below points to a remote hugging face dataset hosted on the hub)
   dataset = RetentionTimeDataset(
       data_source="wilhelmlab/prospect-rt",
       data_format="hub"
   )


Sequence Processing Pipeline
-----------------------------

Datasets automatically process sequences through:

1. **Parsing**: Extract sequences with PTMs (e.g., ``M[UNIMOD:35]``), where PTM representation follows `Unimod_`'s convention.
2. **Encoding**: Convert amino acids to integers using the alphabet (with options to learn the alphabet from the data)
3. **Padding**: Pad to ``max_seq_len`` (default uses ``padding_value="-"``, which is encoded as `0`'s by default')
4. **Feature Extraction**: Add computed features from the sequence and/or the PTM information

.. _Unimod: https://unimod.org

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       max_seq_len=50,              # Pad/truncate to length 50
       pad=True,                     # Enable padding
       padding_value="-",           # Use '-' for padding
       with_termini=True,           # Add N/C termini markers, []- and -[], even if there are no terminal modifications present

       # see below and data/processing/feature_extractions.py for more details
       features_to_extract=["delta_mass", "mod_gain"]
   )

Feature Extraction
------------------

Built-in feature extractors add computed features to your dataset that are converted to tensors and can be fed into your model:

There are two options to use feature extractors; (1) dlomix built-in feature extractors as string arguments and (2) custom feature extractors passed as python function objects.

Available features in the framework as lookup python dicts are:

* ``mod_loss``
* ``delta_mass``
* ``mod_gain``
* ``atom_count``
* ``red_smiles``

Custom feature extractors can either:
    (1) use the `FeatureExtractor` class or
    (2) write a function that can be mapped (`dataset.map()`) to the Hugging Face dataset.

In both cases, you can access the parsed sequence information from the dataset using the following keys, which all provide python lists:
    - `_parsed_sequence`: parsed sequence
    - `_n_term_mods`: N-terminal modifications
    - `_c_term_mods`: C-terminal modifications

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       features_to_extract=["mod_loss", "atom_count"], # extracted feautres from the sequence and modifications
       model_features=["collision_energy"], # other features present already in the dataset columns
   )



Encoding Schemes and Alphabets
------------------------------

Sequences are parsed and are integer encoded to be fed into sequence models (specifically to the embedding layers). Two important parameters control the parsing and encoding: (1) the encoding scheme, and (2) the alphabet or vocabulary used.

Two primary encoding schemes for sequences are available:

* **UNMOD**: Assumes the sequences do not contain modifications, hence any [UNIMOD] strings are removed.
* **NAIVE_MODS**: Assumes sequences contain modifications in UNIMOD format (e.g., ``M[UNIMOD:35]``) and encodes them as distinct tokens; separate token from the amino acid.

The alphabet is a python dict that maps each character (amino acid or amino acid + PTM combination) to a unique integer index. It can either be learnt from the provided data implicitily or provided by the user.

The user can:
- use built-in alphabets from ``dlomix.constants``
- or define custom alphabets as needed and pass them as a python dict to the alphabet argument,
- or provide no alphabet or `None` as the alphabet to trigger learning the alphabet from the data based on the selected encoding scheme.


Note that if an alphabet is provided, the user has to ensure that it covers all the amino acids (or amino acid + PTM combinations) present in the data.

1. Use built-in alphabets from ``dlomix.constants``

.. code-block:: python

   from dlomix.data import RetentionTimeDataset
   from dlomix.constants import ALPHABET_UNMOD, ALPHABET_NAIVE_MODS

   # Unmodified sequences, uses built-in unmodified alphabet
   dataset = RetentionTimeDataset(
       data_source="data.csv",
       encoding_scheme="unmod",
       alphabet=ALPHABET_UNMOD
   )

   # With PTMs, uses built-in naive-mods alphabet with tokens for some amino acids + PTMs combinations
   dataset = RetentionTimeDataset(
       data_source="data.csv",
       encoding_scheme="naive-mods",
       alphabet=ALPHABET_NAIVE_MODS
   )

2. Define and use a custom alphabet

The dataset class uses the provided alphabet for encoding sequences, so it must cover all characters present in the data, else the unknown tokens are all encoded as unknown with the same integer index.

.. code-block:: python

   # Custom alphabet with special amino acids and PTMs
   CUSTOM_ALPHABET = {
       # ....
   }

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       encoding_scheme="naive-mods",
       alphabet=CUSTOM_ALPHABET
   )


3. Learn alphabet from data

If no alphabet is provided (i.e., ``alphabet=None``), the dataset class learns the alphabet from the data based on the selected encoding scheme.
If a pre-defined split is provided, the alphabet is learned from the training and validation data only, the test set is not used for learning the alphabet to allow for proper evaluation on unseen data.

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       encoding_scheme="naive-mods",
       alphabet=None  # Learn from data
   )

   # Access the learned alphabet after the class is initialized and the processing is done
   print(dataset.extended_alphabet)



Tensor Datasets for Model Training
==================================

The tensor datasets can be accessed via the ``train_data``, ``val_data``, and ``test_data`` attributes of the dataset class. They are ready to be fed into TensorFlow or PyTorch models depending on the selected ``dataset_type``.

.. code-block:: python

   from dlomix.data import RetentionTimeDataset

   # Load and process dataset
   dataset = RetentionTimeDataset(...)

   # model initialization, compilation, etc..

   # pass to model.fit() in Keras
   model.fit(dataset.train_data,
            validation_data=dataset.val_data,
            epochs=10,
            **kwargs)


Advanced Features
=================

Custom Feature Extraction
--------------------------

Provide custom feature extraction functions:

.. code-block:: python

   # define the custom feature extraction function
   def hydrophobicity_score(input_data):
   """Calculate hydrophobicity score"""

        # lookup table for hydrophobicity values
        hydro_values = {'A': 1.8, 'R': -4.5, 'N': -3.5, ...}

        # access the parsed sequence or any other column in the dataset
        sequence = input_data["_parsed_sequence"]

        # add the new column with the feature name
        input_data["hydrophobicity_score"] = sum(hydro_values.get(aa, 0) for aa in sequence)

        # return the whole row with the new feature added
        return input_data

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       features_to_extract=[hydrophobicity_score],
       model_features=["collision_energy"]
   )

Custom functions receive the row as a dictionary and should return the row again after adding the feature.

Multi-Processing and Performance
---------------------------------

Optimize dataset processing with these parameters:

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="large_dataset.parquet",
       num_proc=4,                    # Use 4 CPU cores for processing
       batch_processing_size=5000,    # Process 5000 rows at a time
       disable_cache=False,           # Enable HF datasets caching
       auto_cleanup_cache=True,       # Clean temp files after processing
       enable_tf_dataset_cache=True   # Cache TF datasets in memory
   )

**Performance tips:**

* ``num_proc``: Set to number of CPU cores for large datasets
* ``batch_processing_size``: Increase for better throughput (default: 1000)
* ``enable_tf_dataset_cache``: Speeds up repeated iterations but uses more memory
* ``disable_cache=False``: Reuses processed datasets across runs. This is the hugging face datasets caching mechanism.
* ``auto_cleanup_cache=True``: Cleans temporary files after processing to save disk space.

Saving and Loading Processed Datasets
--------------------------------------

Save processed datasets to disk to avoid reprocessing:

.. code-block:: python

   # Save processed dataset
   dataset = RetentionTimeDataset(
       data_source="train.csv",
       val_ratio=0.2
   )
   dataset.save_to_disk("processed_datasets/rt_dataset")

   # Load later
   from dlomix.data import load_processed_dataset

   # loads the dataset along with its configuration and metadata, no re-processing needed
   dataset = load_processed_dataset("processed_datasets/rt_dataset")

   # Access tensor data immediately
   train_data = dataset.train_data

This saves configuration, processed HuggingFace datasets, and metadata.


TensorFlow vs PyTorch
=====================

Generating TensorFlow Datasets
-------------------------------

Default behavior returns ``tf.data.Dataset`` objects:

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       dataset_type="tf",  # Default
       batch_size=64
   )

   # Returns batched tf.data.Dataset
   train_data = dataset.train_data
   val_data = dataset.val_data

   # Use directly with Keras
   model.fit(train_data, validation_data=val_data, epochs=10)

Generating PyTorch Datasets
----------------------------

Use ``dataset_type="pt"`` for PyTorch DataLoaders. Since PyTorch DataLoaders must be created to provide tensors, you can pass additional DataLoader arguments via ``torch_dataloader_kwargs``.

.. code-block:: python

   dataset = RetentionTimeDataset(
       data_source="data.csv",
       dataset_type="pt",
       batch_size=64,
       sequence_column="sequence",
       label_column="retention_time",
       torch_dataloader_kwargs={
           "shuffle": True,
           # other DataLoader args
       }
   )

   # Returns PyTorch DataLoader
   train_loader = dataset.train_data
   val_loader = dataset.val_data

   # Training loop
   for batch, label in train_loader:
       sequences = batch["sequence"]
       # ... training code


Configuration Reference
=======================

DatasetConfig Parameters
-------------------------

**Data Sources**

* ``data_source``: Path/URL to training data or HF dataset
* ``val_data_source``: Path to validation data (optional)
* ``test_data_source``: Path to test data (optional)
* ``data_format``: Format: ``"csv"``, ``"parquet"``, ``"hub"``, ``"hf"``

**Columns**

* ``sequence_column``: Column name containing peptide sequences
* ``label_column``: Column(s) name(s) containing label(s) (str or list)
* ``dataset_columns_to_keep``: Additional columns to retain in the dataset after processing, else extra columns are dropped to save memory.

**Processing**

* ``max_seq_len``: Maximum sequence length for padding
* ``pad``: Enable padding (default: True)
* ``padding_value``: Character for padding (default: ``"-"``)
* ``with_termini``: Add N/C termini markers (default: True)
* ``encoding_scheme``: ``"unmod"`` or ``"naive-mods"``
* ``alphabet``: Dict mapping tokens to integers

**Features**

* ``features_to_extract``: List of feature names or functions
* ``model_features``: Features to include in tensor output that already exist in the provided data and are to be carried over as tensors to be fed into the model.

**Training**

* ``val_ratio``: Validation split ratio (0-1)
* ``batch_size``: Batch size for tensor datasets
* ``dataset_type``: ``"tf"`` or ``"pt"``
* ``shuffle``: Shuffle data (default: False)


Best Practices
==============

**Caching Strategy**

* Enable HF caching (``disable_cache=False``) for repeated experiments
* Save processed datasets with ``save_to_disk()`` to save time when iterating over the same data
* Use ``enable_tf_dataset_cache=True`` only if dataset fits in the available memory

**Batch Size Selection**

* Start with meaningful defaults if you have limited GPU memory (e.g., 64 or 128)
* Increase based on GPU memory availability and utilization
* Reduce if encountering OOM (Out-of-Memory) errors

**Validation Splits**

* Prefer explicit ``val_data_source`` for consistent evaluation
* Always use a separate test dataset for final evaluation, can also be created independently using another Dataset instance with ``test_data_source`` only.

**Feature Engineering**

* List all features in ``model_features`` to include them in tensors
* Use custom extractors for domain-specific features

**Performance**

* Set ``num_proc`` to match available CPU cores
* Use Parquet format for large datasets
* Process data once and save with ``save_to_disk()``
