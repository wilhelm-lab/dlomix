import logging
import warnings
from typing import Optional, Union

from datasets import Dataset, DatasetDict

from .dataset_config import DatasetConfig
from .dataset_splitter import SplitConfig, create_splitter
from .dataset_utils import EncodingScheme, get_num_processors, resolve_num_proc
from .loading import DataSourceLoader, _DatasetSplitMode
from .processing.pipeline import PipelineContext, ProcessingPipeline
from .serialization import save_dataset
from .tensor_conversion import (
    cast_feature_columns_to_float,
    to_tf_tensor_dataset,
    to_torch_dataloader,
)

logger = logging.getLogger(__name__)


class PeptideDataset:
    """
    PeptideDataset class to handle peptide datasets for deep learning models.
    The class is designed to handle peptide datasets in various formats and process them into a format that can be used by deep learning models.
    The class is built on top of the Hugging Face datasets library and provides a simple interface to load, process and save peptide datasets.

    Parameters
    ----------
    data_source : Union[str, List]
        Path to the data source file or list of paths to the data source files.
    val_data_source : Union[str, List]
        Path to the validation data source file or list of paths to the validation data source files.
    test_data_source : Union[str, List]
        Path to the test data source file or list of paths to the test data source files.
    data_format : str
        Format of the data source file(s). Example formats are 'csv', 'json', 'parquet', etc. Use 'hub' for datasets from the Hugging Face Hub and 'hf' for in-memory HF Dataset/DatasetDict objects.
    sequence_column : str
        Name of the column in the data source file that contains the peptide sequences.
    label_column : Union[str, List]
        Name of the column(s) in the data source file that contains the labels.
    val_ratio : float
        Ratio of the validation data to the training data. The value should be between 0 and 1.
    max_seq_len : int
        Maximum sequence length to pad the sequences to. If set to 0, the sequences will not be padded.
    dataset_type : str
        Type of the tensor dataset to be generated afterwards. Possible values are "tf" and "pt" for TensorFlow and PyTorch, respectively. Fallback is to TensorFlow dataset tensors.
    batch_size : int
        Batch size for the tensor dataset.
    model_features : List[str]
        List of column names in the data source file that contain features to be used by the model.
    dataset_columns_to_keep : Optional[List[str]]
        List of column names in the data source file that should be kept in the Hugging Face dataset but not returned as tensors.
    features_to_extract : Optional[List[Union[Callable, str]]]
        List of feature extractors to be applied to the sequences. The feature extractors can be either a function or a string that corresponds to a predefined feature extractor.
    pad : bool
        Flag to indicate whether to pad the sequences to the maximum sequence length.
    padding_value : str
        Value to use for padding the sequences.
    alphabet : Dict
        Alphabet to use for encoding the amino acids in the sequences.
    with_termini : bool
        Flag to indicate whether to include the N- and C-termini []- and -[] in the sequences.
    encoding_scheme : Union[str, EncodingScheme]
        Encoding scheme to use for encoding the sequences. Possible values are "unmod" and "naive-mods" for unmodified sequences and sequences with PTMs respectively.
    processed : bool
        Flag to indicate whether the dataset has been processed or not.
    enable_tf_dataset_cache : bool
        Flag to indicate whether to enable TensorFlow Dataset caching (call `.cache()` on the generated TF Datasets).
    disable_cache : bool
        Flag to indicate whether to disable Hugging Face Datasets caching. Default is False.
    auto_cleanup_cache : bool
        Flag to indicate whether to automatically clean up the temporary Hugging Face Datasets cache files. Default is True.
    num_proc : Optional[int]
        Number of processes to use for processing the dataset.
        Set to ``-1`` to use all available processors, ``None`` to force single-process execution,
        or a positive integer to use an explicit number of processors.
    batch_processing_size : Optional[int]
        Batch size for processing the dataset, passed to the HuggingFace `Dataset.map()` function calls. Default is 1000.

    Attributes
    ----------
    DEFAULT_SPLIT_NAMES : List[str]
        Default split names for the dataset.
    CONFIG_JSON_NAME : str
        Name of the configuration JSON file.

    Methods
    -------
    save_to_disk(path: str)
        Save the dataset to disk.
    load_from_disk(path: str)
        Load the dataset from disk.
    from_dataset_config(config: DatasetConfig)
        Create a PeptideDataset object from a DatasetConfig object.
    """

    DEFAULT_SPLIT_NAMES = ["train", "val", "test"]
    CONFIG_JSON_NAME = "dlomix_peptide_dataset_config.json"
    METADATA_JSON_NAME = "dlomix_peptide_dataset_metadata.json"
    SERIALIZATION_VERSION = "0.1"
    PADDING_VALUE_DEFAULT_INDEX = 0

    def __init__(self, dataset_config: DatasetConfig, **kwargs):
        super(PeptideDataset, self).__init__()
        self.__dict__.update(**dataset_config.__dict__)
        self._kwargs = kwargs

        # to be kept in the hf dataset, but not returned in the tensor dataset
        if dataset_config.dataset_columns_to_keep is None:
            self.dataset_columns_to_keep = []
        else:
            self.dataset_columns_to_keep = dataset_config.dataset_columns_to_keep

        self.encoding_scheme = EncodingScheme(dataset_config.encoding_scheme)

        self._set_hf_cache_management()

        self.extended_alphabet = {}
        self.learning_alphabet_mode = True

        if self.alphabet:
            self.extended_alphabet = self.alphabet.copy()
            self.learning_alphabet_mode = False

        # add padding value to the alphabet if not present
        if self.extended_alphabet.get(self.padding_value) is None:
            self.extended_alphabet[self.padding_value] = (
                PeptideDataset.PADDING_VALUE_DEFAULT_INDEX
            )

        self._config = dataset_config

        # explcit assignments of processed attribute
        self.processed = dataset_config.processed
        if not self.processed:
            self._num_proc = dataset_config.num_proc
            self._set_num_proc()

            load_result = DataSourceLoader(dataset_config, self._kwargs).load()
            self.hf_dataset: Optional[Union[Dataset, DatasetDict]] = (
                load_result.hf_dataset
            )
            self._data_files_available_splits = load_result.available_splits
            self._empty_dataset_mode = load_result.empty
            self._split_mode: Optional[_DatasetSplitMode] = load_result.split_mode

            self._relevant_columns = []
            self._extracted_features_columns = []

            if not self._empty_dataset_mode:
                self._remove_unnecessary_columns()
                self._split_dataset()
                self._run_processing_pipeline()
                if (
                    self.model_features is not None
                    or len(self._extracted_features_columns) > 0
                ):
                    self._cast_model_feature_types_to_float()
                self._cleanup_temp_dataset_cache_files()
                self.processed = True

    def _set_num_proc(self):
        n_processors = get_num_processors()
        self._num_proc, was_capped = resolve_num_proc(self._num_proc, n_processors)

        if was_capped:
            warnings.warn(
                f"Number of processors provided is greater than the available processors. Using the maximum number of processors available: {n_processors}."
            )

    def _set_hf_cache_management(self):
        if self.disable_cache:
            from datasets import disable_caching

            disable_caching()

    def _remove_unnecessary_columns(self):
        self._relevant_columns = [self.sequence_column, *self.label_column]

        if self.model_features is not None:
            self._relevant_columns.extend(self.model_features)

        if self.dataset_columns_to_keep is not None:
            # additional columns to keep in the hugging face dataset only and not return as tensors
            self._relevant_columns.extend(self.dataset_columns_to_keep)

        # Preserve stratify_by_column through splitting; track if it needs removal afterward
        self._temp_stratify_column: Optional[str] = None
        if (
            self.stratify_by_column is not None
            and self.stratify_by_column not in self._relevant_columns
        ):
            self._relevant_columns.append(self.stratify_by_column)
            self._temp_stratify_column = self.stratify_by_column

        # select only relevant columns from the Hugging Face Dataset (includes label column)
        self.hf_dataset = self.hf_dataset.select_columns(self._relevant_columns)

    def _split_dataset(self):
        if self._split_mode != _DatasetSplitMode.AUTO:
            return

        assert isinstance(self.hf_dataset, DatasetDict)

        stratify_column = self.stratify_by_column

        # Require stratify_by_column for stratified splitting (dataset-facing name;
        # SplitConfig would otherwise raise using its own 'stratify_column' parameter).
        if (
            self.split_strategy
            and self.split_strategy.lower() == "stratified"
            and stratify_column is None
        ):
            raise ValueError(
                "stratify_by_column must be provided when split_strategy='stratified'."
            )

        # Validate stratify_by_column exists in the data if provided
        if stratify_column is not None:
            train_columns = self.hf_dataset[
                PeptideDataset.DEFAULT_SPLIT_NAMES[0]
            ].column_names
            if (
                stratify_column not in self.label_column
                and stratify_column not in train_columns
            ):
                raise ValueError(
                    f"Stratification column '{stratify_column}' not found in dataset. "
                    f"Available columns: {train_columns}"
                )

        split_config = SplitConfig(
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            strategy=self.split_strategy if self.split_strategy else "random",
            seed=self.split_seed,
            stratify_column=stratify_column,
            sequence_column=self.sequence_column,
        )

        splitter = create_splitter(split_config)
        self.hf_dataset = splitter.split(
            self.hf_dataset[PeptideDataset.DEFAULT_SPLIT_NAMES[0]]
        )

        # Drop the stratify column from all splits if it was only kept temporarily
        if self._temp_stratify_column is not None:
            self.hf_dataset = self.hf_dataset.remove_columns(self._temp_stratify_column)
            self._relevant_columns.remove(self._temp_stratify_column)
            self._temp_stratify_column = None

    def _run_processing_pipeline(self):
        pipeline = ProcessingPipeline.from_config(
            sequence_column=self.sequence_column,
            encoding_scheme=self.encoding_scheme,
            with_termini=self.with_termini,
            max_seq_len=self.max_seq_len,
            padding_value=self.padding_value,
            alphabet=self.extended_alphabet,
            learning_alphabet_mode=self.learning_alphabet_mode,
            pad=self.pad,
            features_to_extract=self.features_to_extract,
            split_names=PeptideDataset.DEFAULT_SPLIT_NAMES,
        )

        # parsed columns are kept in the HF dataset but not returned as tensors
        self.dataset_columns_to_keep.extend(pipeline.parsed_columns)
        self._extracted_features_columns = pipeline.extracted_feature_names

        ctx = PipelineContext(
            alphabet=self.extended_alphabet,
            num_proc=self._num_proc,
            batch_size=self.batch_processing_size,
            fit_splits=tuple(PeptideDataset.DEFAULT_SPLIT_NAMES[0:2]),
        )
        self.hf_dataset = pipeline.apply(self.hf_dataset, ctx)
        # the encoding processor learns/extends the alphabet during the run
        self.extended_alphabet = ctx.alphabet

    def _cast_model_feature_types_to_float(self):
        features_to_cast = set().union(
            self.model_features or [],
            self._extracted_features_columns or [],
        )

        for split in self.hf_dataset.keys():
            self.hf_dataset[split] = cast_feature_columns_to_float(
                self.hf_dataset[split],
                features_to_cast,
                num_proc=self._num_proc,
                batch_size=self.batch_processing_size,
            )

    def _cleanup_temp_dataset_cache_files(self):
        if self.auto_cleanup_cache:
            cleaned_up = self.hf_dataset.cleanup_cache_files()
            logger.info("Cleaned up cache files: %s.", cleaned_up)

    def save_to_disk(self, path: str, overwrite: bool = False) -> bool:
        """Save the dataset (config, runtime state, and HF data) to ``path``."""
        return save_dataset(self, path, overwrite)

    @classmethod
    def from_dataset_config(cls, config: DatasetConfig):
        return cls(**config.__dict__)

    def __getitem__(self, index):
        return self.hf_dataset[index]

    def __str__(self):
        return self.hf_dataset.__str__()

    def __repr__(self):
        return self.hf_dataset.__repr__()

    def __len__(self):
        return len(self.hf_dataset)

    def __iter__(self):
        return iter(self.hf_dataset)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.hf_dataset, attr)
        else:
            return self.__dict__[attr]

    def _get_input_tensor_column_names(self):
        # return a list of columns to be used as input tensors
        input_tensor_columns = self._relevant_columns.copy()

        # remove the label column(s) from the input tensor columns since the to_tf_dataset method has a separate label_cols argument
        for label in self.label_column:
            input_tensor_columns.remove(label)

        # remove the columns that are not needed in the tensor dataset
        input_tensor_columns = list(
            set(input_tensor_columns) - set(self.dataset_columns_to_keep)
        )

        # add the extracted features columns to the input tensor columns
        input_tensor_columns.extend(self._extracted_features_columns)

        return input_tensor_columns

    @property
    def tensor_train_data(self):
        """TensorFlow or Torch Dataset object for the training data"""
        if self.dataset_type == "pt":
            return self._get_split_torch_dataset(PeptideDataset.DEFAULT_SPLIT_NAMES[0])
        else:
            tf_dataset = self._get_split_tf_dataset(
                PeptideDataset.DEFAULT_SPLIT_NAMES[0]
            )

            if self.enable_tf_dataset_cache:
                tf_dataset = tf_dataset.cache()

            return tf_dataset

    @property
    def tensor_val_data(self):
        """TensorFlow or Torch Dataset object for the val data"""
        if self.dataset_type == "pt":
            return self._get_split_torch_dataset(PeptideDataset.DEFAULT_SPLIT_NAMES[1])
        else:
            tf_dataset = self._get_split_tf_dataset(
                PeptideDataset.DEFAULT_SPLIT_NAMES[1]
            )

            if self.enable_tf_dataset_cache:
                tf_dataset = tf_dataset.cache()

            return tf_dataset

    @property
    def tensor_test_data(self):
        """TensorFlow or Torch Dataset object for the test data"""
        if self.dataset_type == "pt":
            return self._get_split_torch_dataset(PeptideDataset.DEFAULT_SPLIT_NAMES[2])
        else:
            tf_dataset = self._get_split_tf_dataset(
                PeptideDataset.DEFAULT_SPLIT_NAMES[2]
            )

            return tf_dataset

    def get_preprocessor(self):
        """Return a :class:`PeptidePreprocessor` capturing this dataset's recipe.

        The preprocessor reproduces the exact training-time preprocessing (alphabet,
        encoding, padding, feature extractors) and can be applied to raw inputs for
        inference, or bundled with a model. See :mod:`dlomix.data.inference`.
        """
        from .inference import PeptidePreprocessor

        return PeptidePreprocessor.from_dataset(self)

    def _check_if_split_exists(self, split_name: str):
        existing_splits = list(self.hf_dataset.keys())
        if split_name not in existing_splits:
            raise ValueError(
                f"Split '{split_name}' does not exist in the dataset. Available splits are: {existing_splits}"
            )
        return True

    def _get_split_tf_dataset(self, split_name: str):
        self._check_if_split_exists(split_name)

        return to_tf_tensor_dataset(
            self.hf_dataset[split_name],
            input_columns=self._get_input_tensor_column_names(),
            batch_size=self.batch_size,
            label_cols=self.label_column,
            shuffle=(
                self.shuffle
                if split_name == PeptideDataset.DEFAULT_SPLIT_NAMES[0]
                else False
            ),
        )

    def _get_split_torch_dataset(self, split_name: str):
        self._check_if_split_exists(split_name)

        return to_torch_dataloader(
            self.hf_dataset[split_name],
            input_columns=self._get_input_tensor_column_names(),
            batch_size=self.batch_size,
            label_cols=self.label_column,
            shuffle=(
                self.shuffle
                if split_name == PeptideDataset.DEFAULT_SPLIT_NAMES[0]
                else False
            ),
            dataloader_kwargs=getattr(self, "torch_dataloader_kwargs", None),
        )


# ``load_processed_dataset`` lives in serialization.py (exported from ``dlomix.data``).
