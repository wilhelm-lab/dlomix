# adjust encoding schemes workflow --> check ppt for details
# check if config can wrap all the parameters and be used to manage the attributes of the class, reduce the assignment of attributes in the init method

import importlib
import logging
import os
import warnings
from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, Sequence, Value, load_dataset, load_from_disk

from ..constants import ALPHABET_UNMOD
from .dataset_config import DatasetConfig
from .dataset_utils import EncodingScheme, get_num_processors
from .processing.feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    FEATURE_EXTRACTORS_PARAMETERS,
    LookupFeatureExtractor,
)
from .processing.processors import (
    FunctionProcessor,
    SequenceEncodingProcessor,
    SequencePaddingProcessor,
    SequenceParsingProcessor,
    SequencePTMRemovalProcessor,
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
        Format of the data source file(s). Example formats are 'csv', 'json', 'parquet', etc.
    sequence_column : str
        Name of the column in the data source file that contains the peptide sequences.
    label_column : str
        Name of the column in the data source file that contains the labels.
    val_ratio : float
        Ratio of the validation data to the training data. The value should be between 0 and 1.
    max_seq_len : int
        Maximum sequence length to pad the sequences to. If set to 0, the sequences will not be padded.
    dataset_type : str
        Type of the tensor dataset to be generated afterwards. Possible values are "tf" and "pt" for TensorFlow and PyTorch respectively.
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
    padding_value : int
        Value to use for padding the sequences.
    alphabet : Dict
        Alphabet to use for encoding the amino acids in the sequences.
    encoding_scheme : Union[str, EncodingScheme]
        Encoding scheme to use for encoding the sequences. Possible values are "unmod" and "naive-mods" for unmodified sequences and sequences with PTMs respectively.
    processed : bool
        Flag to indicate whether the dataset has been processed or not.
    enable_tf_dataset_cache : bool
        Flag to indicate whether to enable TensorFlow Dataset caching (call `.cahce()` on the generate TF Datasets).
    disable_cache : bool
        Flag to indicate whether to disable Hugging Face Datasets caching. Default is False.
    auto_cleanup_cache : bool
        Flag to indicate whether to automatically clean up the temporary Hugging Face Datasets cache files. Default is True.
    num_proc : Optional[int]
        Number of processes to use for processing the dataset. Default is None, no multi-processing.
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

    def __init__(
        self,
        data_source: Union[str, List],
        val_data_source: Union[str, List],
        test_data_source: Union[str, List],
        data_format: str,
        sequence_column: str,
        label_column: str,
        val_ratio: float,
        max_seq_len: int,
        dataset_type: str,
        batch_size: int,
        model_features: Optional[List[str]],
        dataset_columns_to_keep: Optional[List[str]],
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        alphabet: Dict = ALPHABET_UNMOD,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.UNMOD,
        processed: bool = False,
        enable_tf_dataset_cache: bool = False,
        disable_cache: bool = False,
        auto_cleanup_cache: bool = True,
        num_proc: Optional[int] = None,
        batch_processing_size: Optional[int] = 1000,
    ):
        super(PeptideDataset, self).__init__()
        self.data_source = data_source
        self.val_data_source = val_data_source
        self.test_data_source = test_data_source

        self.data_format = data_format

        self.sequence_column = sequence_column
        self.label_column = label_column

        self.val_ratio = val_ratio
        self.max_seq_len = max_seq_len
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.model_features = model_features

        # to be kept in the hf dataset, but not returned in the tensor dataset
        if dataset_columns_to_keep is None:
            self.dataset_columns_to_keep = []
        else:
            self.dataset_columns_to_keep = dataset_columns_to_keep

        self.features_to_extract = features_to_extract
        self.pad = pad
        self.padding_value = padding_value
        self.alphabet = alphabet
        self.encoding_scheme = EncodingScheme(encoding_scheme)
        self.processed = processed
        self.enable_tf_dataset_cache = enable_tf_dataset_cache
        self.disable_cache = disable_cache
        self.auto_cleanup_cache = auto_cleanup_cache
        self._set_hf_cache_management()

        self.extended_alphabet = self.alphabet.copy()

        self._refresh_config()

        if not self.processed:
            self.hf_dataset: Optional[Union[Dataset, DatasetDict]] = None
            self._empty_dataset_mode = False
            self._is_predefined_split = False
            self._test_set_only = False
            self._num_proc = num_proc
            self._set_num_proc()
            self._batch_processing_size = batch_processing_size

            self._data_files_available_splits = {}
            self._load_dataset()
            self._decide_on_splitting()

            self._relevant_columns = []
            self._extracted_features_columns = []

            if not self._empty_dataset_mode:
                self._processors = []
                self._remove_unnecessary_columns()
                self._split_dataset()
                self._parse_sequences()

                self._configure_processing_pipeline()
                self._apply_processing_pipeline()
                if self.model_features is not None:
                    self._cast_model_feature_types_to_float()
                self._cleanup_temp_dataset_cache_files()
                self.processed = True
                self._refresh_config()

    def _set_num_proc(self):
        if self._num_proc:
            n_processors = get_num_processors()
            if self._num_proc > n_processors:
                warnings.warn(
                    f"Number of processors provided is greater than the available processors. Using the maximum number of processors available: {n_processors}."
                )
                self._num_proc = n_processors

    def _set_hf_cache_management(self):
        if self.disable_cache:
            from datasets import disable_caching

            disable_caching()

    def _refresh_config(self):
        self._config = DatasetConfig(
            data_source=self.data_source,
            val_data_source=self.val_data_source,
            test_data_source=self.test_data_source,
            data_format=self.data_format,
            sequence_column=self.sequence_column,
            label_column=self.label_column,
            val_ratio=self.val_ratio,
            max_seq_len=self.max_seq_len,
            dataset_type=self.dataset_type,
            batch_size=self.batch_size,
            model_features=self.model_features,
            dataset_columns_to_keep=self.dataset_columns_to_keep,
            features_to_extract=self.features_to_extract,
            pad=self.pad,
            padding_value=self.padding_value,
            alphabet=self.alphabet,
            encoding_scheme=self.encoding_scheme,
            processed=self.processed,
        )

        self._config._additional_data.update(
            {
                k: v
                for k, v in self.__dict__.items()
                if k.startswith("_") and k != "_config"
            }
        )

        self._config._additional_data.update({"cls": self.__class__.__name__})

    def _load_dataset(self):
        data_sources = [self.data_source, self.val_data_source, self.test_data_source]

        for split_name, source in zip(PeptideDataset.DEFAULT_SPLIT_NAMES, data_sources):
            if source is not None:
                self._data_files_available_splits[split_name] = source

        if len(self._data_files_available_splits) == 0:
            self._empty_dataset_mode = True
            warnings.warn(
                "No data files provided, please provide at least one data source if you plan to use this dataset directly. Otherwise, you can later load data into this empty dataset"
            )

        else:
            self._empty_dataset_mode = False

            self.hf_dataset = load_dataset(
                self.data_format, data_files=self._data_files_available_splits
            )

    def _decide_on_splitting(self):
        count_loaded_data_sources = len(self._data_files_available_splits)

        # one non-train data source provided -> if test, then test only, if val, then do not split
        if count_loaded_data_sources == 1:
            if self.test_data_source is not None:
                self._test_set_only = True
            if self.val_data_source is not None:
                self._is_predefined_split = True

        # two or more data sources provided -> no splitting in all cases
        if count_loaded_data_sources >= 2:
            if self.val_data_source is not None:
                self._is_predefined_split = True

        if self._is_predefined_split:
            warnings.warn(
                f"""
                Multiple data sources or a single non-train data source provided {self._data_files_available_splits}, please ensure that the data sources are already split into train, val and test sets
                since no splitting will happen. If not, please provide only one data_source and set the val_ratio to split the data into train and val sets."
                """
            )

    def _remove_unnecessary_columns(self):
        self._relevant_columns = [self.sequence_column, self.label_column]

        if self.model_features is not None:
            self._relevant_columns.extend(self.model_features)

        if self.dataset_columns_to_keep is not None:
            # additional columns to keep in the hugging face dataset only and not return as tensors
            self._relevant_columns.extend(self.dataset_columns_to_keep)

        # select only relevant columns from the Hugging Face Dataset (includes label column)
        self.hf_dataset = self.hf_dataset.select_columns(self._relevant_columns)

    def _split_dataset(self):
        if self._is_predefined_split or self._test_set_only:
            return

        # only a train dataset or a train and a test but no val -> split train into train/val

        splitted_dataset = self.hf_dataset[
            PeptideDataset.DEFAULT_SPLIT_NAMES[0]
        ].train_test_split(test_size=self.val_ratio)

        self.hf_dataset["train"] = splitted_dataset["train"]
        self.hf_dataset["val"] = splitted_dataset["test"]

        del splitted_dataset["train"]
        del splitted_dataset["test"]
        del splitted_dataset

    def _parse_sequences(self):
        # parse sequence in all encoding schemes
        sequence_parsing_processor = SequenceParsingProcessor(
            self.sequence_column,
            batched=True,
        )

        self.dataset_columns_to_keep.extend(
            SequenceParsingProcessor.PARSED_COL_NAMES.values()
        )
        self._processors.append(sequence_parsing_processor)

        if self.encoding_scheme == EncodingScheme.UNMOD:
            warnings.warn(
                f"""Encoding scheme is {self.encoding_scheme}, this enforces removing all occurences of PTMs in the sequences.
If you prefer to encode the (amino-acids)+PTM combinations as tokens in the vocabulary, please use the encoding scheme 'naive-mods'.
"""
            )

            self._processors.append(
                SequencePTMRemovalProcessor(
                    sequence_column_name=self.sequence_column, batched=True
                )
            )

    def _configure_processing_pipeline(self):
        self._configure_encoding_step()
        self._configure_padding_step()
        self._configure_feature_extraction_step()

    def _configure_encoding_step(self):
        encoding_processor = None

        if (
            self.encoding_scheme == EncodingScheme.UNMOD
            or self.encoding_scheme == EncodingScheme.NAIVE_MODS
        ):
            encoding_processor = SequenceEncodingProcessor(
                sequence_column_name=self.sequence_column,
                alphabet=self.extended_alphabet,
                batched=True,
            )

        else:
            raise NotImplementedError(
                f"Encoding scheme {self.encoding_scheme} is not implemented. Available encoding schemes are: {list(EncodingScheme.__members__)}."
            )

        self._processors.append(encoding_processor)

    def _configure_padding_step(self):
        if not self.pad:
            warnings.warn(
                "Padding is turned off, sequences will have variable lengths. Converting this dataset to tensors will cause errors unless proper stacking of examples is done."
            )
            return

        if self.max_seq_len > 0:
            seq_len = self.max_seq_len
        else:
            raise ValueError(
                f"Max sequence length provided is an integer but not a valid value: {self.max_seq_len}, only positive non-zero values are allowed."
            )

        padding_processor = SequencePaddingProcessor(
            sequence_column_name=self.sequence_column,
            batched=True,
            padding_value=self.padding_value,
            max_length=seq_len,
        )

        self._processors.append(padding_processor)

    def _configure_feature_extraction_step(self):
        if self.features_to_extract is None or len(self.features_to_extract) == 0:
            return

        for feature in self.features_to_extract:
            if isinstance(feature, str):
                feature_name = feature.lower()

                if feature_name not in AVAILABLE_FEATURE_EXTRACTORS:
                    warnings.warn(
                        f"Skipping feature extractor {feature} since it is not available. Please choose from the available feature extractors: {AVAILABLE_FEATURE_EXTRACTORS}."
                    )
                    continue

                # We pass here the parsed sequence to the feature extractor since it will always be a list with AA+PTM as elements
                feature_extactor = LookupFeatureExtractor(
                    sequence_column_name=SequenceParsingProcessor.PARSED_COL_NAMES[
                        "seq"
                    ],
                    feature_column_name=feature_name,
                    **FEATURE_EXTRACTORS_PARAMETERS[feature_name],
                    max_length=self.max_seq_len,
                    batched=True,
                )
            elif isinstance(feature, Callable):
                warnings.warn(
                    (
                        f"Using custom feature extractor from the user function {feature.__name__}"
                        "please ensure that the provided function pads the feature to the sequence length"
                        "so that all tensors have the same sequence length dimension."
                    )
                )

                feature_name = feature.__name__
                feature_extactor = FunctionProcessor(feature)
            else:
                raise ValueError(
                    f"Feature extractor {feature} is not a valid type. Please provide a function or a string that is a valid feature extractor name."
                )

            self._extracted_features_columns.append(feature_name)
            self._processors.append(feature_extactor)

    def _apply_processing_pipeline(self):
        for processor in self._processors:
            logger.info(f"Applying step: {processor.__class__.__name__}...")
            logger.debug(f"Applying step with arguments:\n\n{processor}...")
            self.hf_dataset = self.hf_dataset.map(
                processor,
                desc=f"Mapping {processor.__class__.__name__}",
                batched=processor.batched,
                batch_size=self._batch_processing_size,
                num_proc=self._num_proc,
            )
            logger.info(f"Done with step: {processor.__class__.__name__}.\n")

            if isinstance(processor, SequencePaddingProcessor):
                for split in self.hf_dataset.keys():
                    if split != "test":
                        logger.info(
                            f"Removing truncated sequences in the {split} split ..."
                        )

                        self.hf_dataset[split] = self.hf_dataset[split].filter(
                            lambda batch: batch[processor.KEEP_COLUMN_NAME],
                            batched=True,
                            num_proc=self._num_proc,
                            batch_size=self._batch_processing_size,
                        )
                self.hf_dataset = self.hf_dataset.remove_columns(
                    processor.KEEP_COLUMN_NAME
                )

    def _cast_model_feature_types_to_float(self):
        for split in self.hf_dataset.keys():
            new_features = self.hf_dataset[split].features.copy()

            for feature_name, feature_type in self.hf_dataset[split].features.items():
                # ensure model features are casted to float for concatenation later
                if feature_name not in self.model_features:
                    continue
                if feature_type.dtype.startswith("float"):
                    continue
                if isinstance(feature_type, Sequence):
                    new_features[feature_name] = Sequence(Value("float32"))
                if isinstance(feature_type, Value):
                    new_features[feature_name] = Value("float32")

            self.hf_dataset[split] = self.hf_dataset[split].cast(
                new_features,
                num_proc=self._num_proc,
                batch_size=self._batch_processing_size,
            )

    def _cleanup_temp_dataset_cache_files(self):
        if self.auto_cleanup_cache:
            cleaned_up = self.hf_dataset.cleanup_cache_files()
            logger.info(f"Cleaned up cache files: {cleaned_up}.")

    def save_to_disk(self, path: str):
        """
        Save the dataset to disk.

        Parameters
        ----------
        path : str
            Path to save the dataset to.

        """

        self._config.save_config_json(
            os.path.join(path, PeptideDataset.CONFIG_JSON_NAME)
        )

        self.hf_dataset.save_to_disk(path)

    @classmethod
    def load_from_disk(cls, path: str):
        """
        Load the dataset from disk.

        Parameters
        ----------
        path : str
            Path to load the dataset from.

        Returns
        -------
        PeptideDataset
            PeptideDataset object loaded from disk.
        """

        config = DatasetConfig.load_config_json(
            os.path.join(path, PeptideDataset.CONFIG_JSON_NAME)
        )

        hf_dataset = load_from_disk(path)

        dataset = cls.from_dataset_config(config)
        dataset.hf_dataset = hf_dataset
        return dataset

    @classmethod
    def from_dataset_config(cls, config: DatasetConfig):
        d = cls(
            data_source=config.data_source,
            val_data_source=config.val_data_source,
            test_data_source=config.test_data_source,
            data_format=config.data_format,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            val_ratio=config.val_ratio,
            max_seq_len=config.max_seq_len,
            dataset_type=config.dataset_type,
            batch_size=config.batch_size,
            model_features=config.model_features,
            dataset_columns_to_keep=config.dataset_columns_to_keep,
            features_to_extract=config.features_to_extract,
            pad=config.pad,
            padding_value=config.padding_value,
            alphabet=config.alphabet,
            encoding_scheme=config.encoding_scheme,
            processed=config.processed,
        )

        for k, v in config._additional_data.items():
            setattr(d, k, v)

        d._refresh_config()
        return d

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

        # remove the label column from the input tensor columns since the to_tf_dataset method has a separate label_cols argument
        input_tensor_columns.remove(self.label_column)

        # remove the columns that are not needed in the tensor dataset
        input_tensor_columns = list(
            set(input_tensor_columns) - set(self.dataset_columns_to_keep)
        )

        # add the extracted features columns to the input tensor columns
        input_tensor_columns.extend(self._extracted_features_columns)

        return input_tensor_columns

    @property
    def tensor_train_data(self):
        """TensorFlow Dataset object for the training data"""
        tf_dataset = self._get_split_tf_dataset(PeptideDataset.DEFAULT_SPLIT_NAMES[0])

        if self.enable_tf_dataset_cache:
            tf_dataset = tf_dataset.cache()

        return tf_dataset

    @property
    def tensor_val_data(self):
        """TensorFlow Dataset object for the val data"""
        tf_dataset = self._get_split_tf_dataset(PeptideDataset.DEFAULT_SPLIT_NAMES[1])

        if self.enable_tf_dataset_cache:
            tf_dataset = tf_dataset.cache()

        return tf_dataset

    @property
    def tensor_test_data(self):
        """TensorFlow Dataset object for the test data"""
        tf_dataset = self._get_split_tf_dataset(PeptideDataset.DEFAULT_SPLIT_NAMES[2])

        if self.enable_tf_dataset_cache:
            tf_dataset = tf_dataset.cache()

        return tf_dataset

    def _get_split_tf_dataset(self, split_name: str):
        existing_splits = list(self.hf_dataset.keys())
        if split_name not in existing_splits:
            raise ValueError(
                f"Split '{split_name}' does not exist in the dataset. Available splits are: {existing_splits}"
            )

        return self.hf_dataset[split_name].to_tf_dataset(
            columns=self._get_input_tensor_column_names(),
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )


def load_processed_dataset(path: str):
    """
    Load a processed peptide dataset from a given path.

    Parameters
    ----------
    path : str
        Path to the peptide dataset.

    Returns
    -------
    dlomix.data.PeptideDataset or one of its child classes
        Peptide dataset.
    """

    module = importlib.import_module("dlomix.data")

    config = DatasetConfig.load_config_json(
        os.path.join(path, PeptideDataset.CONFIG_JSON_NAME)
    )
    class_obj = getattr(module, config._additional_data.get("cls", "PeptideDataset"))
    hf_dataset = load_from_disk(path)

    dataset = class_obj.from_dataset_config(config)
    dataset.hf_dataset = hf_dataset
    return dataset
