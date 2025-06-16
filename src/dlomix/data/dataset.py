# adjust encoding schemes workflow --> check ppt for details

import importlib
import logging
import os
import warnings
from typing import Callable, Optional, Union

from datasets import Dataset, DatasetDict, Sequence, Value, load_dataset, load_from_disk

from .dataset_config import DatasetConfig
from .dataset_utils import EncodingScheme, get_num_processors
from .processing.feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    FEATURE_EXTRACTORS_PARAMETERS,
    LookupFeatureExtractor,
)
from .processing.processors import (
    FunctionProcessor,
    PeptideDatasetBaseProcessor,
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
    padding_value : int
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

        if isinstance(dataset_config.label_column, str):
            self.label_column = [dataset_config.label_column]
        elif isinstance(dataset_config.label_column, list):
            self.label_column = dataset_config.label_column
        else:
            raise ValueError(
                "The label_column parameter should be a string or a list of strings."
            )

        self._set_hf_cache_management()

        self.extended_alphabet = None
        self.learning_alphabet_mode = True

        if self.alphabet:
            self.extended_alphabet = self.alphabet.copy()
            self.extended_alphabet.update({str(self.padding_value): 0})
            self.learning_alphabet_mode = False

        self._config = dataset_config

        # explcit assignments of processed attribute
        self.processed = dataset_config.processed
        if not self.processed:
            self.hf_dataset: Optional[Union[Dataset, DatasetDict]] = None
            self._empty_dataset_mode = False
            self._is_predefined_split = False
            self._test_set_only = False
            self._num_proc = dataset_config.num_proc
            self._set_num_proc()

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
        # load original config
        self._config = DatasetConfig(**self._config.__dict__)

        # update the config with the current object's attributes
        self._config.__dict__.update(
            {
                k: self.__dict__[k]
                for k in self._config.__dict__.keys()
                if k not in ["_additional_data"]
            }
        )

        # update the additional data in the config with the current object's attributes
        self._config._additional_data.update(
            {
                k: v
                for k, v in self.__dict__.items()
                if k.startswith("_") and k not in ["_config", "_additional_data"]
            }
        )

        # update the additional data in the config with the current object's class name
        self._config._additional_data.update({"cls": self.__class__.__name__})

    def _load_dataset(self):
        if self.data_format == "hub":
            self._load_from_hub()
            return

        if self.data_format == "hf":
            self._load_from_inmemory_hf_dataset()
            return

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

    def _load_from_hub(self):
        self.hf_dataset = load_dataset(self.data_source, **self._kwargs)
        self._empty_dataset_mode = False
        self._is_predefined_split = True
        warnings.warn(
            'The provided data is assumed to be hosted on the Hugging Face Hub since data_format is set to "hub". Validation and test data sources will be ignored.'
        )
        if isinstance(self.hf_dataset, DatasetDict):
            for split in self.hf_dataset.keys():
                if split not in PeptideDataset.DEFAULT_SPLIT_NAMES:
                    raise ValueError(
                        f"The split name {split} is not a valid split name. Please use one of the default split names: {PeptideDataset.DEFAULT_SPLIT_NAMES}."
                    )
            self._data_files_available_splits = {
                split: f"HF hub dataset - {self.data_source} - {split}"
                for split in self.hf_dataset
            }

        else:
            self._data_files_available_splits = {
                PeptideDataset.DEFAULT_SPLIT_NAMES[
                    0
                ]: f"HF hub dataset - {self.data_source}"
            }

    def _load_from_inmemory_hf_dataset(self):
        self._empty_dataset_mode = False
        self._is_predefined_split = True
        warnings.warn(
            f'The provided data is assumed to be an in-memory Hugging Face Dataset or DatasetDict object since data_format is set to "hf". Validation and test data sources will be ignored and the split names of the DatasetDict has to follow the default namings {PeptideDataset.DEFAULT_SPLIT_NAMES}.'
        )

        if isinstance(self.data_source, DatasetDict):
            self.hf_dataset = self.data_source
            self._data_files_available_splits = dict.fromkeys(self.hf_dataset)
            self._data_files_available_splits = {
                split: f"in-memory Dataset object - {split}"
                for split in self.hf_dataset
            }

        elif isinstance(self.data_source, Dataset):
            self.hf_dataset = DatasetDict()
            self.hf_dataset[PeptideDataset.DEFAULT_SPLIT_NAMES[0]] = self.data_source
            self._data_files_available_splits = {
                PeptideDataset.DEFAULT_SPLIT_NAMES[0]: "in-memory Dataset object"
            }
        else:
            raise ValueError(
                "The provided data source is not a valid Hugging Face Dataset/DatasetDict object. The data_format value should be set to 'hf' if you plan to use an in-memory Hugging Face Dataset/DatasetDict object."
            )

    def _decide_on_splitting(self):
        count_loaded_data_sources = len(self._data_files_available_splits)

        # one data source provided -> if test, then test only, if val, then do not split
        if count_loaded_data_sources == 1:
            if (
                self.test_data_source is not None
                or PeptideDataset.DEFAULT_SPLIT_NAMES[2]
                in self._data_files_available_splits
            ):
                # test data source provided OR hugging face dataset with test split only
                self._test_set_only = True
            if self.val_data_source is not None:
                self._is_predefined_split = True

        # two or more data sources provided -> no splitting in all cases
        if count_loaded_data_sources >= 2:
            self._is_predefined_split = True

        if self._is_predefined_split:
            warnings.warn(
                f"""
                Multiple data sources or a single non-train data source provided {self._data_files_available_splits}, please ensure that the data sources are already split into train, val and test sets
                since no splitting will happen. If not, please provide only one data_source and set the val_ratio to split the data into train and val sets."
                """
            )

    def _remove_unnecessary_columns(self):
        self._relevant_columns = [self.sequence_column, *self.label_column]

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
            with_termini=self.with_termini,
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
                extend_alphabet=self.learning_alphabet_mode,
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
            for split in self.hf_dataset.keys():
                logger.info(
                    "Applying step: %s on split %s...",
                    processor.__class__.__name__,
                    split,
                )

                logger.info(
                    "Applying step: %s on split %s...",
                    processor.__class__.__name__,
                    split,
                )

                logger.debug(
                    "Applying step with arguments:\n\n %s on split %s", processor, split
                )

                # split-specific logic for encoding
                if isinstance(processor, SequenceEncodingProcessor):
                    if split in PeptideDataset.DEFAULT_SPLIT_NAMES[0:2]:
                        # train/val split -> learn the alphabet unless otherwise specified
                        self._apply_processor_to_split(processor, split)

                        self.extended_alphabet = processor.alphabet.copy()

                    elif split == PeptideDataset.DEFAULT_SPLIT_NAMES[2]:
                        # test split -> use the learned alphabet from the train/val split
                        # and enable fallback to encoding unseen (AA, PTM) as unmodified Amino acids
                        processor.extend_alphabet = False
                        processor.set_alphabet(self.extended_alphabet)
                        processor.set_fallback(True)

                        self._apply_processor_to_split(processor, split)

                    else:
                        raise Warning(
                            f"When applying processors, found split '{split}' which is not a valid split name. Please use one of the default split names: {PeptideDataset.DEFAULT_SPLIT_NAMES} to ensure correct behavior."
                        )
                else:
                    # --------------------------------------------------------------------
                    # split-agnostic logic -> run processor for all splits
                    self._apply_processor_to_split(processor, split)
                    # --------------------------------------------------------------------

                # split-specific logic for truncating train/val sequences only after padding
                if isinstance(processor, SequencePaddingProcessor):
                    if split != PeptideDataset.DEFAULT_SPLIT_NAMES[2]:
                        logger.info("Removing truncated sequences in the %s ", split)

                        self.hf_dataset[split] = self.hf_dataset[split].filter(
                            lambda batch: batch[processor.KEEP_COLUMN_NAME],
                            batched=True,
                            num_proc=self._num_proc,
                            batch_size=self.batch_processing_size,
                        )

                logger.info("Done with step: %s \n", processor.__class__.__name__)

        self.hf_dataset = self.hf_dataset.remove_columns(
            SequencePaddingProcessor.KEEP_COLUMN_NAME
        )

    def _apply_processor_to_split(
        self, processor: PeptideDatasetBaseProcessor, split: str
    ):
        self.hf_dataset[split] = self.hf_dataset[split].map(
            processor,
            desc=f"Mapping {processor.__class__.__name__}",
            batched=processor.batched,
            batch_size=self.batch_processing_size,
            num_proc=self._num_proc,
        )

    def _cast_model_feature_types_to_float(self):
        def cast_to_float(feature):
            """Recursively casts Sequence and Value features to float32."""
            if isinstance(feature, Sequence):
                # Recursively apply the transformation to the nested feature
                return Sequence(cast_to_float(feature.feature))
            if isinstance(feature, Value):
                return Value("float32")
            return feature  # Return as is for unsupported feature types

        for split in self.hf_dataset.keys():
            new_features = self.hf_dataset[split].features.copy()

            for feature_name, feature_type in self.hf_dataset[split].features.items():
                # Ensure model features are casted to float for concatenation later
                if feature_name not in self.model_features:
                    continue
                new_features[feature_name] = cast_to_float(feature_type)

            self.hf_dataset[split] = self.hf_dataset[split].cast(
                new_features,
                num_proc=self._num_proc,
                batch_size=self.batch_processing_size,
            )

    def _cleanup_temp_dataset_cache_files(self):
        if self.auto_cleanup_cache:
            cleaned_up = self.hf_dataset.cleanup_cache_files()
            logger.info("Cleaned up cache files: %s.", cleaned_up)

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
        config_dict = config.__dict__.copy()

        # remove the additional data from the config dict
        config_dict.pop("_additional_data")

        d = cls(**config_dict)

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

            if self.enable_tf_dataset_cache:
                tf_dataset = tf_dataset.cache()

            return tf_dataset

    def _check_if_split_exists(self, split_name: str):
        existing_splits = list(self.hf_dataset.keys())
        if split_name not in existing_splits:
            raise ValueError(
                f"Split '{split_name}' does not exist in the dataset. Available splits are: {existing_splits}"
            )
        return True

    def _get_split_tf_dataset(self, split_name: str):
        self._check_if_split_exists(split_name)

        return self.hf_dataset[split_name].to_tf_dataset(
            columns=self._get_input_tensor_column_names(),
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )

    def _get_split_torch_dataset(self, split_name: str):
        self._check_if_split_exists(split_name)

        from torch.utils.data import DataLoader

        data_loader = DataLoader(
            dataset=self.hf_dataset[split_name].with_format(
                type="torch",
                columns=[*self._get_input_tensor_column_names(), *self.label_column],
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        return data_loader


# ------------------------------------------------------------------------------------------------


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
