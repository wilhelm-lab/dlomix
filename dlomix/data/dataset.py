import os
import warnings
from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from ..constants import ALPHABET_UNMOD
from .dataset_config import DatasetConfig
from .dataset_utils import EncodingScheme, get_num_processors, remove_ptms
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
)


class PeptideDataset:
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
        model_features: List[str],
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        vocab: Dict = ALPHABET_UNMOD,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.NO_MODS,
        processed: bool = False,
        disable_cache: bool = True,
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
        self.features_to_extract = features_to_extract
        self.pad = pad
        self.padding_value = padding_value
        self.vocab = vocab
        self.encoding_scheme = EncodingScheme(encoding_scheme)
        self.processed = processed
        self.disable_cache = disable_cache
        self._set_hf_cache_management()

        self.extended_vocab = self.vocab.copy()

        self._refresh_config()

        if not self.processed:
            self.hf_dataset: Optional[Union[Dataset, DatasetDict]] = None
            self._empty_dataset_mode = False
            self._is_predefined_split = False
            self._default_num_proc = get_num_processors()
            self._default_batch_processing_size = 1000
            self._load_dataset()

            self._relevant_columns = None

            if not self._empty_dataset_mode:
                self._processors = []
                self._remove_unnecessary_columns()
                self._split_dataset()
                self._parse_sequences()

                self._configure_processing_pipeline()
                self._apply_processing_pipeline()
                self.processed = True
                self._refresh_config()

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
            features_to_extract=self.features_to_extract,
            pad=self.pad,
            padding_value=self.padding_value,
            vocab=self.vocab,
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

    def _load_dataset(self):
        data_files_available_splits = {}

        if self.data_source is not None:
            data_files_available_splits[
                PeptideDataset.DEFAULT_SPLIT_NAMES[0]
            ] = self.data_source
        if self.val_data_source is not None:
            data_files_available_splits[
                PeptideDataset.DEFAULT_SPLIT_NAMES[1]
            ] = self.val_data_source
        if self.test_data_source is not None:
            data_files_available_splits[
                PeptideDataset.DEFAULT_SPLIT_NAMES[2]
            ] = self.test_data_source

        if len(data_files_available_splits) == 0:
            self._empty_dataset_mode = True
            warnings.warn(
                "No data files provided, please provide at least one data source if you plan to use this dataset directly. You can later load data into this empty dataset"
            )
        if len(data_files_available_splits) > 1:
            self._is_predefined_split = True
            warnings.warn(
                f"""
                Multiple data sources provided {data_files_available_splits}, please ensure that the data sources are already split into train, val and test sets
                since no splitting will happen. If not, please provide only one data source and set the val_ratio to split the data into train and val sets."
                """
            )

        self.hf_dataset = load_dataset(
            self.data_format, data_files=data_files_available_splits
        )

    def _remove_unnecessary_columns(self):
        self._relevant_columns = [self.sequence_column, self.label_column]

        if self.model_features is not None:
            self._relevant_columns.extend(self.model_features)

        # select only relevant columns
        self.hf_dataset = self.hf_dataset.select_columns(self._relevant_columns)

        # remove label column
        self._relevant_columns = list(
            set(self._relevant_columns) - set(self.label_column)
        )

    def _split_dataset(self):
        # logic to split
        if not self._is_predefined_split:
            # only a train dataset is availble in the DatasetDict
            self.hf_dataset = self.hf_dataset[
                PeptideDataset.DEFAULT_SPLIT_NAMES[0]
            ].train_test_split(test_size=self.val_ratio)
            self.hf_dataset["val"] = self.hf_dataset["test"]
            del self.hf_dataset["test"]

    def _parse_sequences(self):
        if self.encoding_scheme == EncodingScheme.NO_MODS:
            warnings.warn(
                f"""
                          Encoding scheme is {self.encoding_scheme}, this enforces removing all occurences of PTMs in the sequences.
                          If you prefer to encode the sequence+PTM combinations as new tokens in the vocabulary, please use the encoding scheme 'naive-mods'.
                          """
            )

            # ensure no ptm info is present
            # ToDo: convert to processor style and decide on flow, whether this is a valid use-case or not
            self.hf_dataset = self.hf_dataset.map(
                lambda x: remove_ptms(x, self.sequence_column), desc="Removing PTMs..."
            )
        else:
            # parse sequences only if encoding scheme is not unmod
            # here naive mods is the only other option for now (hence, extend the vocabulary)
            # self.parser = ProformaParser(build_naive_vocab=True, base_vocab=self.vocab)
            # self.dataset = self.dataset.map(
            #    lambda x: add_parsed_sequence_info(
            #        x, self.sequence_column, self.parser
            #    ),
            #    desc="Parsing sequences...",
            # )

            # get the extended vocab from the parser
            # self.extended_vocab = self.parser.extended_vocab

            sequence_parsing_processor = SequenceParsingProcessor(
                self.sequence_column, batched=True
            )

            self._processors.append(sequence_parsing_processor)

    def _configure_processing_pipeline(self):
        self._configure_encoding_step()
        self._configure_padding_step()
        self._configure_feature_extraction_step()

    def _configure_encoding_step(self):
        # ToDo: add sequence encoder
        encoding_processor = None

        if self.encoding_scheme == EncodingScheme.NO_MODS:
            encoding_processor = SequenceEncodingProcessor(
                sequence_column_name=self.sequence_column,
                alphabet=self.vocab,
                batched=True,
            )

        elif self.encoding_scheme == EncodingScheme.NAIVE_MODS:
            warnings.warn(
                "Naive encoding for PTMs: please use the dataset attribute extended_vocab for the full alphabet and pass it to the model if needed. \nUsage: dataset.extended_vocab"
            )

            # encode proforma sequence with the extended vocab provided by the parser
            # ToDo: double check if that is needed, if not, this can be independent of the encoding scheme
            encoding_processor = SequenceEncodingProcessor(
                sequence_column_name=self.sequence_column,
                alphabet=self.extended_vocab,
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
                "Padding is turned off, sequences will have a variable length."
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

                feature_extactor = LookupFeatureExtractor(
                    sequence_column_name="raw_sequence",
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
            self._relevant_columns.append(feature_name)
            self.model_features.append(feature_name)
            self._processors.append(feature_extactor)

    def _apply_processing_pipeline(self):
        for processor in self._processors:
            print(f"Applying processor:\n {processor}...")
            self.hf_dataset = self.hf_dataset.map(
                processor,
                desc=f"Applying {processor.__class__.__name__}",
                batched=processor.batched,
                batch_size=self._default_batch_processing_size,
                num_proc=self._default_num_proc,
            )

            if isinstance(processor, SequencePaddingProcessor):
                for split in self.hf_dataset.keys():
                    if split != "test":
                        print(f"Truncating sequences in the {split} split ...")

                        self.hf_dataset[split] = self.hf_dataset[split].filter(
                            lambda batch: batch[processor.KEEP_COLUMN_NAME],
                            batched=True,
                            num_proc=self._default_num_proc,
                            batch_size=self._default_batch_processing_size,
                        )
                self.hf_dataset = self.hf_dataset.remove_columns(
                    processor.KEEP_COLUMN_NAME
                )

            print("-" * 100)

    def __getitem__(self, index):
        return self.hf_dataset[index]

    def save_to_disk(self, path: str):
        self.hf_dataset.save_to_disk(path)
        self._config.save_config_json(
            os.path.join(path, PeptideDataset.CONFIG_JSON_NAME)
        )

    @classmethod
    def load_from_disk(cls, path: str):
        hf_dataset = load_from_disk(path)
        config = DatasetConfig.load_config_json(
            os.path.join(path, PeptideDataset.CONFIG_JSON_NAME)
        )
        print(config)
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
            features_to_extract=config.features_to_extract,
            pad=config.pad,
            padding_value=config.padding_value,
            vocab=config.vocab,
            encoding_scheme=config.encoding_scheme,
            processed=config.processed,
        )

        for k, v in config._additional_data.items():
            setattr(d, k, v)

        d._refresh_config()
        return d

    @property
    def tensor_train_data(self):
        """TensorFlow Dataset object for the training data"""
        return self.hf_dataset["train"].to_tf_dataset(
            columns=self._relevant_columns,
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )

    @property
    def tensor_val_data(self):
        """TensorFlow Dataset object for the val data"""
        return self.hf_dataset["val"].to_tf_dataset(
            columns=self._relevant_columns,
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )

    @property
    def tensor_test_data(self):
        """TensorFlow Dataset object for the test data"""
        return self.hf_dataset["test"].to_tf_dataset(
            columns=self._relevant_columns,
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )
