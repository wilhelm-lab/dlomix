import abc
import warnings
from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from datasets_sql import query

from ..constants import ALPHABET_UNMOD
from .dataset_utils import (
    EncodingScheme,
    add_parsed_sequence_info,
    encode_sequence,
    pad_drop_sequence,
    remove_ptms,
    update_sequence_with_splitted_proforma_format,
)
from .parsers import ProformaParser


class AbstractPeptideDataset(abc.ABC):
    DEFAULT_SPLIT_NAMES = ["train", "val", "test"]

    def __init__(
        self,
        data_source: Union[str, List],
        val_data_source: Union[str, List],
        test_data_source: Union[str, List],
        data_format: str,
        sequence_column: str,
        label_column: str,
        val_ratio: float,
        max_seq_len: Union[int, str],
        dataset_type: str,
        batch_size: int,
        model_features: List[str],
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        vocab: Dict = ALPHABET_UNMOD,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.NO_MODS,
    ):
        super(AbstractPeptideDataset, self).__init__()
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
        self.extended_vocab = self.vocab.copy()

        self.dataset: Optional[Union[Dataset, DatasetDict]] = None
        self.empty_dataset_mode = False
        self.predefined_split = False
        self._load_dataset()

        self.relevant_columns = None

        if not self.empty_dataset_mode:
            self._remove_unnecessary_columns()
            self._split_dataset()
            self._parse_sequences()

            self.processing_pipeline_steps = []
            self.processing_pipeline_args = []

            self._configure_processing_pipeline()
            self._apply_procesing_pipeline()

    def _load_dataset(self):
        data_files_available_splits = {}

        if self.data_source is not None:
            data_files_available_splits[
                AbstractPeptideDataset.DEFAULT_SPLIT_NAMES[0]
            ] = self.data_source
        if self.val_data_source is not None:
            data_files_available_splits[
                AbstractPeptideDataset.DEFAULT_SPLIT_NAMES[1]
            ] = self.val_data_source
        if self.val_data_source is not None:
            data_files_available_splits[
                AbstractPeptideDataset.DEFAULT_SPLIT_NAMES[2]
            ] = self.test_data_source

        if len(data_files_available_splits) == 0:
            self.empty_dataset_mode = True
            warnings.warn(
                "No data files provided, please provide at least one data source if you plan to use this dataset directly. You can later load data into this empty dataset"
            )
        if len(data_files_available_splits) > 1:
            self.predefined_split = True
            warnings.warn(
                f"""
                Multiple data sources provided {data_files_available_splits}, please ensure that the data sources are already split into train, val and test sets
                since no splitting will happen. If not, please provide only one data source and set the val_ratio to split the data into train and val sets."
                """
            )

        self.dataset = load_dataset(
            self.data_format, data_files=data_files_available_splits
        )

    def _remove_unnecessary_columns(self):
        self.relevant_columns = [self.sequence_column, self.label_column]

        if self.model_features is not None:
            self.relevant_columns.extend(self.model_features)
        self.dataset = self.dataset.select_columns(self.relevant_columns)

    def _split_dataset(self):
        # logic to split
        if not self.predefined_split:
            # only a train dataset is availble in the DatasetDict
            self.dataset = self.dataset[
                AbstractPeptideDataset.DEFAULT_SPLIT_NAMES[0]
            ].train_test_split(test_size=self.val_ratio)
            self.dataset["val"] = self.dataset["test"]
            del self.dataset["test"]

    def _parse_sequences(self):
        if self.encoding_scheme == EncodingScheme.NO_MODS:
            warnings.warn(
                f"""
                          Encoding scheme is {self.encoding_scheme}, this enforces removing all occurences of PTMs in the sequences.
                          If you prefer to encode the sequence+PTM combinations as new tokens in the vocabulary, please use the encoding scheme 'naive-mods'.
                          """
            )

            # ensure no ptm info is present
            self.dataset = self.dataset.map(
                lambda x: remove_ptms(x, self.sequence_column), desc="Removing PTMs..."
            )
        else:
            # parse sequences only if encoding scheme is not unmod
            # here naive mods is the only other option for now (hence, extend the vocabulary)
            self.parser = ProformaParser(build_naive_vocab=True, base_vocab=self.vocab)
            self.dataset = self.dataset.map(
                lambda x: add_parsed_sequence_info(
                    x, self.sequence_column, self.parser
                ),
                desc="Parsing sequences...",
            )

            # get the extended vocab from the parser
            self.extended_vocab = self.parser.extended_vocab

    def _configure_processing_pipeline(self):
        # Filter and Pad
        # drop sequences longer than max_seq
        self._configure_encoding_step()

        if not self.pad:
            warnings.warn(
                "Padding is turned off, sequences will have a variable length."
            )
        else:
            self._configure_padding_step()

        self._configure_feature_extraction_step()

    def _configure_padding_step(self):
        if isinstance(self.max_seq_len, str):
            if self.max_seq_len == "max":
                # find max sequence length in the dataset
                # experimental = not working!
                d = self.dataset["train"]
                if self.encoding_scheme == EncodingScheme.NO_MODS:
                    seq_len = query(
                        f"SELECT MAX(LENGTH({self.sequence_column})) FROM d"
                    )
                else:
                    seq_len = query("SELECT MAX(LENGTH(raw_sequence)) FROM d")
                pass
            else:
                raise ValueError(
                    f"Max sequence length provided is a string but not a valid value: {self.max_seq_len}, only 'max' is allowed value."
                )

        elif isinstance(self.max_seq_len, int):
            if self.max_seq_len > 0:
                seq_len = self.max_seq_len
                pass
            else:
                raise ValueError(
                    f"Max sequence length provided is an integer but not a valid value: {self.max_seq_len}, only positive values are allowed."
                )
        else:
            raise ValueError(
                f"Max sequence length provided is neither an int nor a string; type provided is {type(self.max_seq_len)}, value is {self.max_seq_len}."
            )

        self.processing_pipeline_steps.append(pad_drop_sequence)
        self.processing_pipeline_args.append(
            {
                "seq_len": seq_len,
                "padding": self.padding_value,
                "sequence_column_name": self.sequence_column,
            }
        )

    def _configure_encoding_step(self):
        if self.encoding_scheme == EncodingScheme.NO_MODS:
            self.processing_pipeline_steps.append(encode_sequence)
            self.processing_pipeline_args.append(
                {"sequence_column_name": self.sequence_column, "alphabet": self.vocab}
            )
        elif self.encoding_scheme == EncodingScheme.NAIVE_MODS:
            warnings.warn(
                "Naive encoding for PTMs: please use the dataset attribute extended_vocab for the full alphabet and pass it to the model if needed. \nUsage: dataset.extended_vocab"
            )
            # add proforma sequence splitted column
            self.processing_pipeline_steps.append(
                update_sequence_with_splitted_proforma_format
            )
            self.processing_pipeline_args.append(
                {
                    "raw_sequence_column_name": "raw_sequence",
                    "mods_column_name": "mods",
                    "n_term_column_name": "n_terminal_mods",
                    "c_term_column_name": "c_terminal_mods",
                    "new_column_name": self.sequence_column,
                }
            )

            # encode proforma sequence with the extended vocab provided by the parser
            self.processing_pipeline_steps.append(encode_sequence)
            self.processing_pipeline_args.append(
                {
                    "sequence_column_name": self.sequence_column,
                    "alphabet": self.extended_vocab,
                }
            )
        else:
            raise NotImplementedError(
                f"Encoding scheme {self.encoding_scheme} is not implemented. Available encoding schemes are: {list(EncodingScheme.__members__)}."
            )

    def _configure_feature_extraction_step(self):
        pass

    def _apply_procesing_pipeline(self):
        for step, args in zip(
            self.processing_pipeline_steps, self.processing_pipeline_args
        ):
            print(f"For the next step, using the following args: {args}")
            self.dataset = self.dataset.map(
                lambda x: step(x, **args), desc=f"Applying {step.__name__}..."
            )
            print("-" * 100)

    def _extract_features(self):
        pass
        # for feature in self.features_to_extract:
        # if isinstance(feature, str):
        # feature_function = get_feature_function("feature_friendly_name")
        # elif isinstance(feature, Callable):
        # feature_function = feature
        # else:
        #    raise ValueError("")
        # self.dataset = self.dataset.map(feature_function, desc=f"Extracting {feature_friendly_name}")

        # self.dataset = self.dataset.map(get_mod_loss_feature, desc="Extracting modification loss feature...")

    @property
    def tensor_train_data(self):
        """TensorFlow Dataset object for the training data"""
        return self.dataset["train"].to_tf_dataset(
            columns=[self.sequence_column, *self.model_features]
            if self.model_features is not None
            else self.sequence_column,
            label_cols=self.label_column,
            shuffle=True,
            batch_size=self.batch_size,
        )

    @property
    def tensor_val_data(self):
        """TensorFlow Dataset object for the val data"""
        return self.dataset["val"].to_tf_dataset(
            columns=[self.sequence_column, *self.model_features]
            if self.model_features is not None
            else self.sequence_column,
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )

    @property
    def tensor_test_data(self):
        """TensorFlow Dataset object for the test data"""
        return self.dataset["test"].to_tf_dataset(
            columns=[self.sequence_column, *self.model_features]
            if self.model_features is not None
            else self.sequence_column,
            label_cols=self.label_column,
            shuffle=False,
            batch_size=self.batch_size,
        )
