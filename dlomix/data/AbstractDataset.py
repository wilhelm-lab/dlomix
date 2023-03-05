import abc

import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils import lower_and_trim_strings
from .parsers import ProformaParser

# what characterizes a datasets -->
#   1. reading mode (string, CSV, json, parquet, in-memory, etc..)
#   2. inputs (define sequence column name and additional existing feature names)
#   3. features to extract --> abstracted out in featureextractors list
#   4. outputs --> targets to use (names of column or key name in a dict)

# 1. identify reading mode
#    and call static readerclass that take a data source and return a DataFrame (later consider other data structures)
# 2. pick inputs from the data after reader has finished, maintain the inputs dict
# 3. pick targets from the data after the reader has finished, maintain the targets dict
# 4. run feature extractors based on input sequences, maintain features dict
# 5. build TF Datasets accordingly

# Consider collecting member variables related to the sequences in a named tuple (sequence, mod, n_term, c_term, etc..)

# consider making the dataset object iterable --> iterate over main split tf dataset


class AbstractDataset(abc.ABC):
    r"""Base class for datasets.

    Parameters
    -----------
    data_source : str, tuple of two numpy.ndarray, numpy.ndarray, optional
        source can be a tuple of two arrays (sequences, targets), single array (sequences), useful for test data, or a str with a file path to a csv file. Defaults to None.
    sep : str, optional
        separator to be used if the data source is a CSV file. Defaults to ",".
    sequence_col :  str, optional
        name of the column containing the sequences in the provided CSV. Defaults to "sequence".
    target_col : str, optional
        name of the column containing the targets (indexed retention time). Defaults to "irt".
    feature_cols : list, optional
        a list of columns containing other features that can be used later as inputs to a model. Defaults to None.
    seq_length : int, optional
        the sequence length to be used, where all sequences will be padded to this length, longer sequences will be removed and not truncated. Defaults to 0.
    parser : str, optional
        name of the parser to use. Available parsers are in `dlomix.data.parsers.py`. Defaults to None; no parsing to be done on the sequence (works for unmodified sequences).
    features_to_extract: list(dlomix.data.feature_extractors.SequenceFeatureExtractor), optional
        List of feature extractor objects. Defaults to None; no features to extract.
    batch_size : int, optional
        the batch size to be used for consuming the dataset in training a model. Defaults to 32.
    val_ratio : int, optional
        a fraction to determine the size of the validation data (0.2 = 20%). Defaults to 0.
    seed: int, optional
        a seed to use for splitting the data to allow for a reproducible split. Defaults to 21.
    test :bool, optional
        a boolean whether the dataset is a test dataset or not. Defaults to False.
    sample_run : bool, optional
        a boolean to limit the number of examples to a small number, SAMPLE_RUN_N, for testing and debugging purposes. Defaults to False.
    """

    ATOM_TABLE = None
    SPLIT_NAMES = ["train", "val", "test"]
    BATCHES_TO_PREFETCH = tf.data.AUTOTUNE

    SAMPLE_RUN_N = 100
    METADATA_KEY = "metadata"
    PARAMS_KEY = "parameters"
    TARGET_NAME_KEY = "target_column_key"
    SEQUENCE_COLUMN_KEY = "sequence_column_key"

    def __init__(
        self,
        data_source,
        sep,
        sequence_col,
        target_col,
        feature_cols=None,
        seq_length=0,
        parser=None,
        features_to_extract=None,
        batch_size=32,
        val_ratio=0,
        path_aminoacid_atomcounts=None,
        seed=21,
        test=False,
        sample_run=False,
    ):
        super(AbstractDataset, self).__init__()
        np.random.seed(seed)

        self.seed = seed
        np.random.seed(self.seed)

        self.data_source = data_source
        self.sep = sep
        self.sequence_col = sequence_col.lower()
        self.target_col = target_col.lower()

        if feature_cols:
            self.feature_cols = lower_and_trim_strings(feature_cols)
        else:
            self.feature_cols = []

        self.sample_run = sample_run

        # if seq_length is 0 (default) -> no padding
        self.seq_length = seq_length
        self.parser = parser
        self.features_to_extract = features_to_extract

        self._data_mean = 0
        self._data_std = 1

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.testing_mode = test

        # main split is "train" if not in testing mode, otherwise "test"
        self.main_split = (
            AbstractDataset.SPLIT_NAMES[0]
            if not self.testing_mode
            else AbstractDataset.SPLIT_NAMES[2]
        )

        # initialize TF Datasets dict
        self.tf_dataset = (
            {self.main_split: None, AbstractDataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

        self.indicies_dict = (
            {self.main_split: None, AbstractDataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

        # if path to counts lookup table is provided, include count features, otherwise not
        self.include_count_features = True if path_aminoacid_atomcounts else False

        if self.include_count_features:
            self.aminoacid_atom_counts_csv_path = (
                path_aminoacid_atomcounts  # "../lookups/aa_comp_rel.csv"
            )
            self._init_atom_table()

        self._resolve_parser()

    def _resolve_parser(self):
        if self.parser is None:
            return
        elif self.parser == "proforma":
            self.parser = ProformaParser()
        else:
            raise ValueError(
                f"Invalid parser provided {self.parser}. For a list of available parsers, check dlomix.data.parsers.py"
            )

        self.unmodified_sequences = None
        self.modifications = None
        self.n_term_modifications = None
        self.c_term_modifications = None

    def _parse_sequences(self):
        (
            self.sequences,
            self.modifications,
            self.n_term_modifications,
            self.c_term_modifications,
        ) = self.parser.parse_sequences(self.sequences)

    def _extract_features(self):
        if self.features_to_extract:
            self.sequence_features = []
            self.sequence_features_names = []
            for feature_class in self.features_to_extract:
                print("-" * 50)
                print("Extracting feature: ", feature_class)
                extractor_class = feature_class

                self.sequence_features.append(
                    np.array(
                        extractor_class.extract_all(
                            self.sequences,
                            self.modifications,
                            self.seq_length if extractor_class.pad_to_seq_length else 0,
                        )
                    )
                )
                self.sequence_features_names.append(
                    extractor_class.__class__.__name__.lower()
                )

    def get_examples_at_indices(self, examples, split):
        if isinstance(examples, np.ndarray):
            return examples[self.indicies_dict[split]]
        # to handle features
        if isinstance(examples, list):
            return [
                examples_single[self.indicies_dict[split]]
                for examples_single in examples
            ]
        raise ValueError(
            f"Provided data structure to subset for examples at split indices is neither a list nor a numpy array, but rather a {type(examples)}."
        )

    def _init_atom_table(self):
        atom_counts = pd.read_csv(self.aminoacid_atom_counts_csv_path)
        atom_counts = atom_counts.astype(str)

        keys_tensor = tf.constant(atom_counts["aa"].values)
        values_tensor = tf.constant(
            ["_".join(c) for c in list(atom_counts.iloc[:, 1:].values)]
        )
        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)
        AbstractDataset.ATOM_TABLE = tf.lookup.StaticHashTable(
            init, default_value="0_0_0_0_0"
        )

    @abc.abstractmethod
    def load_data(self, data):
        """load data from source and populate numpy arrays to use for tf.Dataset

        Args:
            data (str, tuple, dict): Path to csv or parquet file, tuple with numpy arrays, or a dict with keys
            `AbstractDataset.METADATA_KEY`, `AbstractDataset.PARAMS_KEY`,
            `AbstractDataset.TARGET_NAME_KEY`, `AbstractDataset.SEQUENCE_COLUMN_KEY`.
        """
        pass

    @abc.abstractmethod
    def _build_tf_dataset(self):
        """Build the tf.Dataset object for available splits using the data loaded by `load_data`.
        Example:
        `for split in self.tf_dataset.keys():
            self.tf_dataset[split] = tf.data.Dataset.from_tensor_slices(
                (self.inputs, self.outputs)
            )`
        """
        pass

    @abc.abstractmethod
    def _preprocess_tf_dataset(self):
        """Add processing logic (tensorflow functions) to apply to all tf.Datasets."""
        pass

    @abc.abstractmethod
    def get_split_targets(self, split="val"):
        """Retrieve all targets (original labels) for a specific split (dependent on the task at hand)
        Args:
            split (str, optional): Name of the split, check `AbstractDataset.SPLIT_NAMES`. Defaults to "val".
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _convert_inputs_to_dict(inputs, target):
        """Collect all inputs into a python dict with corresponding keys.
            When multiple inputs are used,this function is used at the beginning of the pre-processing
            of TF.Datasets.

        Args:
            inputs (tuple(tf.Tensor)): tuple of input tensors
            target (tf.Tensor): target label tensor
        """
        pass

    def _pad_sequences(self, inputs, target):
        if isinstance(inputs, dict):
            inputs["sequence"] = self._pad_seq(inputs["sequence"])
            return inputs, target
        else:
            return self._pad_seq(inputs), target

    def _pad_seq(self, seq):
        pad_len = tf.abs(self.seq_length - tf.size(seq))
        paddings = tf.concat([[0], [pad_len]], axis=0)
        seq = tf.pad(seq, [paddings], "CONSTANT")
        seq.set_shape([self.seq_length])
        return seq

    def _split_sequence(self, inputs, target):
        if isinstance(inputs, dict):
            inputs["sequence"] = tf.strings.bytes_split(inputs["sequence"])
            return inputs, target
        else:
            inputs = tf.strings.bytes_split(inputs)
        return inputs, target

    def _generate_single_counts(self, inputs, target):
        inputs["counts"] = tf.map_fn(
            lambda x: AbstractDataset.ATOM_TABLE.lookup(x), inputs["sequence"]
        )
        inputs["counts"] = tf.map_fn(
            lambda x: tf.strings.split(x, sep="_"), inputs["counts"]
        )
        inputs["counts"] = tf.strings.to_number(inputs["counts"])
        inputs["counts"].set_shape([self.seq_length, 5])

        return inputs, target

    def _generate_di_counts(self, inputs, target):
        # add every two neighboring elements without overlap [0 0 1 1 2 2 .... pad_length/2 pad_length/2]
        segments_to_add = [i // 2 for i in range(self.seq_length)]
        inputs["di_counts"] = tf.math.segment_sum(
            inputs["counts"], tf.constant(segments_to_add)
        )
        inputs["di_counts"].set_shape([self.seq_length // 2, 5])

        return inputs, target

    def _get_tf_dataset(self, split=None):
        assert (
            split in self.tf_dataset.keys()
        ), f"Requested data split {split} is not available, available splits are {self.tf_dataset.keys()}"
        if split in self.tf_dataset.keys():
            return self.tf_dataset[split]
        return self.tf_dataset

    @property
    def train_data(self):
        """TensorFlow Dataset object for the training data"""
        return self._get_tf_dataset(AbstractDataset.SPLIT_NAMES[0])

    @property
    def val_data(self):
        """TensorFlow Dataset object for the validation data"""
        return self._get_tf_dataset(AbstractDataset.SPLIT_NAMES[1])

    @property
    def test_data(self):
        """TensorFlow Dataset object for the test data"""
        return self._get_tf_dataset(AbstractDataset.SPLIT_NAMES[2])

    @property
    def data_mean(self):
        """Mean value of the targets"""
        return self._data_mean

    @property
    def data_std(self):
        """Standard deviation value of the targets"""
        return self._data_std

    @data_mean.setter
    def data_mean(self, value):
        self._data_mean = value

    @data_std.setter
    def data_std(self, value):
        self._data_std = value
