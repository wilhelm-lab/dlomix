# deprecated

import abc
import json
import pandas as pd
import numpy as np
import tensorflow as tf

from dlomix.constants import DEFAULT_PARQUET_ENGINE

# abstract out generics (reading from sources, static methods, member variables, etc...)
# list abstract methods and annotate them @abc.abstractmethod

class Dataset(abc.ABC):
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
        batch_size=32,
        val_ratio=0,
        seed=21,
        test=False,
        sample_run=False,
    ):
        super(Dataset, self).__init__()

        np.random.seed(seed)
        self.seed = seed

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

        self._data_mean = 0
        self._data_std = 1

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.testing_mode = test

        # main split is "train" if not in testing mode, otherwise "test"
        self.main_split = (
            Dataset.SPLIT_NAMES[0]
            if not self.testing_mode
            else Dataset.SPLIT_NAMES[2]
        )

        # initialize sequences, targets, features, and identifiers
        self.sequences = None
        self.targets = None
        self.features_df = None
        self.example_id = None

        # initialize TF Datasets dict
        self.tf_dataset = (
            {self.main_split: None, Dataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

        self.indicies_dict = (
            {self.main_split: None, Dataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

    def load_data(self, data): # can be turned into abstract method, with a sample workflow
        """Load data into the dataset object, can be used to load data at a later point after initialization.
        This function triggers the whole pipeline of: data loading, validation (against sequence length), splitting, building TensorFlow dataset objects, and apply preprocessing.

        :param data: can be: tuple of two arrays (sequences, targets), single array (sequences), useful for test data, or a `str` with a file path toa csv file
        :return: None
        """
        self.data_source = data

        if isinstance(self.data_source, dict):
            # source is dict
            self._update_data_loading_for_json_format()


        self._read_data() # refactor into methods, all stays in abstract
        self._validate_remove_long_sequences() # stays in abstract
        self._split_data() # random split in base class, other options in child classes
        self._build_tf_dataset() # check abstraction
        self._preprocess_tf_dataset() # abstract method mostly

    """
    numpy array --> either a tuple or a single array
        - Tuple --> means (sequences, targets)
        - single ndarray --> means sequences only, useful for test dataset
    str --> path to csv file or compressed csv file
    """

    def _read_data(self):
        if not isinstance(self.data_source, (tuple, np.ndarray, str, dict)):
            # Raise error, not a valid input
            raise ValueError(
                "Data source has to be either a tuple of two numpy arrays, a single numpy array, an in-memory dict,"
                "or a string with a path to a csv/parquet/json file."
            )
        elif isinstance(self.data_source, tuple):
            self._read_tuple_data_source(self)

        elif isinstance(self.data_source, np.ndarray):
            self._read_array_data_source(self)

        else:
            self._read_string_or_dict_data_source(self)
            
        # give the index of the element as an ID for later reference if needed
        self.example_id = list(range(len(self.sequences)))

    def _read_tuple_data_source(self):
        # source is tuple
        tuple_size_is_two = len(self.data_source) == 2
        if tuple_size_is_two:
            tuple_elements_are_ndarray = isinstance(
                self.data_source[0], np.ndarray
            ) and isinstance(self.data_source[1], np.ndarray)
            if tuple_elements_are_ndarray:
                self.sequences = self.data_source[0]
                self.targets = self.data_source[1]
        else:
            raise ValueError(
                "If a tuple is provided, it has to have a length of 2 and both elements should be numpy arrays."
            )

    def _read_array_data_source(self):
        # source is numpy array
        # only sequences are provided
        self.sequences = self.data_source
        self.targets = np.zeros(self.sequences.shape[0])
        self._data_mean, self._data_std = 0, 1

    def _read_string_or_dict_data_source(self):
        # source is string or dict
        if isinstance(self.data_source, dict):
            #  a dict is passed via the json
            df = pd.DataFrame(self.data_source)
        else:
            # a string path is passed via the json or as a constructor argument
            df = self._resolve_string_data_path()

        # used only for testing with a smaller sample from a csv file
        if self.sample_run:
            df = df.head(Dataset.SAMPLE_RUN_N)

        # lower all column names
        df.columns = lower_and_trim_strings(df.columns)

        self.sequences, self.targets = (
            df[self.sequence_col].values,
            df[self.target_col].values,
        )
        self._data_mean, self._data_std = np.mean(self.targets), np.std(
            self.targets
        )

        self.features_df = df[self.feature_cols]
    
    
    def _update_data_loading_for_json_format(self):
        json_dict = self.data_source

        self.data_source = json_dict.get(Dataset.METADATA_KEY, "")
        self.target_col = json_dict.get(Dataset.PARAMS_KEY, {}).get(
            Dataset.TARGET_NAME_KEY, self.target_col
        )
        
        self.sequence_col = json_dict.get(Dataset.PARAMS_KEY, {}).get(
            Dataset.SEQUENCE_COLUMN_KEY, self.sequence_col
        )

    def _resolve_string_data_path(self):
        is_json_file = self.data_source.endswith(".json")

        if is_json_file:
            json_dict = read_json_file(self.data_source)
            self._update_data_loading_for_json_format(json_dict)

        is_parquet_url = ".parquet" in self.data_source and self.data_source.startswith(
            "http"
        )
        is_parquet_file = self.data_source.endswith(".parquet")
        is_csv_file = self.data_source.endswith(".csv")

        if is_parquet_url or is_parquet_file:
            df = read_parquet_file_pandas(self.data_source, DEFAULT_PARQUET_ENGINE)
            return df
        elif is_csv_file:
            df = pd.read_csv(self.data_source)
            return df
        else:
            raise ValueError(
                "Invalid data source provided as a string, please provide a path to a csv, parquet, or "
                "or a json file."
            )

    def _validate_remove_long_sequences(self) -> None:
        """
        Validate if all sequences are shorter than the padding length, otherwise drop them.
        """
        assert self.sequences.shape[0] > 0, "No sequences in the provided data."
        assert len(self.sequences) == len(
            self.targets
        ), "Count of examples does not match for sequences and targets."

        limit = self.seq_length
        vectorized_len = np.vectorize(lambda x: len(x))
        mask = vectorized_len(self.sequences) <= limit
        self.sequences, self.targets = self.sequences[mask], self.targets[mask]
        # once feature columns are introduced, apply the mask to the feature columns (subset the dataframe as well)

    def _split_data(self):
        n = len(self.sequences)

        if self.val_ratio != 0 and (not self.testing_mode):
            # add randomization for now and later consider the splitting logic
            self.indicies_dict[Dataset.SPLIT_NAMES[1]] = np.arange(n)[
                : int(n * self.val_ratio)
            ]
            self.indicies_dict[self.main_split] = np.arange(n)[
                int(n * self.val_ratio) :
            ]
        else:
            self.indicies_dict[self.main_split] = np.arange(n)

    def _build_tf_dataset(self):
        # consider adding the feature columns or extra inputs
        for split in self.tf_dataset.keys():
            self.tf_dataset[split] = tf.data.Dataset.from_tensor_slices(
                (
                    self.sequences[self.indicies_dict[split]],
                    self.targets[self.indicies_dict[split]],
                )
            )

    def _preprocess_tf_dataset(self):
        for split in self.tf_dataset.keys():
            # avoid normalizing targets for test data --> should not be needed
            if self.normalize_targets and not self.testing_mode:
                self.tf_dataset[split] = self.tf_dataset[split].map(
                    lambda s, t: self._normalize_target(s, t),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

            self.tf_dataset[split] = (
                self.tf_dataset[split]
                .map(
                    lambda s, t: self._split_sequence(s, t),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .map(
                    lambda s, t: self._pad_sequences(s, t),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            )

            if self.include_count_features:
                self.tf_dataset[split] = (
                    self.tf_dataset[split]
                    .map(
                        Dataset._convert_inputs_to_dict,
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                    .map(
                        lambda s, t: self._generate_single_counts(s, t),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                    .map(
                        lambda s, t: self._generate_di_counts(s, t),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                )

            self.tf_dataset[split] = (
                self.tf_dataset[split]
                .batch(self.batch_size)
                .prefetch(Dataset.BATCHES_TO_PREFETCH)
            )

    def get_split_targets(self, split="val"):
        """Retrieve all targets (original labels) for a specific split.

        :param split: a string specifiying the split name (train, val, test)
        :return: nd.array with the targets
        """
        if split not in self.indicies_dict.keys():
            raise ValueError(
                "requested split does not exist, availabe splits are: "
                + list(self.indicies_dict.keys())
            )

        return self.targets[self.indicies_dict[split]]

    def denormalize_targets(self, targets):
        """Denormalize the given targets (can also be predictions) by multiplying the standard deviation and adding the mean.

        :param targets: an nd.array with targets or predictions
        :return: a denormalized nd.array with the targets or the predictions
        """
        return targets * self._data_std + self._data_mean

    def _pad_sequences(self, seq, target):
        pad_len = tf.abs(self.seq_length - tf.size(seq))
        paddings = tf.concat([[0], [pad_len]], axis=0)
        seq = tf.pad(seq, [paddings], "CONSTANT")
        seq.set_shape([self.seq_length])
        return seq, target

    def _normalize_target(self, seq, target):

        target = tf.math.divide(
            tf.math.subtract(target, self._data_mean), self._data_std
        )
        return seq, target

    def _split_sequence(self, seq, target):
        splitted_seq = tf.strings.bytes_split(seq)

        return splitted_seq, target

    """
    if more than one input is added, inputs are added to a python dict, the following methods assume that
    """

    @staticmethod
    def _convert_inputs_to_dict(seq, target):
        return {"seq": seq}, target

    def _generate_single_counts(self, inputs, target):
        inputs["counts"] = tf.map_fn(
            lambda x: Dataset.ATOM_TABLE.lookup(x), inputs["seq"]
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
        ), f"Requested data split is not available, available splits are {self.tf_dataset.keys()}"
        if split in self.tf_dataset.keys():
            return self.tf_dataset[split]
        return self.tf_dataset

    @property
    def train_data(self):
        """TensorFlow Dataset object for the training data"""
        return self._get_tf_dataset(Dataset.SPLIT_NAMES[0])

    @property
    def val_data(self):
        """TensorFlow Dataset object for the validation data"""
        return self._get_tf_dataset(Dataset.SPLIT_NAMES[1])

    @property
    def test_data(self):
        """TensorFlow Dataset object for the test data"""
        return self._get_tf_dataset(Dataset.SPLIT_NAMES[2])

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


# to go to reader classes or reader utils
def read_parquet_file_pandas(filepath, parquet_engine):
    try:
        df = pd.read_parquet(filepath, engine=parquet_engine)
    except ImportError:
        raise ImportError(
            "Parquet engine is missing, please install fastparquet using pip or conda."
        )
    return df

def read_json_file(filepath):
    with open(filepath, "r") as j:
        json_dict = json.loads(j.read())
    return json_dict

def lower_and_trim_strings(strings):
    return [s.lower().trim() for s in strings]

