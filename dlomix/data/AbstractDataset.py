import abc
import json
import pandas as pd
import numpy as np
import tensorflow as tf

#from dlomix.constants import DEFAULT_PARQUET_ENGINE

# abstract out generics (reading from sources, static methods, member variables, etc...)
# list abstract methods and annotate them @abc.abstractmethod


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
        pass

    # @abc.abstractmethod
    # def _read_data(self):
    #     if not isinstance(self.data_source, (tuple, np.ndarray, str, dict)):
    #         # Raise error, not a valid input
    #         raise ValueError(
    #             "Data source has to be either a tuple of two numpy arrays, a single numpy array, an in-memory dict,"
    #             "or a string with a path to a csv/parquet/json file."
    #         )
    #     else:
    #         pass

    # def _read_tuple_data_source(self):
    #     # source is tuple
    #     tuple_size_is_two = len(self.data_source) == 2
    #     if tuple_size_is_two:
    #         tuple_elements_are_ndarray = isinstance(
    #             self.data_source[0], np.ndarray
    #         ) and isinstance(self.data_source[1], np.ndarray)
    #         if tuple_elements_are_ndarray:
    #             self.sequences = self.data_source[0]
    #             self.targets = self.data_source[1]
    #     else:
    #         raise ValueError(
    #             "If a tuple is provided, it has to have a length of 2 and both elements should be numpy arrays."
    #         )

    # def _read_array_data_source(self):
    #     # source is numpy array
    #     # only sequences are provided
    #     self.sequences = self.data_source
    #     self.targets = np.zeros(self.sequences.shape[0])
    #     self._data_mean, self._data_std = 0, 1

    # def _read_string_or_dict_data_source(self):
    #     # source is string or dict
    #     if isinstance(self.data_source, dict):
    #         #  a dict is passed via the json
    #         df = pd.DataFrame(self.data_source)
    #     else:
    #         # a string path is passed via the json or as a constructor argument
    #         df = self._resolve_string_data_path()

    #     # used only for testing with a smaller sample from a csv file
    #     if self.sample_run:
    #         df = df.head(AbstractDataset.SAMPLE_RUN_N)

    #     # lower all column names
    #     df.columns = lower_and_trim_strings(df.columns)

    #     self.sequences, self.targets = (
    #         df[self.sequence_col].values,
    #         df[self.target_col].values,
    #     )
    #     self._data_mean, self._data_std = np.mean(self.targets), np.std(
    #         self.targets
    #     )

    #     self.features_df = df[self.feature_cols]
    
    
    # def _update_data_loading_for_json_format(self):
    #     json_dict = self.data_source

    #     self.data_source = json_dict.get(AbstractDataset.METADATA_KEY, "")
    #     self.target_col = json_dict.get(AbstractDataset.PARAMS_KEY, {}).get(
    #         AbstractDataset.TARGET_NAME_KEY, self.target_col
    #     )
        
    #     self.sequence_col = json_dict.get(AbstractDataset.PARAMS_KEY, {}).get(
    #         AbstractDataset.SEQUENCE_COLUMN_KEY, self.sequence_col
    #     )

    # def _resolve_string_data_path(self):
    #     is_json_file = self.data_source.endswith(".json")

    #     if is_json_file:
    #         json_dict = read_json_file(self.data_source)
    #         self._update_data_loading_for_json_format(json_dict)

    #     is_parquet_url = ".parquet" in self.data_source and self.data_source.startswith(
    #         "http"
    #     )
    #     is_parquet_file = self.data_source.endswith(".parquet")
    #     is_csv_file = self.data_source.endswith(".csv")

    #     if is_parquet_url or is_parquet_file:
    #         df = read_parquet_file_pandas(self.data_source, DEFAULT_PARQUET_ENGINE)
    #         return df
    #     elif is_csv_file:
    #         df = pd.read_csv(self.data_source)
    #         return df
    #     else:
    #         raise ValueError(
    #             "Invalid data source provided as a string, please provide a path to a csv, parquet, or "
    #             "or a json file."
    #         )

    @abc.abstractmethod
    def _build_tf_dataset(self):
        # consider adding the feature columns or extra inputs
        pass
        # for split in self.tf_dataset.keys():
        #     self.tf_dataset[split] = tf.data.Dataset.from_tensor_slices(
        #         (
        #             self.inputs, self.outputs
        #         )
        #     )

    @abc.abstractmethod
    def _preprocess_tf_dataset(self):
        pass
    
    @abc.abstractmethod
    def get_split_targets(self, split="val"):
        """Retrieve all targets (original labels) for a specific split.
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

    """
    if more than one input is added, inputs are added to a python dict, the following methods assume that
    """

    @staticmethod
    @abc.abstractmethod
    def _convert_inputs_to_dict(inputs, target):
        pass

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
        ), f"Requested data split is not available, available splits are {self.tf_dataset.keys()}"
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

