import pandas as pd
import numpy as np
import tensorflow as tf

"""
 TODO: check if it is better to abstract out a generic class for TF dataset wrapper, including:
 - splitting data logic (e.g. include task-specific stratification based on sequence length, iRT values)
 - loading data logic
 """

# allow the possiblity to have three different dataset objects, one for train, val, and test

# for intensity model, the output of a vector


class SpectraDataset:
    """A dataset class for Retention Time prediction tasks"""

    ATOM_TABLE = None
    SPLIT_NAMES = ["train", "val", "test"]
    BATCHES_TO_PREFETCH = tf.data.AUTOTUNE

    # TODO: For test dataset --> examples with longer sequences --> do not drop, add NaN for prediction

    def __init__(
        self,
        data_source=None,
        sep=",",
        sequence_col="sequence",
        target_col="irt",
        feature_cols=None,
        normalize_targets=False,
        seq_length=0,
        batch_size=32,
        val_ratio=0,
        seed=21,
        test=False,
        path_aminoacid_atomcounts=None,
        sample_run=False,
    ):
        """Initialize a dataset object wrapping tf.Dataset and preprocessing steps
        :param data_source: source can be a tuple of two arrays (sequences, targets), single array (sequences), useful for test data,
                or a str with a file path to a csv file
        :param sep:
        :param sequence_col:
        :param target_col:
        :param feature_cols:
        :param normalize_targets:
        :param seq_length:
        :param batch_size:
        :param val_ratio:
        :param seed:
        :param test:
        :param path_aminoacid_atomcounts:
        :param sample_run:
        """
        super(RetentionTimeDataset, self).__init__()

        np.random.seed(seed)
        self.seed = seed

        self.data_source = data_source
        self.sep = sep
        self.sequence_col = sequence_col.lower()
        self.target_col = target_col.lower()
        if feature_cols:
            self.feature_cols = [f.lower() for f in feature_cols]
        else:
            self.feature_cols = []

        self.normalize_targets = normalize_targets
        self.sample_run = sample_run

        # if seq_length is 0 (default) -> no padding
        self.seq_length = seq_length

        self._data_mean = 0
        self._data_std = 1

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.testing_mode = test

        self.main_split = (
            RetentionTimeDataset.SPLIT_NAMES[2]
            if self.testing_mode
            else RetentionTimeDataset.SPLIT_NAMES[0]
        )

        self.sequences = None
        self.targets = None
        self.features_df = None

        # if path to counts lookup table is provided, include count features, otherwise not
        self.include_count_features = True if path_aminoacid_atomcounts else False

        if self.include_count_features:
            self.aminoacid_atom_counts_csv_path = (
                path_aminoacid_atomcounts  # "../lookups/aa_comp_rel.csv"
            )
            self._init_atom_table()

        # initialize TF Datasets dict
        self.tf_dataset = (
            {self.main_split: None, RetentionTimeDataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

        self.indicies_dict = (
            {self.main_split: None, RetentionTimeDataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

        # if data is provided with the constructor call --> load, otherwise --> done
        if self.data_source is not None:
            self.load_data(data=data_source)

    def _init_atom_table(self):
        atom_counts = pd.read_csv(self.aminoacid_atom_counts_csv_path)
        atom_counts = atom_counts.astype(str)

        keys_tensor = tf.constant(atom_counts["aa"].values)
        values_tensor = tf.constant(
            ["_".join(c) for c in list(atom_counts.iloc[:, 1:].values)]
        )
        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)
        RetentionTimeDataset.ATOM_TABLE = tf.lookup.StaticHashTable(
            init, default_value="0_0_0_0_0"
        )

    def load_data(self, data):
        """Load data into the dataset object, can be used to load data at a later point after initialization.
        This function triggers the whole pipeline of: data loading, validation (against sequence length), splitting, building TensorFlow dataset objects, and apply preprocessing.

        :param data: can be: tuple of two arrays (sequences, targets), single array (sequences), useful for test data, or a `str` with a file path toa csv file
        :return: None
        """
        self.data_source = data

        self._read_data()
        self._validate_remove_long_sequences()
        self._split_data()
        self._build_tf_dataset()
        self._preprocess_tf_dataset()

    """
    numpy array --> either a tuple or a single array
        - Tuple --> means (sequences, targets)
        - single ndarray --> means sequences only, useful for test dataset
    str --> path to csv file or compressed csv file
    """

    def _read_data(self):
        if isinstance(self.data_source, tuple):
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

        elif isinstance(self.data_source, np.ndarray):
            self.sequences = self.data_source
            self.targets = np.zeros(self.sequences.shape[0])
            self._data_mean, self._data_std = 0, 1

        elif isinstance(self.data_source, str):
            df = pd.read_csv(self.data_source)
            if self.sample_run:
                df = df.head(100)

            # lower all column names
            df.columns = [col_name.lower() for col_name in df.columns]

            self.sequences, self.targets = (
                df[self.sequence_col].values,
                df[self.target_col].values,
            )
            self._data_mean, self._data_std = np.mean(self.targets), np.std(
                self.targets
            )

            self.features_df = df[self.feature_cols]
        else:
            raise ValueError(
                "Data source has to be either a tuple of two numpy arrays, a single numpy array, "
                "or a string path to a csv file."
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
        # apply the mask to the feature columns (subset the dataframe as well)

    def _split_data(self):
        n = len(self.targets)

        if self.val_ratio != 0 and (not self.testing_mode):
            # add randomization for now and later consider the splitting logic
            self.indicies_dict[RetentionTimeDataset.SPLIT_NAMES[1]] = np.arange(n)[
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
                        RetentionTimeDataset._convert_inputs_to_dict,
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
                .prefetch(RetentionTimeDataset.BATCHES_TO_PREFETCH)
            )

    def get_split_targets(self, split="val"):
        if split not in self.indicies_dict.keys():
            raise ValueError(
                "requested split does not exist, availabe splits are: "
                + list(self.indicies_dict.keys())
            )

        return self.targets[self.indicies_dict[split]]

    def denormalize_targets(self, targets):
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
            lambda x: RetentionTimeDataset.ATOM_TABLE.lookup(x), inputs["seq"]
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

    # change this to private
    def get_tf_dataset(self, split=None):
        assert (
            split in self.tf_dataset.keys()
        ), f"Requested data split is not available, available splits are {self.tf_dataset.keys()}"
        if split in self.tf_dataset.keys():
            return self.tf_dataset[split]
        return self.tf_dataset

    @property
    def train_data(self):
        return self.get_tf_dataset(RetentionTimeDataset.SPLIT_NAMES[0])

    @property
    def val_data(self):
        return self.get_tf_dataset(RetentionTimeDataset.SPLIT_NAMES[1])

    @property
    def test_data(self):
        return self.get_tf_dataset(RetentionTimeDataset.SPLIT_NAMES[2])

    @property
    def data_mean(self):
        return self._data_mean

    @property
    def data_std(self):
        return self._data_std

    @data_mean.setter
    def data_mean(self, value):
        self._data_mean = value

    @data_std.setter
    def data_std(self, value):
        self._data_std = value
