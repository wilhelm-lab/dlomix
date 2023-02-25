import numpy as np
import pandas as pd
import tensorflow as tf

from dlomix.utils import convert_nested_list_to_numpy_array

# take into consideration if the pandas dataframe is pickled or not and then call read_pickle instead of read_csv
# allow the possiblity to have three different dataset objects, one for train, val, and test


class IntensityDataset:
    r"""A dataset class for Intensity prediction tasks. It initialize a dataset object wrapping tf.Dataset and some relevant preprocessing steps.

    Parameters
    -----------
    data_source : str, tuple of two numpy.ndarray, numpy.ndarray, optional
        source can be a tuple of two arrays (sequences, targets), single array (sequences), useful for test data, or a str with a file path to a csv file. Defaults to None.
    sep : str, optional
        separator to be used if the data source is a CSV file. Defaults to ",".
    sequence_col :  str, optional
        name of the column containing the sequences in the provided CSV. Defaults to "sequence".
    target_col : str, optional
        name of the column containing the targets (vector of intensities). Defaults to "intensities_raww".
    feature_cols : list, optional
        a list of columns containing other features that can be used later as inputs to a model. Defaults to None.
    normalize_targets : bool, optional
        a boolean whether to normalize the targets or not (subtract mean and divied by standard deviation). Defaults to False.
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
    path_aminoacid_atomcounts : str, optional
        a string with a path to a CSV table with the atom counts of the different amino acids (can be used for feature extraction). Defaults to None.
    sample_run : bool, optional
        a boolean to limit the number of examples to a small number, SAMPLE_RUN_N, for testing and debugging purposes. Defaults to False.
    """
    SPLIT_NAMES = ["train", "val", "test"]
    BATCHES_TO_PREFETCH = tf.data.AUTOTUNE

    SAMPLE_RUN_N = 100

    # TODO: For test dataset --> examples with longer sequences --> do not drop, add NaN for prediction

    def __init__(
        self,
        data_source=None,
        sep=",",
        sequence_col="sequence",
        collision_energy_col="collision_energy_aligned_normed",
        precursor_charge_col="precursor_charge_onehot",
        intensities_col="intensities",
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
        super(IntensityDataset, self).__init__()

        np.random.seed(seed)
        self.seed = seed

        self.data_source = data_source
        self.sep = sep
        self.sequence_col = sequence_col.lower()
        self.collision_energy_col = collision_energy_col.lower()
        self.precursor_charge_col = precursor_charge_col.lower()
        self.intensities_col = intensities_col.lower()

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
        self.no_intensities = self.testing_mode

        self.main_split = (
            IntensityDataset.SPLIT_NAMES[2]
            if self.testing_mode
            else IntensityDataset.SPLIT_NAMES[0]
        )

        self.sequences = None
        self.collision_energy = None
        self.precursor_charge = None
        self.intensities = None

        self.features_df = None
        self.example_id = None

        # if path to counts lookup table is provided, include count features, otherwise not
        self.include_count_features = True if path_aminoacid_atomcounts else False

        if self.include_count_features:
            self.aminoacid_atom_counts_csv_path = (
                path_aminoacid_atomcounts  # "../lookups/aa_comp_rel.csv"
            )
            self._init_atom_table()

        # initialize TF Datasets dict
        self.tf_dataset = (
            {self.main_split: None, IntensityDataset.SPLIT_NAMES[1]: None}
            if val_ratio != 0
            else {self.main_split: None}
        )

        self.indicies_dict = (
            {self.main_split: None, IntensityDataset.SPLIT_NAMES[1]: None}
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
        IntensityDataset.ATOM_TABLE = tf.lookup.StaticHashTable(
            init, default_value="0_0_0_0_0"
        )

    def load_data(self, data):
        """Load data into the dataset object, can be used to load data at a later point after initialization.
        This function triggers the whole pipeline of: data loading, validation (against sequence length), splitting, building TensorFlow dataset objects, and apply preprocessing.

        :param data: a `str` with a file path to csv file
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
        - Tuple --> means (sequences, collision_energy, precursor_charge, intensities)
        - single ndarray --> means sequences only, useful for test dataset
    str --> path to csv file or compressed csv file
    """

    def _read_data(self):
        if isinstance(self.data_source, tuple):
            tuple_size_is_three_or_four = (
                len(self.data_source) == 3 or len(self.data_source) == 4
            )
            if tuple_size_is_three_or_four:
                tuple_elements_are_ndarray = all(
                    [isinstance(x, np.ndarray) for x in self.data_source]
                )
                if tuple_elements_are_ndarray:
                    self.sequences = self.data_source[0]
                    self.collision_energy = self.data_source[1]
                    self.precursor_charge = self.data_source[2]
                    if len(self.data_source) == 4:
                        self.intensities = self.data_source[3]
                        self.no_intensities = False

                    else:
                        self.intensities = np.zeros()
                        self.no_intensities = True
            else:
                raise ValueError(
                    "If a tuple is provided, it has to have a length of 4 and all elements should be numpy arrays."
                )
        elif isinstance(self.data_source, str):
            df = pd.read_csv(self.data_source)

            # used only for testing with a smaller sample from a csv file
            if self.sample_run:
                df = df.head(IntensityDataset.SAMPLE_RUN_N)

            # lower all column names
            df.columns = [col_name.lower() for col_name in df.columns]

            # retrieve columns from the dataframe
            self.sequences = df[self.sequence_col]
            self.collision_energy = df[self.collision_energy_col]
            self.precursor_charge = df[self.precursor_charge_col]
            self.intensities = df[self.intensities_col]

            # parse strings into lists, for precursor charge and intensities
            if isinstance(self.precursor_charge.iloc[0], str):
                self.precursor_charge = self.precursor_charge.apply(eval)

            if isinstance(self.intensities.iloc[0], str):
                self.intensities = self.intensities.apply(eval)

            # get numpy arrays with .values() for all inputs and intensities

            self.sequences = self.sequences.values

            # for concatenation later, we expand dimensions
            self.collision_energy = self.collision_energy.values.reshape(-1, 1)

            self.precursor_charge = convert_nested_list_to_numpy_array(
                self.precursor_charge.values, dtype=np.float64
            )
            self.intensities = convert_nested_list_to_numpy_array(
                self.intensities.values
            )

            self.features_df = df[self.feature_cols]
        else:
            raise ValueError(
                "Data source has to be either a tuple of four numpy arrays,"
                "or a string path to a csv file."
            )

        # give the index of the element as an ID for later reference if needed
        self.example_id = list(range(len(self.sequences)))

    def _validate_remove_long_sequences(self) -> None:
        """
        Validate if all sequences are shorter than the padding length, otherwise drop them.
        """
        assert self.sequences.shape[0] > 0, "No sequences in the provided data."

        # check if count of examples matches for all provided inputs
        lengths = [
            len(self.sequences),
            len(self.collision_energy),
            len(self.precursor_charge),
        ]
        if not self.no_intensities:
            lengths = lengths + [len(self.intensities)]

        assert np.all(
            lengths == np.array(lengths[0])
        ), "Count of examples does not match for sequences and targets."

        limit = self.seq_length
        vectorized_len = np.vectorize(lambda x: len(x))
        mask = vectorized_len(self.sequences) <= limit
        self.sequences = self.sequences[mask]
        self.collision_energy = self.collision_energy[mask]
        self.precursor_charge = self.precursor_charge[mask]
        self.intensities = self.intensities[mask]

        # once feature columns are introduced, apply the mask to the feature columns (subset the dataframe as well)

    def _split_data(self):
        n = len(self.sequences)

        if self.val_ratio != 0 and (not self.testing_mode):
            # add randomization for now and later consider the splitting logic
            self.indicies_dict[IntensityDataset.SPLIT_NAMES[1]] = np.arange(n)[
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
                    self.collision_energy[self.indicies_dict[split]],
                    self.precursor_charge[self.indicies_dict[split]],
                    self.intensities[self.indicies_dict[split]],
                )
            )

    def _preprocess_tf_dataset(self):
        # ToDo: convert input to dict and assume this as the general case --> abstract out in parent class

        for split in self.tf_dataset.keys():
            self.tf_dataset[split] = (
                self.tf_dataset[split]
                .map(
                    IntensityDataset._convert_inputs_to_dict,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .map(
                    lambda i, t: self._split_sequence(i, t),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .map(
                    lambda i, t: self._pad_sequences(i, t),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            )

            # Here: feature engineering on the fly if needed (atom counts, etc...)

            self.tf_dataset[split] = (
                self.tf_dataset[split]
                .batch(self.batch_size)
                .prefetch(IntensityDataset.BATCHES_TO_PREFETCH)
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

        return self.intensities[self.indicies_dict[split]]

    def denormalize_targets(self, targets):
        """Denormalize the given targets (can also be predictions) by multiplying the standard deviation and adding the mean.

        :param targets: an nd.array with targets or predictions
        :return: a denormalized nd.array with the targets or the predictions
        """
        return targets * self._data_std + self._data_mean

    def _pad_sequences(self, inputs, target):
        pad_len = tf.abs(self.seq_length - tf.size(inputs["sequence"]))
        paddings = tf.concat([[0], [pad_len]], axis=0)
        inputs["sequence"] = tf.pad(inputs["sequence"], [paddings], "CONSTANT")
        inputs["sequence"].set_shape([self.seq_length])
        return inputs, target

    def _normalize_target(self, seq, target):

        target = tf.math.divide(
            tf.math.subtract(target, self._data_mean), self._data_std
        )
        return seq, target

    def _split_sequence(self, inputs, target):

        inputs["sequence"] = tf.strings.bytes_split(inputs["sequence"])

        return inputs, target

    """
    if more than one input is added, inputs are added to a python dict, the following methods assume that
    """

    @staticmethod
    def _convert_inputs_to_dict(seq, collision, precursor, target):
        inputs_dict = {
            "sequence": seq,
            "collision_energy": collision,
            "precursor_charge": precursor,
        }
        return inputs_dict, target

    def _generate_single_counts(self, inputs, target):
        inputs["counts"] = tf.map_fn(
            lambda x: IntensityDataset.ATOM_TABLE.lookup(x), inputs["seq"]
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
        return self._get_tf_dataset(IntensityDataset.SPLIT_NAMES[0])

    @property
    def val_data(self):
        """TensorFlow Dataset object for the validation data"""
        return self._get_tf_dataset(IntensityDataset.SPLIT_NAMES[1])

    @property
    def test_data(self):
        """TensorFlow Dataset object for the test data"""
        return self._get_tf_dataset(IntensityDataset.SPLIT_NAMES[2])

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
