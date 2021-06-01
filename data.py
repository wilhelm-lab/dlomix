import pandas as pd
import numpy as np
import tensorflow as tf


class RTDataset:
    TARGETS_MEAN, TARGETS_STD = 0, 1
    SEQ_LENGTH = 0
    ATOM_TABLE = None

    # if pad_length is 0 -> no padding

    def __init__(self, data_source=None, sep=",", sequence_col="sequence", target_col="irt", feature_cols=[],
                 normalize_targets=True, pad_length=0, batch_size=32, val_ratio=0.2, seed=21,
                 test=False, path_aminoacid_atomcounts=None, sample_run=False):
        super(RTDataset, self).__init__()

        np.random.seed(seed)

        self.data_source = data_source
        self.sep = sep
        self.sequence_col = sequence_col.lower()
        self.target_col = target_col.lower()
        self.feature_cols = [f.lower() for f in feature_cols]

        self.normalize_targets = normalize_targets
        self.pad_length = pad_length
        self.sample_run = sample_run

        RTDataset.SEQ_LENGTH = pad_length

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.main_split = 'test' if test else 'train'

        self.sequences = None
        self.targets = None
        self.features_df = None

        # if path to counts lookup table is provided, include count features, otherwise not
        self.include_count_features = True if path_aminoacid_atomcounts else False

        if self.include_count_features:
            self.aminoacid_atom_counts_csv_path = path_aminoacid_atomcounts # "../lookups/aa_comp_rel.csv"
            self._init_atom_table()

        self.tf_dataset = {self.main_split: None, 'val': None} if val_ratio != 0 else {self.main_split: None}
        self.indicies_dict = {self.main_split: None, 'val': None} if val_ratio != 0 else {self.main_split: None}

        # if data is provided with the constructor call --> load, otherwise --> done
        if self.data_source is not None:
            self.load_data(data=data_source)

    def _init_atom_table(self):
        atom_counts = pd.read_csv(self.aminoacid_atom_counts_csv_path)
        atom_counts = atom_counts.astype(str)

        keys_tensor = tf.constant(atom_counts['aa'].values)
        values_tensor = tf.constant(['_'.join(c) for c in list(atom_counts.iloc[:, 1:].values)])
        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)
        RTDataset.ATOM_TABLE = tf.lookup.StaticHashTable(init, default_value='0_0_0_0_0')

    def load_data(self, data):
        self.data_source = data

        self._read_data()
        self._split_data()
        self._build_dataset()
        self._preprocess_dataset()

    '''
    numpy array --> either a tuple or a single array
        - Tuple --> means (sequences, targets)
        - single ndarray --> means sequences only, useful for test dataset
    '''

    def _read_data(self):
        if isinstance(self.data_source, tuple):
            if len(self.data_source) == 2 and isinstance(self.data_source[0], np.ndarray) and isinstance(self.data_source[1], np.ndarray):
                self.sequences = self.data_source[0]
                self.targets = self.data_source[1]
            else:
                raise ValueError('If a tuple is provided, it has to have a length of 2 and both elements should be numpy arrays.')

        elif isinstance(self.data_source, np.ndarray):
            self.sequences = self.data_source
            self.targets = np.zeros(self.sequences.shape[0])
            RTDataset.TARGETS_MEAN, RTDataset.TARGETS_STD = 0, 1

        elif isinstance(self.data_source, str):
            df = pd.read_csv(self.data_source)
            if self.sample_run:
                df = df.head(100)

            # lower all column names
            df.columns = [col_name.lower() for col_name in df.columns]

            self.sequences, self.targets = df[self.sequence_col].values, df[self.target_col].values
            RTDataset.TARGETS_MEAN, RTDataset.TARGETS_STD = np.mean(self.targets), np.std(self.targets)

            self.features_df = df[self.feature_cols]
        else:
            raise ValueError('Data source has to be either a tuple of two numpy arrays, a single numpy array, '
                             'or a string path to a csv file.')

    def _split_data(self):
        n = len(self.targets)

        if self.val_ratio != 0:
            self.indicies_dict['val'] = np.arange(n)[:int(n * self.val_ratio)]
            self.indicies_dict[self.main_split] = np.arange(n)[int(n * self.val_ratio):]
        else:
            self.indicies_dict[self.main_split] = np.arange(n)

    def _build_dataset(self):
        for split in self.tf_dataset.keys():
            self.tf_dataset[split] = tf.data.Dataset.from_tensor_slices(
                (self.sequences[self.indicies_dict[split]], self.targets[self.indicies_dict[split]]))

    def _preprocess_dataset(self):
        for split in self.tf_dataset.keys():
            if self.normalize_targets:
                self.tf_dataset[split] = self.tf_dataset[split].map(RTDataset.normalize_target)

            self.tf_dataset[split] = self.tf_dataset[split].map(RTDataset.split_sequence).map(
                RTDataset.pad_sequences)

            if self.include_count_features:
                self.tf_dataset[split] = self.tf_dataset[split]\
                    .map(RTDataset.convert_inputs_to_dict)\
                    .map(RTDataset.generate_single_counts)\
                    .map(RTDataset.generate_di_counts)


            self.tf_dataset[split] = self.tf_dataset[split].batch(self.batch_size)


    def get_split_targets(self, split='val'):
        if split not in self.indicies_dict.keys():
            raise ValueError('requested split does not exist, availabe splits are: ' + list(self.indicies_dict.keys()))

        return self.targets[self.indicies_dict[split]]

    def denormalize_targets(self, targets):
        return targets * RTDataset.TARGETS_STD + RTDataset.TARGETS_MEAN

    @staticmethod
    def pad_sequences(seq, target):
        pad_len = tf.abs(RTDataset.SEQ_LENGTH - tf.size(seq))
        paddings = tf.concat([[0], [pad_len]], axis=0)
        seq = tf.pad(seq, [paddings], "CONSTANT")
        seq.set_shape([RTDataset.SEQ_LENGTH])
        return seq, target

    @staticmethod
    def normalize_target(seq, target):

        target = tf.math.divide(tf.math.subtract(target, RTDataset.TARGETS_MEAN), RTDataset.TARGETS_STD)

        return seq, target

    @staticmethod
    def split_sequence(seq, target):
        splitted_seq = tf.strings.bytes_split(seq)

        return splitted_seq, target

    '''
    
    if more than one input is added, inputs are added to a python dict, the following methods assume that
    
    '''

    @staticmethod
    def convert_inputs_to_dict(seq, target):
        return {'seq': seq}, target

    @staticmethod
    def generate_single_counts(inputs, target):
        inputs["counts"] = tf.map_fn(lambda x: RTDataset.ATOM_TABLE.lookup(x), inputs["seq"])
        inputs["counts"] = tf.map_fn(lambda x: tf.strings.split(x, sep='_'), inputs["counts"])
        inputs["counts"] = tf.strings.to_number(inputs["counts"])
        inputs["counts"].set_shape([RTDataset.SEQ_LENGTH, 5])

        return inputs, target

    @staticmethod
    def generate_di_counts(inputs, target):
        # add every two neighboring elements without overlap [0 0 1 1 2 2 .... pad_length/2 pad_length/2]
        segments_to_add = [i // 2 for i in range(RTDataset.SEQ_LENGTH)]
        inputs["di_counts"] = tf.math.segment_sum(inputs["counts"], tf.constant(segments_to_add))
        inputs["di_counts"].set_shape([RTDataset.SEQ_LENGTH // 2, 5])

        return inputs, target

    def get_tf_dataset(self, split=None):
        if split in self.tf_dataset.keys():
            return self.tf_dataset[split]
        return self.tf_dataset
