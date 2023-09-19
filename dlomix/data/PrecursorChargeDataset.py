import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit


class PrecursorChargeStateDataset:
    def __init__(self, classification_type="multi_class", charge_states=None,
                 dir_path='data/',
                 columns_to_keep=None, test_ratio=0.1, validation_ratio=0.2):

        """
        Class for creating a dataset for the precursor charge state prediction task.
        @param classification_type: str, either "multi_class" or "multi_label"
        @param charge_states: list, list of charge states to be used for the dataset
        @param dir_path: str, path to the directory containing the files
        @param columns_to_keep: list, list of columns to be kept from the files
        @param test_ratio: float, ratio of the test set
        @param validation_ratio: float, ratio of the validation set
        """

        # set defaults if None
        if columns_to_keep is None:
            columns_to_keep = ['modified_sequence', 'precursor_charge', 'precursor_intensity']
        if charge_states is None:
            charge_states = [1, 2, 3, 4, 5, 6]

        # check if classification_type correct:
        if isinstance(classification_type, str):
            if classification_type not in ["multi_class", "multi_label"]:
                raise ValueError(
                    f"Error: {classification_type} was set. classification_type must be either 'multi_class' or "
                    f"'multi_label'.")
            else:
                classification_type = classification_type.lower()
        else:
            raise TypeError(f"Error: {classification_type}. classification_type must be a string.")
        if classification_type == "multi_label":
            if len(charge_states) < 2:
                raise ValueError(
                    f"Error: {charge_states}. For multi_label classification at least two charge states must be "
                    f"provided.")

        # check if charge states correct:
        if isinstance(charge_states, list):
            if not all(isinstance(item, int) for item in charge_states):
                raise ValueError(f"Error: {charge_states}. Charge_states must be a list of integers.")
        else:
            raise TypeError(f"Error: {charge_states}. Charge_states must be a list.")

        # check dir_path
        if isinstance(dir_path, str):
            if not os.path.isdir(dir_path):
                raise ValueError(f"Error: {dir_path}. dir_path must be a valid directory.")
        else:
            raise TypeError(f"Error: {dir_path}. dir_path must be a string.")

        # check columns_to_keep
        if isinstance(columns_to_keep, list):
            if not all(isinstance(item, str) for item in columns_to_keep):
                raise ValueError(
                    f"Error: {columns_to_keep}. columns_to_keep must be a list of strings. In Order: "
                    f"'modified_sequence', 'precursor_charge',"
                    "'precursor_intensity'.")
        else:
            raise TypeError(f"Error: {columns_to_keep}.columns_to_keep must be a list.")

        '''
        File import for .parquet, .tsv and .csv files
        At the moment a mix of .parquet, .tsv and .csv files will also be combined into one dataframe.

        Providing a column mapping is optional. If no column mapping is provided, the default mapping will be used 

        input: 
        dir_path: path to the directory containing the files
        file_types: list of file types to be imported
        column_mapping: dictionary with column names as keys and the corresponding column names in the files as values

        defaults: dir_path: 'data/' file_types: ['.parquet', '.tsv', '.csv'] columns: {modified_sequence: 
        'modified_sequence', precursor_charge: 'precursor_charge', precursor_intensity: 'precursor_intensity'}

        output: 
        df: dataframe containing the imported data
        '''

        # Data path
        self.dir_path = dir_path

        # resulting dataframe
        self.df = None

        # attributes (needed for model class: get_attributes())
        self.charge_states = charge_states
        self.num_classes = len(self.charge_states)
        self.padding_length = None
        self.vocabulary = None
        self.voc_len = None
        self.validation_ratio = validation_ratio
        self.test_mode = True
        self.classification_types = ['multi_class', 'multi_label']
        self.classification_type = classification_type

        # data
        self.df_test = None
        self.training_validation_df = None
        self.training_validation_split = None
        self.df_train, self.df_val = None, None
        self.test_label, self.test_data = None, None
        self.train_label, self.train_data = None, None
        self.val_label, self.val_data = None, None

        # Preprocessing
        # Step 0/12
        print("Step 0/12 complete. Initializing PrecursorChargeStateDataset.")

        # combine all files in the directory into one dataframe
        # Step 1/12
        self.df = combine_files_into_df(dir_path)

        # drop all rows with NaN values in the precursor_intensity column
        # Step 2/12
        self.df = drop_na(self.df, 'precursor_intensity')

        # keep only desired charge states
        # Step 3/12
        self.df = keep_desired_charges(self.df, self.charge_states)

        # aggregate all sequences to unique sequences
        # Step 4/12
        self.df = aggregate_sequences(self.df)

        # remove sequences of specific length represented less than a certain number of times
        # Step 5/12
        self.df, self.padding_length = remove_rare_sequence_lengths(self.df)

        # filter out all sequences where has_skipped_charges() returns True
        # Step 6/12
        self.vocabulary, self.voc_len = complete_vocabulary(self.df)

        # encode modified sequences according to the vocabulary
        # Step 7/12
        self.df = sequence_encoder(self.df, self.padding_length, self.vocabulary)

        # normalize for precursor intensities and select the most abundant charge state
        # Step 8/12
        self.df = normalize_and_select_most_abundant(self.df, self.charge_states)

        # encode all occuring charge states per unique sequence in a binary vector
        # Step 9/12
        self.df = encode_charge_states(self.df, self.charge_states)

        # filter out all sequences where has_skipped_charges() returns True (if > 1000 occurrences)
        # Step 10/12
        self.df = skip_charges_for_occurrences(self.df)

        # get topK charge states for each sequence
        # Step 11/12
        self.df = get_topK_charge_states(self.df)

        # Step 12/12
        print(f"Step 12/12 complete. Generated dataset with {len(self.df)} sequences.")

        # keep only the needed columns after preprocessing
        if self.classification_type == 'multi_class':
            self.df = self.df[['modified_sequence_vector', 'most_abundant_charge_vector', 'top_2_charge_states']]
        elif self.classification_type == 'multi_label':
            self.df = self.df[['modified_sequence_vector', 'charge_state_vector', 'top_2_charge_states']]
        else:
            raise ValueError(
                f"Error: {classification_type}. classification_type must be one of the following: 'multi_class', 'multi_label'")

        # split the dataset into training+validation (trainval) and test set
        if test_ratio > 0:
            self.test_ratio = test_ratio
            self.df_test = self.df.sample(frac=self.test_ratio)
            self.test_mode = True
        else:
            self.df_test = pd.DataFrame()
            self.test_mode = False
        self.training_validation_df = self.df.drop(self.df_test.index)
        self.training_validation_split = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_ratio)

        # create training and validation dataset from remaining training + validation (trainval)  split
        self.df_train, self.df_val = create_training_validation_split(self.training_validation_df,
                                                                      self.training_validation_split,
                                                                      self.classification_type)

        if self.classification_type == "multi_class":
            self.test_label, self.test_data = to_tensor(self.df_test)
            self.train_label, self.train_data = to_tensor(self.df_train)
            self.val_label, self.val_data = to_tensor(self.df_val)
        elif self.classification_type == "multi_label":
            self.test_label, self.test_data = to_tensor(self.df_test, True)
            self.train_label, self.train_data = to_tensor(self.df_train, True)
            self.val_label, self.val_data = to_tensor(self.df_val, True)

    # getter for Model Class
    def get_attributes(self):
        """
        Returns all "attributes" of the class
        @return: dataset_dict: dict, dictionary containing all attributes
        """
        dataset_dict = dict()
        dataset_dict['charge_states'] = self.charge_states
        dataset_dict['num_classes'] = self.num_classes
        dataset_dict['padding_length'] = self.padding_length
        dataset_dict['vocabulary'] = self.vocabulary
        dataset_dict['voc_len'] = self.voc_len
        dataset_dict['validation_ratio'] = self.validation_ratio
        dataset_dict['test_mode'] = self.test_mode
        dataset_dict['classification_types'] = self.classification_types
        dataset_dict['classification_type'] = self.classification_type
        return dataset_dict


def combine_files_into_df(directory_path='data/', file_types=None, column_mapping=None):
    """
    Combines all files in a directory into one DataFrame
    @param directory_path: str, path to the directory containing the files
    @param file_types: list, list of file types to be imported
    @param column_mapping: dictionary with column names as keys and the corresponding column names in the files as values
    @return: df: DataFrame
    """
    if file_types is None:
        file_types = ['.parquet', '.tsv', '.csv']
    dfs = []

    if column_mapping is None:
        column_mapping = {
            'modified_sequence': 'modified_sequence',
            'precursor_charge': 'precursor_charge',
            'precursor_intensity': 'precursor_intensity'
        }

    for file in os.listdir(directory_path):
        if any(file.endswith(file_format) for file_format in file_types):
            file_path = os.path.join(directory_path, file)

            if file.endswith('.parquet'):
                df = pd.read_parquet(file_path, engine='fastparquet')
            elif file.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            elif file.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f'File type {file_type} not supported')

            # Rename columns based on the provided mapping
            df = df.rename(columns=column_mapping)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Step 1/12 complete. Combined {len(dfs)} files into one DataFrame.")
    return df


def drop_na(df, column='precursor_intensity'):
    """
    Drop all rows with NaN values in a specific column
    Default: drop na from precursor_intensity column
    @param df: DataFrame
    @param column: column to drop NaN values from
    @return: df: DataFrame
    """

    df = df[df[column].notna()]
    print(f"Step 2/12 complete. Dropped rows with NaN for intensities.")
    return df


def keep_desired_charges(df, charge_list=None):
    """
    Keep only desired charge states
    Default: keep charge states 1-6
    @param df: DataFrame
    @param charge_list: list of charge states to be kept
    """
    if charge_list is None:
        charge_list = [1, 2, 3, 4, 5, 6]
    df = df[df['precursor_charge'].isin(charge_list)]
    print(f"Step 3/12 complete. Removed charge states not in {charge_list}.")
    return df


'''
Find all UNIMOD annotations and add them to the vocabulary
(The length of the vocabulary +1 is used later for the embedding layer)
'''


def complete_vocabulary(df):
    """
    Completes the vocabulary with all the possible amino acids and modifications
    @param df: DataFrame
    @return: vocabulary: list, list of all amino acids and modifications
    @return: vocab_len: int, length of the vocabulary
    """
    vocabulary = []
    vocabulary += list('XACDEFGHIKLMNPQRSTVWY')
    annotations = re.findall(r'(\w\[UNIMOD:\d+])', ' '.join(df['modified_sequence']))
    for item in annotations:
        if item not in vocabulary:
            vocabulary.append(item)
    vocab_len = len(vocabulary)
    print(f"Step 6/12 complete. Completed vocabulary with {vocab_len} entries.")
    return vocabulary, vocab_len


def aggregate_sequences(df):
    """
    Aggregates all sequences to unique sequences
    @param df: DataFrame
    @return: df: DataFrame
    """
    df = df.groupby("modified_sequence")[["precursor_charge", "precursor_intensity"]].agg(list).reset_index()
    print(f"Step 4/12 complete. Aggregated all sequences to unique sequences.")
    return df


def normalize_precursor_intensities(df_charge_list, df_intensity_list):
    """
    Normalizes the precursor intensities based on the most abundant precursor intensity
    @param df_charge_list: list, list of charge states
    @param df_intensity_list: list, list of precursor intensities
    @return: charge_dict: dict, dictionary with charge states as keys and normalized intensities as values
    """
    # Get the index of the most abundant precursor intensity
    charge_dict = dict()
    for index, i in enumerate(df_charge_list):
        charge_dict[i] = []
        charge_dict[i].append(df_intensity_list[index])

    # Normalize the precursor intensity based on the most abundant precursor intensity
    for key, value in charge_dict.items():
        if len(value) > 1:
            charge_dict[key] = sum(value) - min(value) / (max(value) - min(value))

    # convert list of one float to float values
    charge_dict = {key: value[0] for key, value in charge_dict.items()}
    return charge_dict


def get_most_abundant(df_charge_list, df_intensity_list, distributions=False):
    """
    Get the most abundant charge state
    @param df_charge_list: list, list of charge states
    @param df_intensity_list: list, list of precursor intensities
    @param distributions: bool, if True returns a dictionary with all charge states and their intensities
    @return: charge_dict: dict, dictionary with charge states as keys and intensities as values
    """
    charge_dict = dict()
    for index, i in enumerate(df_charge_list):
        if i not in charge_dict:
            charge_dict[i] = df_intensity_list[index]
        else:
            charge_dict[i] += df_intensity_list[index]
    if distributions:
        return charge_dict
    else:
        return max(charge_dict, key=charge_dict.get)


def one_hot_encode_charge(df, charge_list=None):
    """
    One-hot encodes the most abundant charge state
    @param df: DataFrame
    @param charge_list: list, list of charge states
    @return: df: DataFrame
    """
    if charge_list is None:
        charge_list = [1, 2, 3, 4, 5, 6]
    df['most_abundant_charge_vector'] = df['most_abundant_charge'].apply(
        lambda x: [1 if x == i else 0 for i in charge_list])
    return df


def normalize_and_select_most_abundant(df, charge_list=None):
    """
    Normalizes the precursor intensities and selects the most abundant charge state
    @param df: DataFrame
    @param charge_list: list, list of charge states
    @return: df: DataFrame
    """

    if charge_list is None:
        charge_list = [1, 2, 3, 4, 5, 6]
    df['normalized'] = df.apply(
        lambda x: normalize_precursor_intensities(x["precursor_charge"], x["precursor_intensity"]), axis=1)
    df['pre_normalization'] = df.apply(
        lambda x: get_most_abundant(x["precursor_charge"], x["precursor_intensity"], True), axis=1)
    df['most_abundant_charge'] = df['normalized'].apply(lambda x: max(x, key=x.get))
    df = one_hot_encode_charge(df, charge_list)
    print(
        f"Step 8/12 complete. Applied normalization, selected most abundant charge state and one-hot encoded "
        f"it.")
    return df


def get_topK_charge_states(df, k=2):
    """
    get topK charge states for each sequence according to the normalized precursor intensity
    Default: k=2
    @param df: DataFrame
    @param k: int, number of top charge states to be selected
    @return: df: DataFrame
    """

    def get_topK(label_dict):
        allowed_keys = list()
        sorted_values = sorted(label_dict.values(), reverse=True)
        for i in sorted_values:
            for key, value in label_dict.items():
                if i == value and len(allowed_keys) <= k - 1:
                    allowed_keys.append(key)
        return allowed_keys

    df[f'top_{k}_charge_states'] = df['normalized'].apply(get_topK)
    print(f"Step 11/12 complete. Selected top {k} charge states per sequence.")
    return df


def remove_rare_sequence_lengths(df, representation_threshold=100):
    """
    Remove sequences of specific length represented less than a certain number of times

    input: df containing "modified_sequence" column, representation_threshold
    output:
    - df containing only sequence lengths represented more than representation_threshold times
    - padding_length
    default: representation_threshold = 100

    Calculate the sequence lengths and their counts
    Filter out sequences with counts below the threshold
    Filter the original DataFrame based on sequence length
    Drop the temporary column

    @param df: DataFrame
    @param representation_threshold: int, threshold for the number of times a sequence length must be represented
    @return: df: DataFrame
    @return: padding_length: int, length of the longest sequence
    """
    before_len = len(df)
    df['sequence_length_prepadding'] = df['modified_sequence'].apply(len)
    len_counts = df['sequence_length_prepadding'].value_counts().reset_index()
    len_counts.columns = ['seq_len', 'count']
    filtered_lengths = len_counts[len_counts['count'] >= representation_threshold]['seq_len']
    df = df[df['sequence_length_prepadding'].isin(filtered_lengths)].copy()
    padding_length = df['sequence_length_prepadding'].max()
    df = df[df['sequence_length_prepadding'].isin(filtered_lengths)]
    after_len = len(df)
    print(
        f"Step 5/12 complete. Removed {before_len - after_len} of {before_len} sequences if sequence-length "
        f"is represented less than {representation_threshold} times.")
    return df, padding_length


def encode_charge_states(df, charge_states=None):
    """
    Encode all occuring charge states per unique sequence in a binary vector

    input: df containing "precursor_charge" column output: df containing an additional "charge_state_vector"
    column encoding all occuring charge states per unique sequence in a binary vector
    @param df: DataFrame
    @return: df: DataFrame
    """
    df['charge_state_vector'] = df['precursor_charge'].apply(
        lambda x: [1 if i in x else 0 for i in range(charge_states[0], charge_states[-1]+1)])
    print(f"Step 9/12 complete. Encoded all occuring charge states per unique sequence in a binary vector.")
    return df


def has_skipped_charges(charge_state_vector):
    """
    Checks if a vector contains only continuous charge states e.g. [1,1,1,0,0,0]
    False if a vector contains skipped charges e.g. [1,0,0,0,0,1]

    input: charge_state_vector
    output: True if no charge state is skipped, False if a charge state is skipped
    @param charge_state_vector: list, list of charge states
    @return: bool
    """
    was_found = False
    was_concluded = False
    for i in charge_state_vector:
        if i == 1 and not was_found:
            was_found = True
        if i == 0 and was_found:
            was_concluded = True
        if i == 1 and was_concluded:
            return True
    return False


def _filter_skipped_charges(df):
    """
    Filter out all sequences where has_skipped_charges() returns True

    (!) does not use the cutoff of 1000 occurrences

    input: df containing "charge_state_vector" column
    output: df containing only sequences where has_skipped_charges() returns False
    @param df: DataFrame
    @return: df: DataFrame
    """
    return df[df['charge_state_vector'].apply(lambda x: not has_skipped_charges(x))]


def skip_charges_for_occurrences(df, cutoff=1000):
    """
    Removes sequences with skipped charges that occur less than a certain number of times

    input: df containing "charge_state_vector" column, cutoff
    output: df containing only sequences with skipped charges that occur more than cutoff times
    default: cutoff = 1000

    @param df: DataFrame
    @param cutoff: int, threshold for the number of times a sequence with skipped charges must occur
    @return: df: DataFrame
    """
    list_k = []
    list_v = []
    drop_out_index = []
    for index, i in enumerate(df['charge_state_vector'].value_counts()):
        list_k.append(df['charge_state_vector'].value_counts().index[index])
        list_v.append(i)
        if has_skipped_charges(df['charge_state_vector'].value_counts().index[index]) and list_v[
            index] < cutoff:
            drop_out_index.append(index)

    drop_out_list = []
    for i in drop_out_index:
        drop_out_list.append(list_k[i])
    df_out = df[~df['charge_state_vector'].isin(drop_out_list)]
    print(
        f"Step 10/12 complete. Removed {len(df) - len(df_out)} of {len(df)} sequences if unique charge state "
        f"distribution is represented less than {cutoff} times.")
    return df_out


def sequence_encoder(df, padding_length=50, vocabulary=None):
    """
    Encodes the 'modified_sequence' column in a DataFrame and adds a new column 'modified_sequence_vector'.

    input: df containing "modified_sequence" column, vocabulary, padding_length
    output: df containing "modified_sequence_vector" column with padded and encoded sequences

    defaults: padding_length = 50
    @param df: DataFrame
    @param padding_length: int, length of the longest sequence
    @param vocabulary: list, list of all amino acids and modifications
    @return: df: DataFrame
    """

    if 'modified_sequence' not in df.columns:
        raise ValueError("DataFrame must contain a 'modified_sequence' column.")

    aa_dictionary = {aa: index for index, aa in enumerate(vocabulary)}

    def encode_sequence(sequence):
        pattern = r'[A-Z]\[[^\]]*\]|.'
        result = [match for match in re.findall(pattern, sequence)]
        result += ['X'] * (padding_length - len(result))
        return [aa_dictionary.get(aa, aa_dictionary['X']) for aa in result]

    df['modified_sequence_vector'] = df['modified_sequence'].apply(encode_sequence)
    print(f"Step 7/12 complete. Encoded all sequences.")
    return df


def create_training_validation_split(df, sssplit, data_type=None):
    """
    Creates a training and validation split for a DataFrame
    @param df: DataFrame
    @param sssplit: StratifiedShuffleSplit
    @return: df_train: DataFrame
    @return: df_val: DataFrame
    """
    trainval_ds_embed = np.array(df['modified_sequence_vector'])
    if data_type == "multi_label":
        trainval_labels_embed = np.array(df['charge_state_vector'])
    else:
        trainval_labels_embed = np.array(df['most_abundant_charge_vector'])
    # Perform the split train and val
    train_indicies_embed, val_indicies_embed = next(sssplit.split(trainval_ds_embed, trainval_labels_embed))
    # Distribution
    train_ds_embed, train_labels_embed = trainval_ds_embed[train_indicies_embed], trainval_labels_embed[
        train_indicies_embed]
    val_ds_embed, val_labels_embed = trainval_ds_embed[val_indicies_embed], trainval_labels_embed[
        val_indicies_embed]
    # create two dataframes for training and validation
    if data_type == "multi_label":
        df_train = pd.DataFrame(
            {'modified_sequence_vector': train_ds_embed, 'charge_state_vector': train_labels_embed})
        df_val = pd.DataFrame(
            {'modified_sequence_vector': val_ds_embed, 'charge_state_vector': val_labels_embed})
    else:
        df_train = pd.DataFrame(
            {'modified_sequence_vector': train_ds_embed, 'most_abundant_charge_vector': train_labels_embed})
        df_val = pd.DataFrame(
            {'modified_sequence_vector': val_ds_embed, 'most_abundant_charge_vector': val_labels_embed})
    return df_train, df_val


def to_array(df, multi_label=False):
    """
    Converts a DataFrame to a numpy array

    (!) Casting to a list of np.arrays is used to make it always compatible with tf.convert_to_tensor as a
    "one shoe fits all"-solution

    @param df: DataFrame
    @param multi_label: bool, if True returns a numpy array for multi-label classification
    @return: label: np.array
    @return: data: np.array
    """
    if multi_label:
        label = [np.array(x) for x in df['charge_state_vector']]
        data = [np.array(x) for x in df['modified_sequence_vector']]
    else:
        label = [np.array(x) for x in df['most_abundant_charge_vector']]
        data = [np.array(x) for x in df['modified_sequence_vector']]
    return label, data


def to_tensor(df, multi_label=False):
    """
    Converts a DataFrame to a tensor
    @param df: DataFrame
    @param multi_label: bool, if True returns a tensor for multi-label classification
    @return: label: tf.tensor
    @return: data: tf.tensor
    """
    label, data = to_array(df, multi_label)
    label = tf.convert_to_tensor(label)
    data = tf.convert_to_tensor(data)
    return label, data
