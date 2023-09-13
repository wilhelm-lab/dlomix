import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit


class PrecursorChargeStateDataset:

    def __init__(self, classification_type="multi_class", model_type="embedding", charge_states=None,
                 dir_path='data/', file_type='.parquet',
                 columns_to_keep=None, test_ratio=0.1):

        """ CHECK ALL INPUTS """

        # check if classification_type is valid
        if columns_to_keep is None:
            columns_to_keep = ['modified_sequence', 'precursor_charge', 'precursor_intensity']
        if charge_states is None:
            charge_states = [1, 2, 3, 4, 5, 6]
        if isinstance(classification_type, str):
            if classification_type not in ["multi_class", "multi_label"]:
                raise ValueError("classification_type must be either 'multi_class' or 'multi_label'.")
            else:
                classification_type = classification_type.lower()
        else:
            raise TypeError("classification_type must be a string.")

        # check if model_type is valid
        if isinstance(model_type, str):
            if model_type not in ["embedding", "conv2d", "prosit"]:
                raise ValueError("model_type must be 'embedding', 'conv2d', 'prosit'.")
            else:
                model_type = model_type.lower()
        else:
            raise TypeError("model_type must be a string.")

        # check if classification_type and model_type are compatible
        if classification_type == "multi_class" and model_type not in ["embedding", "conv2d", "prosit"]:
            raise ValueError("classification_type and model_type are not compatible.")
        elif classification_type in ["multi_label", "multi_head"] and model_type not in ["embedding"]:
            raise ValueError("classification_type and model_type are not compatible.")

        # check if charge states correct:
        if isinstance(charge_states, list):
            if not all(isinstance(item, int) for item in charge_states):
                raise ValueError("charge_states must be a list of integers.")
        else:
            raise TypeError("charge_states must be a list.")

        # check dir_path
        if isinstance(dir_path, str):
            if not os.path.isdir(dir_path):
                raise ValueError("dir_path must be a valid directory. Is not: {}".format(dir_path))
        else:
            raise TypeError("dir_path must be a string.")

        # check file_type
        if isinstance(file_type, str):
            if not file_type.startswith("."):
                file_type = "." + file_type
        else:
            raise TypeError("file_type must be a string.")

        # check columns_to_keep
        if isinstance(columns_to_keep, list):
            if not all(isinstance(item, str) for item in columns_to_keep):
                raise ValueError(
                    "columns_to_keep must be a list of strings. In Order: 'modified_sequence', 'precursor_charge', "
                    "'precursor_intensity'.")
        else:
            raise TypeError("columns_to_keep must be a list.")

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

        def combine_files_into_df(directory_path='data/', file_types=None, column_mapping=None):
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
            return df

        '''
        Drop all rows with NaN values in a specific column
        Default: drop na from precursor_intensity column
        '''

        def drop_na(df, column='precursor_intensity'):
            df = df[df[column].notna()]
            print(f"Step 2/12 complete. Dropped rows with NaN for intensities.")
            return df

        '''
        Keep only desired charge entires
        Default: keep charges 1-6
        '''

        def keep_desired_charges(df, charge_list=self.charge_states):
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
            :return: list
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

        '''Combine unique sequences and aggregate their precursor_charges and intensity in order to later select the 
        most abundant charge state per sequence.'''

        def aggregate_sequences(df):
            df = df.groupby("modified_sequence")[["precursor_charge", "precursor_intensity"]].agg(list).reset_index()
            print(f"Step 4/12 complete. Aggregated all sequences to unique sequences.")
            return df

        '''
        Normalize precursor intensities for aggregated sequences
        '''

        def normalize_precursor_intensities(df_charge_list, df_intensity_list):
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

        '''
        Select most abundand charge state per unique sequence according to the normalized precursor intensity
        '''

        def get_most_abundant(df_charge_list, df_intensity_list, distributions=False):
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

        '''
        One-Hot encode most abundand charge state
        input: df with "most_abundance_charge" column
        output: new column "most_abundant_charge_vector" containing one-hot encoded vector
        '''

        def one_hot_encode_charge(df, charge_list=self.charge_states):
            df['most_abundant_charge_vector'] = df['most_abundant_charge'].apply(
                lambda x: [1 if x == i else 0 for i in charge_list])
            return df

        '''
        Applying normalization, selecting most abundant charge state and one-hot encoding
        '''

        def normalize_and_select_most_abundant(df):
            df['normalized'] = df.apply(
                lambda x: normalize_precursor_intensities(x["precursor_charge"], x["precursor_intensity"]), axis=1)
            df['pre_normalization'] = df.apply(
                lambda x: get_most_abundant(x["precursor_charge"], x["precursor_intensity"], True), axis=1)
            df['most_abundant_charge'] = df['normalized'].apply(lambda x: max(x, key=x.get))
            df = one_hot_encode_charge(df)
            print(
                f"Step 8/12 complete. Applied normalization, selected most abundant charge state and one-hot encoded "
                f"it.")
            return df

        '''
        get topK charge states for each sequence according to the normalized precursor intensity

        input: df with "normalized" column
        output: new column "topK_charge_states" containing list of topK charge states

        default: k=2
        '''

        def get_topK_charge_states(df, k=2):
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

        '''
        Remove sequences of specific length represented less than a certain number of times

        input: df containig "modified_sequence" column, representation_threshold
        output: 
        - df containing only sequence legths represented more than representation_threshold times
        - padding_length
        default: representation_threshold = 100

        Calculate the sequence lengths and their counts
        Filter out sequences with counts below the threshold
        Filter the original DataFrame based on sequence length
        Drop the temporary column
        '''

        def remove_rare_sequence_lengths(df, representation_threshold=100):
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

        '''
        Encode all occuring charge states per unique sequence in a binary vector

        input: df containing "precursor_charge" column output: df containing an additional "charge_state_vector" 
        column encoding all occuring charge states per unique sequence in a binary vector'''

        def encode_charge_states(df):
            df['charge_state_vector'] = df['precursor_charge'].apply(
                lambda x: [1 if i in x else 0 for i in range(1, 7)])
            print(f"Step 9/12 complete. Encoded all occuring charge states per unique sequence in a binary vector.")
            return df

        '''
        Checks if a vector contains only continous charge states e.g. [1,1,1,0,0,0]
        Flase if a vector contains skipped charges e.g. [1,0,0,0,0,1]

        input: charge_state_vector
        output: True if no charge state is skipped, False if a charge state is skipped
        '''

        def has_skipped_charges(charge_state_vector):
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

        '''
        Filter out all sequences where has_skipped_charges() returns True

        input: df containing "charge_state_vector" column
        output: df containing only sequences where has_skipped_charges() returns False
        '''

        def filter_skipped_charges(df):
            return df[df['charge_state_vector'].apply(lambda x: not has_skipped_charges(x))]

        '''
        Removes sequences with skipped charges that occur less than a certain number of times

        input: df containing "charge_state_vector" column, cutoff
        output: df containing only sequences with skipped charges that occur more than cutoff times
        default: cutoff = 1000
        '''

        def skip_charges_for_occurrences(df, cutoff=1000):
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

        """
        Encodes the 'modified_sequence' column in a DataFrame and adds a new column 'modified_sequence_vector'.

        input: df containing "modified_sequence" column, vocabulary, padding_length
        output: df containing "modified_sequence_vector" column with padded and encoded sequences

        defaults: padding_length = 50
        """

        def sequence_encoder(df, padding_length=50, vocabulary=None):

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

        self.dir_path = dir_path
        self.file_type = file_type

        self.charge_states = charge_states
        self.num_classes = len(self.charge_states)

        self.classification_types = ['multi_class', 'multi_label']
        self.classification_type = classification_type

        self.model_types = ['embedding', 'conv2d', 'prosit']
        self.model_type = model_type

        self.df = combine_files_into_df(dir_path, file_type)
        self.df = drop_na(self.df, 'precursor_intensity')
        self.df = keep_desired_charges(self.df)
        self.df = aggregate_sequences(self.df)
        self.df, self.padding_length = remove_rare_sequence_lengths(self.df)
        self.vocabulary, self.voc_len = complete_vocabulary(self.df)
        self.df = sequence_encoder(self.df, self.padding_length, self.vocabulary)
        self.df = normalize_and_select_most_abundant(self.df)
        self.df = encode_charge_states(self.df)
        self.df = skip_charges_for_occurrences(self.df)
        self.df = get_topK_charge_states(self.df)
        print(f"Step 12/12 complete. Generated dataset with {len(self.df)} sequences.")
        if self.classification_type == 'multi_class':
            self.df = self.df[['modified_sequence_vector', 'most_abundant_charge_vector', 'top_2_charge_states']]
        elif self.classification_type == 'multi_label':
            self.df = self.df[['modified_sequence_vector', 'charge_state_vector', 'top_2_charge_states']]
        else:
            raise ValueError("classification_type must be one of the following: 'multi_class', 'multi_label'")

        if self.classification_type == "multi_class":
            if model_type == "embedding":
                self.data_type = "tensor"
            elif model_type == "conv2d":
                self.data_type = "2d_tensor"
            elif model_type == "prosit":
                self.data_type = "tensor"
            else:
                raise ValueError("model_type must be one of the following: 'embedding', 'conv2d', 'prosit'")
        elif self.classification_type == "multi_label":
            self.data_type = "tensor_multi_label"
        else:
            raise ValueError("classification_type must be one of the following: 'multi_class', 'multi_label'")

        self.validation_ratio = 0.2
        self.test_mode = True
        if test_ratio > 0:
            self.test_ratio = test_ratio
            self.df_test = self.df.sample(frac=self.test_ratio)
            self.test_mode = True
        else:
            self.df_test = pd.DataFrame()
            self.test_mode = False
        self.training_validation_df = self.df.drop(self.df_test.index)
        self.training_validation_split = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_ratio)

        def create_training_validation_split(df=self.training_validation_df, sssplit=self.training_validation_split):
            trainval_ds_embed = np.array(df['modified_sequence_vector'])
            if self.data_type == "tensor_multi_label":
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
            if self.data_type == "tensor_multi_label":
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

        self.df_train, self.df_val = create_training_validation_split(self.training_validation_df,
                                                                      self.training_validation_split)

        def to_array(df, multi_label=False):
            # print(df.head(4))
            if multi_label:
                label = [np.array(x) for x in df['charge_state_vector']]
                data = [np.array(x) for x in df['modified_sequence_vector']]
            else:
                label = [np.array(x) for x in df['most_abundant_charge_vector']]
                data = [np.array(x) for x in df['modified_sequence_vector']]
            return label, data

        def to_tensor(df, multi_label=False):
            label, data = to_array(df, multi_label)
            label = tf.convert_to_tensor(label)
            data = tf.convert_to_tensor(data)
            return label, data

        def to_2d_tensor(df):
            label, data = to_array(df)
            label = tf.convert_to_tensor(label)
            data = [np.reshape(np.array(x), (1, self.padding_length, 1)) for x in data]
            return label, data

        if self.data_type == "array":
            self.test_label, self.test_data = to_array(self.df_test)
            self.train_label, self.train_data = to_array(self.df_train)
            self.val_label, self.val_data = to_array(self.df_val)
        elif self.data_type == "tensor":
            self.test_label, self.test_data = to_tensor(self.df_test)
            self.train_label, self.train_data = to_tensor(self.df_train)
            self.val_label, self.val_data = to_tensor(self.df_val)
        elif self.data_type == "tensor_multi_label":
            self.test_label, self.test_data = to_tensor(self.df_test, True)
            self.train_label, self.train_data = to_tensor(self.df_train, True)
            self.val_label, self.val_data = to_tensor(self.df_val, True)
        elif self.data_type == "2d_tensor":
            self.test_label, self.test_data = to_2d_tensor(self.df_test)
            self.train_label, self.train_data = to_2d_tensor(self.df_train)
            self.val_label, self.val_data = to_2d_tensor(self.df_val)
