# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import re

from dlomix.detectability_model_constants import CLASSES_LABELS, alphabet, aa_to_int_dict, int_to_aa_dict, padding_char

class detectability_dataset:
    r"""A dataset class for detectability prediction tasks. It initialize a dataset object wrapping tf.Dataset and some relevant preprocessing steps.

    Parameters
    -----------
    data_source : str, tuple of two numpy.ndarray, numpy.ndarray, optional
        source can be a tuple of two arrays (sequences, classes), single array (sequences), useful for test data, or a str with a file path to a csv file. Defaults to None.
    protein_data : str, numpy.ndarray, optional
        the protein data can be a single array (protein name, id), or a str with the name of the column with the protein information if a file path to a csv file was provided. Defaults to 'proteins'.    
    sep : str, optional
        separator to be used if the data source is a CSV file. Defaults to ",".
    sequence_col :  str, optional
        name of the column containing the sequences in the provided CSV. Defaults to "sequences".
    classes_col : str, optional
        name of the column containing the classes. Defaults to "classes".
    split_on_protein : bool, optional
        a boolean whether the dataset is going to be split based on proteins (all peptides belonging to a particular protein). Protein_data must be provided. Defaults to False.
    seq_length : int, optional
        the sequence length to be used, where all sequences will be padded to this length, longer sequences will be removed and not truncated. Defaults to 40.
    batch_size : int, optional
        the batch size to be used for consuming the dataset in training a model. Defaults to 32.
    val_ratio : int, optional
        a fraction to determine the size of the validation data. Default to 0.1 (10%). 
    test_ratio : int, optional
        a fraction to determine the size of the test data. Default to 0.2 (20%). 
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
    
    def __init__(
        self,
        data_source = None,
        protein_data = "proteins",
        sep = ",",
        sequence_col = "sequences",
        classes_col = "classes",
        split_on_protein = False,
        seq_length = 40, 
        batch_size = 64,
        val_ratio = 0.10,
        test_ratio = 0.20,
        seed = 21,
        test = False,
        sample_run = False,
    ):
        super(detectability_dataset, self).__init__()

        np.random.seed(seed)
        self.seed = seed

        self.data_source = data_source
        self.sep = sep
        self.sequence_col = sequence_col.lower()
        self.classes_col = classes_col.lower()
        self.protein_data = protein_data
        self.sample_run = sample_run
        self.seq_length = seq_length
        self.split_on_protein = split_on_protein
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio # New line Naim
        self.testing_mode = test
        self.no_classes = self.testing_mode # modified from no_intensities to no classes
        self.proteins_info = self.split_on_protein

        self.main_split = (
            detectability_dataset.SPLIT_NAMES[2]
            if self.testing_mode
            else detectability_dataset.SPLIT_NAMES[0]
        )
        
        self.sequences = None # ???
        self.classes = None # New line Naim 
        self.proteins = None # New line Naim 
        

        if not self.testing_mode: # Added this conditional
            
            assert val_ratio != 0 and test_ratio != 0, 'If training, ratios for validation and testing must be assigned' 


        # initialize TF Datasets dict
        self.tf_dataset = (
            {self.main_split: None, detectability_dataset.SPLIT_NAMES[1]: None, detectability_dataset.SPLIT_NAMES[2]: None}
            if not self.testing_mode
            else {self.main_split: None}
        )

        self.indicies_dict = (
            {self.main_split: None, detectability_dataset.SPLIT_NAMES[1]: None, detectability_dataset.SPLIT_NAMES[2]: None}
            if not self.testing_mode
            else {self.main_split: None}
        )


        # if data is provided with the constructor call --> load, otherwise --> done
        if self.data_source is not None:
            self.load_data(data = data_source)

    def load_data(self, data):
        """Load data into the dataset object, can be used to load data at a later point after initialization.
        This function triggers the whole pipeline of: data loading, validation (against sequence length), splitting, building TensorFlow dataset objects, and apply preprocessing.

        :param data: a `str` with a file path to csv file
        :return: None
        """
        self.data_source = data

        self._read_data()
        self._validate_remove_long_sequences()
        self._encode_seq_and_targets()
        self._pad_sequences()   
        self._split_data()

    """
    numpy array --> either a tuple or a single array
        - Tuple --> means (sequences, classes)
        - single ndarray --> means sequences only, useful for test dataset
    str --> path to csv file or compressed csv file
    """

    def _read_data(self):
        if isinstance(self.data_source, (tuple, np.ndarray)):
            
            tuple_size_is_two = (len(self.data_source) == 2)
                                   
            if tuple_size_is_two:
                
                tuple_elements_are_ndarray = all(
                    [isinstance(x, np.ndarray) for x in self.data_source]
                )
                
                if tuple_elements_are_ndarray:
                    self.sequences = self.data_source[0]
                    
                    if len(self.data_source) == 2: 
                        self.classes = self.data_source[1]
                        self.no_classes = False 
            
                    if isinstance(self.protein_data, np.ndarray): 
                        self.proteins = self.protein_data
                        self.proteins_info = True
            
            elif isinstance(self.data_source, np.ndarray):
                self.sequences = self.data_source
            
                if isinstance(self.protein_data, np.ndarray): 
                    self.proteins = self.protein_data 
                    self.proteins_info = True

            else:
                raise ValueError(
                    "If a tuple is provided, it has to have a length of 2 and all elements should be numpy arrays. Otherwise it should be a numpy array with the peptide sequences"
                )
                
        elif isinstance(self.data_source, str):
            df = pd.read_csv(self.data_source, sep = self.sep)
            
            # used only for testing with a smaller sample from a csv file
            if self.sample_run:
                df = df.head(detectability_dataset.SAMPLE_RUN_N)

            # lower all column names
            df.columns = [col_name.lower() for col_name in df.columns]

            # retrieve columns from the dataframe
            self.sequences = df[self.sequence_col].values
 
            
            if self.classes_col in df.columns:
                self.classes = df[self.classes_col].values
                self.no_classes = False                 
                     
                
            if isinstance(self.protein_data, str):
                if self.protein_data in df.columns:
                    self.proteins = df[self.protein_data].values
                    self.proteins_info = True
                    
                else:
                    raise ValueError("Provided column name for Protein data does not match witi the name in the csv file") ### CHECK NEW CODE LINE #####

                   
            
            elif isinstance(self.protein_data, np.ndarray): 
                self.proteins = self.protein_data
                self.proteins_info = True
                     
#             """Naim: Check what this line do (???)"""
#             if isinstance(self.classes.iloc[0], str): 
#                 self.classes = self.classes.apply(eval) 
                
            # get numpy arrays with .values() for all inputs and intensities

        else:
            raise ValueError(
                "Data source has to be either a tuple of two numpy arrays, a numpy array or a string path to a csv file."
            )


    def _validate_remove_long_sequences(self) -> None:
        """
        Validate if all sequences are shorter than the padding length, otherwise drop them.
        """
        assert self.sequences.shape[0] > 0, "No sequences in the provided data."
        
        assert sum([detectability_dataset.validate_sequence(x) for x in self.sequences]) == 0,  "There are peptides sequences that contain characters different than the twenty amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y "
        
        # check if count of examples matches for all provided inputs
        lengths = [
            len(self.sequences)]
        
        if not self.no_classes:
            assert isinstance(self.classes, np.ndarray), "No classes in the provided data."
            lengths = lengths + [len(self.classes)] 
            
        if self.split_on_protein or self.proteins_info: 
            assert isinstance(self.proteins, np.ndarray), "No proteins in the provided data. Protein information must be provided or split_on_protein must be set to False"
            lengths = lengths + [len(self.proteins)] 

        assert np.all(
            lengths == np.array(lengths[0])
        ), "Count of examples does not match for sequences and targets."

        limit = self.seq_length
        vectorized_len = np.vectorize(lambda x: len(x))
        mask = vectorized_len(self.sequences) <= limit
        self.sequences = self.sequences[mask]
        
        if not self.no_classes:
            self.classes = self.classes[mask] 
            
        if self.proteins_info:
            self.proteins = self.proteins[mask]

        # once feature columns are introduced, apply the mask to the feature columns (subset the dataframe as well)

    def _pad_sequences(self):
    
        """Padding all sequences."""
        
        self.sequences = pad_sequences(self.sequences, maxlen = self.seq_length, padding = 'post', dtype = 'float32', value = padding_char)
        
    def _encode_seq_and_targets(self):
    
        """One-hot encoding all sequences and classes (if available)."""
    
        self.sequences = np.array([detectability_dataset.encode_seq(x, aa_to_int_dict) for x in self.sequences], dtype=object)
        
        if not self.no_classes:
        
            self.classes = np.expand_dims(tf.keras.utils.to_categorical(self.classes), axis = 1)        
    
    @staticmethod
    def validate_sequence(seq):
        return not bool(re.match(r'[ACDEFGHIKLMNPQRSTVWY]*$', seq))

           
    @staticmethod
    def encode_seq(sequence, char_to_int):
    
        """Function for one-hot encoding sequences."""
        
        sequence = sequence.upper()
        integer_encoded = [char_to_int[char] for char in sequence]
        one_hot_encoded = list()

        for value in integer_encoded:
            encoded_aa = [0 for _ in range(len(alphabet))]
            encoded_aa[value] = 1
            one_hot_encoded.append(np.array(encoded_aa)) # Should I turn it into a numpy array before appending??? CHECK 08 2024

        return np.array(one_hot_encoded)

    @staticmethod
    def decode_seq(seq, int_to_char):
    
        """Function for one-hot decoding sequences."""

        decoded_seq = [int_to_char[np.argmax(coded_aa)] for coded_aa in seq] 

        return decoded_seq
    
    @staticmethod    
    def find_index(array, values):
        
        """Function to find indeces of arrays for a set of values."""
        
        index = np.empty([0], dtype = int)
        
        for value in values:
            
            curr_index = np.where(array == value)[0]
            
            index = np.concatenate([index, curr_index])
            
        return index

    def _split_data(self):
        n = np.arange(len(self.sequences))
        
        if self.val_ratio != 0 and self.test_ratio != 0 and (not self.testing_mode):

            if not self.split_on_protein:
                
                train_val_seqs, test_seqs, train_val_classes, test_classes, train_val_n, test_n =  train_test_split(self.sequences, 
                                                                                                                    self.classes, n,
                                                                                                                    test_size = self.test_ratio,
                                                                                                                    stratify = tf.squeeze(self.classes),
                                                                                                                    random_state = self.seed)
                with tf.device('/CPU:0'):
                    self.tf_dataset['test'] = tf.data.Dataset.from_tensor_slices((test_seqs, test_classes))   
                    self.tf_dataset['test'] = self.tf_dataset['test'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['test'] = test_n
                
                train_seqs, val_seqs, train_classes, val_classes, train_n, val_n =  train_test_split(train_val_seqs, 
                                                                                                     train_val_classes, train_val_n,
                                                                                                     test_size = self.val_ratio,
                                                                                                     stratify = tf.squeeze(train_val_classes),
                                                                                                     random_state = self.seed)
                
                with tf.device('/CPU:0'):
                    self.tf_dataset['train'] = tf.data.Dataset.from_tensor_slices((train_seqs, train_classes))   
                    self.tf_dataset['train'] = self.tf_dataset['train'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['train'] = train_n
                    
                    self.tf_dataset['val'] = tf.data.Dataset.from_tensor_slices((val_seqs, val_classes))   
                    self.tf_dataset['val'] = self.tf_dataset['val'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['val'] = val_n
            
            else:
            
                assert isinstance(self.proteins, np.ndarray), 'If splitting based on proteins, protein ids must be provided'
                
                uniq_proteins = np.unique(self.proteins)
                
                train_val_prot, test_prot = train_test_split(uniq_proteins, 
                                                             test_size = self.test_ratio,
                                                             random_state = self.seed)
                                                          
                train_prot, val_prot = train_test_split(train_val_prot, 
                                                        test_size = self.val_ratio,
                                                        random_state = self.seed)
                
                
                train_index = detectability_dataset.find_index(self.proteins, train_prot)
                np.random.shuffle(train_index)
                
                val_index = detectability_dataset.find_index(self.proteins, val_prot)
                np.random.shuffle(val_index)
                
                test_index = detectability_dataset.find_index(self.proteins, test_prot)
                np.random.shuffle(test_index)   
                
                train_seqs = self.sequences[train_index]
                val_seqs = self.sequences[val_index]
                test_seqs = self.sequences[test_index]
                
                train_classes = self.classes[train_index]
                val_classes = self.classes[val_index]
                test_classes = self.classes[test_index]
                
                with tf.device('/CPU:0'):
                    self.tf_dataset['test'] = tf.data.Dataset.from_tensor_slices((test_seqs, test_classes))   
                    self.tf_dataset['test'] = self.tf_dataset['test'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['test'] = test_index
                    
                    self.tf_dataset['train'] = tf.data.Dataset.from_tensor_slices((train_seqs, train_classes))   
                    self.tf_dataset['train'] = self.tf_dataset['train'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['train'] = train_index
                    
                    self.tf_dataset['val'] = tf.data.Dataset.from_tensor_slices((val_seqs, val_classes))   
                    self.tf_dataset['val'] = self.tf_dataset['val'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['val'] = val_index
                   
        elif self.testing_mode:
            
            if not self.no_classes:
            
                with tf.device('/CPU:0'):
                    self.tf_dataset['test'] = tf.data.Dataset.from_tensor_slices((self.sequences, self.classes))   
                    self.tf_dataset['test'] = self.tf_dataset['test'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['test'] = n
                    
            else:
            
                with tf.device('/CPU:0'):
                    self.tf_dataset['test'] = tf.data.Dataset.from_tensor_slices((self.sequences))   
                    self.tf_dataset['test'] = self.tf_dataset['test'].batch(self.batch_size, drop_remainder = False)# Check before drop_remainder = True
                    self.indicies_dict['test'] = n
            
        

    def get_split_targets(self, split = "test"):
        """Retrieve all targets (original labels) for a specific split.

        :param split: a string specifiying the split name (train, val, test)
        :return: nd.array with the targets
        """
        if split not in self.indicies_dict.keys():
            raise ValueError(
                "requested split does not exist, availabe splits are: "
                + list(self.indicies_dict.keys())
            )

        return self.classes[self.indicies_dict[split]]
        
    def get_split_dataframe(self, split = "test"):
        """Retrieve all targets (original labels) for a specific split.

        :param split: a string specifiying the split name (train, val, test)
        :return: nd.array with the targets
        """
        if split not in self.indicies_dict.keys():
            raise ValueError(
                "requested split does not exist, availabe splits are: "
                + list(self.indicies_dict.keys())
            )
        else:    
                 
#            if not self.no_classes:
#                decoded_classes = np.array([np.argmax(x) for x in self.classes[self.indicies_dict[split]]]) # Commented 08 2024
            
            decoded_seq = np.array(["".join(detectability_dataset.decode_seq(x, int_to_aa_dict)).strip('0') \
                                    for x in self.sequences[self.indicies_dict[split]]])
            
            if not self.no_classes:
                
                decoded_classes = np.array([np.argmax(x) for x in self.classes[self.indicies_dict[split]]])
            
                if self.proteins_info: 
                    split_df = pd.DataFrame({'Sequences': decoded_seq, 
                                             'Classes': decoded_classes, 
                                             'Proteins': self.proteins[self.indicies_dict[split]]})
                
                else:
                    split_df = pd.DataFrame({'Sequences': decoded_seq, 
                                             'Classes': decoded_classes})
            else:
            
                if self.proteins_info: 
                    split_df = pd.DataFrame({'Sequences': decoded_seq, 
                                             'Proteins': self.proteins[self.indicies_dict[split]]})
                
                else:
                    split_df = pd.DataFrame({'Sequences': decoded_seq})
            
        return split_df 

    def _get_tf_dataset(self, split = None):
        assert (
            split in self.tf_dataset.keys()
        ), f"Requested data split is not available, available splits are {self.tf_dataset.keys()}"
        if split in self.tf_dataset.keys():
            return self.tf_dataset[split]
        #return self.tf_dataset

    @property
    def train_data(self):
        """TensorFlow Dataset object for the training data"""
        return self._get_tf_dataset(detectability_dataset.SPLIT_NAMES[0])

    @property
    def val_data(self):
        """TensorFlow Dataset object for the validation data"""
        return self._get_tf_dataset(detectability_dataset.SPLIT_NAMES[1])

    @property
    def test_data(self):
        """TensorFlow Dataset object for the test data"""
        return self._get_tf_dataset(detectability_dataset.SPLIT_NAMES[2])