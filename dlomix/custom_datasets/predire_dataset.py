"""
https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18

Using the make_dataset subroutine below, I can create a tensorflow dataset
that loads input and target tensors, and is compatible with the compile and 
train functions of tensorflow.
"""

import tensorflow as tf
import yaml
import math
import pandas as pd
import numpy as np
import re

def order_intensities(ints, annotations, sequence, map_dic):
    out = np.zeros((len(map_dic)), np.float32)
    for ab,ann in zip(ints, annotations):
        if ann[:3]=='Int':
            ann = "".join(ann.split('/')) # Turn Int/{ann} into Int{ann}
            # Convert internal notation to start>extent
            hold = re.sub("[+-]", ',', ann).split(",") # [ann, neut/iso]
            # issue with internals starting at 0
            if sequence.find(hold[0][3:].upper()) == 0:
                # Find first uppercase match after 1st AA
                start = sequence[1:-1].upper().find(hold[0][3:].upper()) + 1
            else: start = sequence.find(hold[0][3:].upper())
            ann = 'Int%d>%d%s'%(start, len(hold[0][3:]), ann[len(hold[0]):])
        if ann in map_dic.keys():
            out[map_dic[ann]] = ab
    out /= max(out)
    
    return out

class Gen:#(tf.keras.utils.Sequence):
    def __init__(self, df, config_path):
        self.df = df
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        self.config = config
        
        # The dict_key is used in the get_y method to sort the intensities into
        # a target vector
        self.dict_key = {
            line.split()[0]: int(line.split()[1])
            for line in open(config['dict_path'])
        } if config['dict_path'] is not None else None
        
        # The vocabs, chlen, and total channels are used in the get_x method
        # to create input tensors
        # sequence vocab
        self.vocab = {n:m for m,n in enumerate(list(config['aa']))}
        # modification vocab
        self.mvocab = {name:i for i, name in enumerate(config['mods'])}
        self.chlen = config['charge'][1]-config['charge'][0]+1
        self.total_channels = ( 
            len(self.vocab) +
            len(self.mvocab)+1 +
            config['charge'][1]-config['charge'][0]+1 +
            1
        )
    
    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)
    
    def get_x(self, idx):
        sample = self.df.iloc[idx]
        # 1 hot AA sequence
        seq = tf.one_hot(
            [self.vocab[m] for m in list(sample['seq'])] + 
            (self.config['max_seq']-len(sample['seq']))*[self.vocab['X']], 
            depth=len(self.vocab)
        )
        
        # Mods
        mi = sample['mod_inds']
        mn = sample['mod_names']
        nomods = tf.ones((self.config['max_seq'], 1))
        nomods = tf.tensor_scatter_nd_update(
            nomods, list(zip(mi, len(mi)*[0])), tf.zeros((len(mi)))
        )
        mods = tf.zeros((self.config['max_seq'], len(self.mvocab)))
        indices = list(zip(
            mi, [self.mvocab[m] for m in mn]
        ))
        updates = tf.ones((len(mi),))
        mods = tf.tensor_scatter_nd_update(mods, indices, updates)
        mods = tf.concat([nomods,mods],axis=-1)
        
        # Charge
        charge = tf.one_hot(
            tf.fill((self.config['max_seq']), sample['charge']-1), 
            depth=self.chlen
        )
        
        # collision energy
        ev = 0.01*sample['ev']*tf.ones((self.config['max_seq'], 1))
        
        one = tf.concat([seq, mods, charge, ev], axis=-1)
        
        return one
    
    def get_y(self, idx):
        sample = self.df.iloc[idx]
        ints = sample['ab']
        anns = sample['anns']
        seqs = sample['seq']
        ordered_ints = order_intensities(ints, anns, seqs, self.dict_key)
        tfints = tf.constant(ordered_ints, tf.float32)
        
        return tfints
    
    def __getitem__(self, idx):
        inp = self.get_x(idx)
        targ = self.get_y(idx)
        
        return inp, targ
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def create_dataset(pd_dataframe, config_path, inds=None, batch_size=100):
    
    # Configuration yaml file with peptide specs
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    # dataframe
    # df = pd.read_pickle(pkl_path)[
    #     ['seq','charge','ev','mod_inds','mod_names','mz','ab','anns']
    # ]
    df = pd_dataframe
    if inds is not None:
        df = df.iloc[inds]
    
    # Filter based on config file
    # length
    df = df[np.vectorize(len)(df['seq'])<config['max_seq']]
    # charge
    df = df.query("charge >= %d and charge <= %d"%tuple(config['charge']))
    # modifications
    mod_bool = [
        # If only 1 unallowed mod in peptide, then get rid of it
        False if False in m else True for m in 
        [
            [
                True if mod in config['mods'] else False 
                for mod in row[1].mod_names
            ] 
            for row in df.iterrows()
        ]
    ]
    df = df[mod_bool]
    
    # Instantiate the (non-tf) generator
    gen = Gen(df, config_path)
    
    # A python function for tf.py_function
    def func(idx):
        i = idx.numpy() # Is this the only reason this function exists?
        inp, output = gen.__getitem__(i)
        return tf.constant(inp, tf.float32), tf.constant(output, tf.float32)
    
    # Instantiate the tf dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: list(range(len(df))), tf.uint8
    )

    dataset = dataset.map(lambda i: tf.py_function(
        func, inp=[i], Tout=[tf.float32, tf.float32]), 
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size)
    dataset.__len__ = lambda: math.ceil(df.__len__() / batch_size) 
    
    return dataset

# # storing my data as pickle file
# pkl_path = "C:/Users/joell/Desktop/test.pkl"
# # config has max sequence length, charge range, modifications, etc.
# config_path = "C:/Users/joell/Desktop/config.yaml"

# dataset = create_dataset(pkl_path, config_path)
