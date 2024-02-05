import pickle

import numpy as np


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def convert_nested_list_to_numpy_array(nested_list, dtype=np.float32):
    return np.array([np.array(x, dtype=dtype) for x in nested_list])
