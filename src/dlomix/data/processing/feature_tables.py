""" Module to load pickled feature dictionaries. """

import os
import pickle

MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
PKL_BASE_PATH = os.path.join(MODULE_PATH, "pickled_feature_dicts")


def load_pickled_feature(pickle_filename: str):
    """
    Load a pickled feature dictionary.

    Parameters
    ----------
    pickle_filename : str
        Name of the pickled file.

    Returns
    -------
    dict
        Loaded pickled dictionary.
    """

    with open(os.path.join(PKL_BASE_PATH, pickle_filename), "rb") as f:
        return pickle.load(f)


PTM_LOSS_LOOKUP = load_pickled_feature("saved_loss_atoms.pkl")
PTM_MOD_DELTA_MASS_LOOKUP = load_pickled_feature("mz_diff.pkl")
PTM_GAIN_LOOKUP = load_pickled_feature("saved_gained_atoms.pkl")
PTM_ATOM_COUNT_LOOKUP = load_pickled_feature("saved_ac_count.pkl")
PTM_RED_SMILES_LOOKUP = load_pickled_feature("red_smiles.pkl")
