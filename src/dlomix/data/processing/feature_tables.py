"""Module to load JSON feature dictionaries."""

import json
import os

MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
FEATURE_DICTS_BASE_PATH = os.path.join(MODULE_PATH, "feature_dicts")


def load_feature_dict(json_filename: str):
    """
    Load a JSON feature dictionary.

    Parameters
    ----------
    json_filename : str
        Name of the JSON file.

    Returns
    -------
    dict
        Loaded feature dictionary.
    """

    with open(os.path.join(FEATURE_DICTS_BASE_PATH, json_filename)) as f:
        return json.load(f)


PTM_LOSS_LOOKUP = load_feature_dict("saved_loss_atoms.json")
PTM_MOD_DELTA_MASS_LOOKUP = load_feature_dict("mz_diff.json")
PTM_GAIN_LOOKUP = load_feature_dict("saved_gained_atoms.json")
PTM_ATOM_COUNT_LOOKUP = load_feature_dict("saved_ac_count.json")
PTM_RED_SMILES_LOOKUP = load_feature_dict("red_smiles.json")
