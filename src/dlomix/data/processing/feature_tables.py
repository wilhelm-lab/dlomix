"""Module to load JSON feature dictionaries."""

import json
import os
from typing import Tuple

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


def _split_loss_gain(combined: dict) -> Tuple[dict, dict]:
    """Split a ``{token: {"loss": [...], "gain": [...]}}`` dict into two flat lookups."""
    return (
        {token: values["loss"] for token, values in combined.items()},
        {token: values["gain"] for token, values in combined.items()},
    )


PTM_LOSS_LOOKUP, PTM_GAIN_LOOKUP = _split_loss_gain(
    load_feature_dict("saved_loss_gain_atoms.json")
)
PTM_MOD_DELTA_MASS_LOOKUP = load_feature_dict("mz_diff.json")
PTM_ATOM_COUNT_LOOKUP = load_feature_dict("saved_ac_count.json")
PTM_RED_SMILES_LOOKUP = load_feature_dict("red_smiles.json")
