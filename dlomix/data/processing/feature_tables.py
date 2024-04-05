import os
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
PKL_BASE_PATH = os.path.join(script_dir, "pickled_feature_dicts")

with open(os.path.join(PKL_BASE_PATH, "saved_loss_atoms.pkl"), "rb") as f:
    PTM_LOSS_LOOKUP = pickle.load(f)

with open(os.path.join(PKL_BASE_PATH, "mz_diff.pkl"), "rb") as f:
    PTM_MOD_DELTA_MASS_LOOKUP = pickle.load(f)

with open(os.path.join(PKL_BASE_PATH, "saved_gained_atoms.pkl"), "rb") as f:
    PTM_GAIN_LOOKUP = pickle.load(f)

with open(os.path.join(PKL_BASE_PATH, "saved_ac_count.pkl"), "rb") as f:
    PTM_ATOM_COUNT_LOOKUP = pickle.load(f)
