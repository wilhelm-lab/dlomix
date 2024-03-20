import pickle 
with open('saved_loss_atoms.pkl', 'rb') as f:
    PTM_LOSS_LOOKUP = pickle.load(f)

with open('mz_diff.pkl', 'rb') as f:
    PTM_MOD_DELTA_MASS_LOOKUP = pickle.load(f)

with open('saved_gained_atoms.pkl', 'rb') as f:
    PTM_GAIN_LOOKUP = pickle.load(f)

with open('saved_ac_count.pkl', 'rb') as f:
    PTM_ATOM_COUNT_LOOKUP = pickle.load(f)