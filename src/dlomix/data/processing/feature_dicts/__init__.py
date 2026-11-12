"""Lookup tables of the built-in PTM feature extractors, one JSON file per table.

Every table maps a token of a parsed sequence to its feature value. A token is a
modified residue (``"M[UNIMOD:35]"``, ``"K[UNIMOD:737][UNIMOD:1]"``) or a modified
terminus (``"[UNIMOD:1]-"``). Tokens without an entry, such as unmodified residues,
get the default value of the feature (zeros).

``saved_ac_count.json`` (feature ``atom_count``)
    The change in atom counts by the modification, as six integers in the order
    ``[H, C, N, O, P, S]`` (:data:`~dlomix.data.processing.feature_tables.PTM_ATOMS`).
    Oxidation, ``"M[UNIMOD:35]"``, is ``[0, 0, 0, 1, 0, 0]``: one oxygen more.
    Phosphorylation, ``"S[UNIMOD:21]"``, is ``[1, 0, 0, 3, 1, 0]``: HPO3.

``saved_loss_gain_atoms.json`` (features ``mod_loss`` and ``mod_gain``)
    ``{"loss": [...], "gain": [...]}`` per token, both in the same atom order: the
    atoms of the group that leaves the residue and of the group that replaces it.
    ``gain - loss`` is the entry of ``saved_ac_count.json``.

``mz_diff.json`` (feature ``delta_mass``)
    One number per token: the mass shift of the modification in Da, as the
    published Prosit PTM features have it, which is 1.0 above the monoisotopic
    mass for most entries (oxidation is stored as 16.9949).

``red_smiles.json`` (feature ``red_smiles``)
    The reduced SMILES of the residue with its modification, encoded as 60
    integers, zero-padded. This table also has the unmodified residues.

The atom features reach the model shifted by +1 (``feature_value_offset`` in
:mod:`dlomix.data.processing.feature_extractors`): the tables hold the true counts,
and padding positions hold 0.
"""
