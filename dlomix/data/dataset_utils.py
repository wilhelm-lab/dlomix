import re

from .sequence_utils import rebuild_proforma_sequence


def add_parsed_sequence_info(
    example,
    sequence_column_name: str,
    parser,
    new_column_names: list = [
        "raw_sequence",
        "mods",
        "n_terminal_mods",
        "c_terminal_mods",
    ],
):
    seq, mods, n_terms, c_terms = parser._parse_sequence(example[sequence_column_name])
    example[new_column_names[0]] = seq
    example[new_column_names[1]] = mods
    example[new_column_names[2]] = n_terms
    example[new_column_names[3]] = c_terms
    return example


def update_sequence_with_splitted_proforma_format(
    example, raw_sequence_column_name, mods_column_name, new_column_name
):
    example[new_column_name] = rebuild_proforma_sequence(
        example[raw_sequence_column_name], example[mods_column_name]
    )
    return example


def remove_ptms(example, sequence_column_name):
    example[sequence_column_name] = re.sub(
        r"\[UNIMOD:\d+\]", "", example[sequence_column_name]
    )
    return example


def encode_sequence(example, sequence_column_name, alphabet):
    # encode with alphabet
    example[sequence_column_name] = [
        alphabet.get(amino_acid) for amino_acid in example[sequence_column_name]
    ]
    return example


def pad_drop_sequence(example, seq_len, padding, sequence_column_name):
    length = len(example[sequence_column_name])
    if length < seq_len:
        example[sequence_column_name].extend([padding] * (seq_len - length))
    if length > seq_len:
        example[sequence_column_name] = example[sequence_column_name][:seq_len]
    return example


def get_mod_loss_feature(
    example,
    sequence_column_name,
    feature_column_name="mod_loss",
    default_value=[0, 0, 0, 0, 0, 0],
):
    PTM_LOSS_LOOKUP = {
        "M[UNIMOD:35]": [0, 0, 0, 0, 0, 0],
        "S[UNIMOD:21]": [1, 0, 0, 0, 0, 0],
        "T[UNIMOD:21]": [1, 0, 0, 0, 0, 0],
        "Y[UNIMOD:21]": [1, 0, 0, 0, 0, 0],
        "R[UNIMOD:7]": [1, 0, 1, 0, 0, 0],
        "K[UNIMOD:1]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:121]": [1, 0, 0, 0, 0, 0],
        "Q(gl)": [9, 4, 2, 1, 0, 0],
        "R[UNIMOD:34]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:34]": [1, 0, 0, 0, 0, 0],
        "T(ga)": [1, 0, 0, 0, 0, 0],
        "S(ga)": [1, 0, 0, 0, 0, 0],
        "T(gl)": [1, 0, 0, 0, 0, 0],
        "S(gl)": [1, 0, 0, 0, 0, 0],
        "C[UNIMOD:4]": [1, 0, 0, 0, 0, 0],
        "[ac]-": [1, 0, 0, 0, 0, 0],
        "E(gl)": [8, 4, 1, 2, 0, 0],
        "K[UNIMOD:36]": [2, 0, 0, 0, 0, 0],
        "K[UNIMOD:37]": [3, 0, 0, 0, 0, 0],
        "K[UNIMOD:122]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:58]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1289]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:747]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:64]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1848]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1363]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1849]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:3]": [1, 0, 0, 0, 0, 0],
        "unknown": [3, 0, 2, 0, 0, 0],
        "R[UNIMOD:36]": [2, 0, 0, 0, 0, 0],
        "P[UNIMOD:35]": [1, 0, 0, 0, 0, 0],
        "Y[UNIMOD:354]": [1, 0, 0, 0, 0, 0],
    }
    sequence = example[sequence_column_name].strip("_")

    if "UNIMOD" not in example[sequence_column_name]:
        example[feature_column_name] = [default_value for _ in range(len(sequence))]
        return example

    example[feature_column_name] = [
        PTM_LOSS_LOOKUP.get(i, [0] * 6) for i in example[sequence_column_name]
    ]
    return example
