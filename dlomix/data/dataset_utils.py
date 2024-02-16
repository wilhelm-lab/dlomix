import re
from enum import Enum

from .sequence_utils import parse_sequence_native, rebuild_proforma_sequence


class EncodingScheme(Enum):
    NO_MODS = "unmod"
    NAIVE_MODS = "naive-mods"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"Provided {cls.__name__} is not defined. Valid values are: {', '.join([e.value for e in cls])}"
        )


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

    example[sequence_column_name] = rebuild_proforma_sequence(
        seq,
        mods,
        n_terms,
        c_terms,
    )

    return example


def add_parsed_sequence_info_fast(
    example,
    sequence_column_name: str,
    new_column_names: list = [
        "raw_sequence",
        "n_terminal_mods",
        "c_terminal_mods",
    ],
):
    n_terms, seq, c_terms = parse_sequence_native(example[sequence_column_name])
    example[new_column_names[0]] = seq
    example[new_column_names[1]] = n_terms
    example[new_column_names[2]] = c_terms

    example[sequence_column_name] = [n_terms] + seq + [c_terms]

    return example


def add_parsed_sequence_info_fast_batched(
    batch,
    sequence_column_name: str,
    new_column_names: list = [
        "raw_sequence",
        "n_terminal_mods",
        "c_terminal_mods",
    ],
):
    batch[new_column_names[0]] = []
    batch[new_column_names[1]] = []
    batch[new_column_names[2]] = []

    for index, sequence in enumerate(batch[sequence_column_name]):
        n_terms, seq, c_terms = parse_sequence_native(sequence)
        batch[new_column_names[0]].append(seq)
        batch[new_column_names[1]].append(n_terms)
        batch[new_column_names[2]].append(c_terms)

        updated_seq = seq
        if n_terms != "[]-":
            updated_seq = [n_terms] + updated_seq
        if c_terms != "-[]":
            updated_seq = updated_seq + [c_terms]

        batch[sequence_column_name][index] = updated_seq

    return batch


def update_sequence_with_splitted_proforma_format(
    example,
    raw_sequence_column_name,
    mods_column_name,
    n_term_column_name,
    c_term_column_name,
    new_column_name,
):
    example[new_column_name] = rebuild_proforma_sequence(
        example[raw_sequence_column_name],
        example[mods_column_name],
        example[n_term_column_name],
        example[c_term_column_name],
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


def encode_sequence_batched(batch, sequence_column_name, alphabet):
    # encode with alphabet

    for index, sequence in enumerate(batch[sequence_column_name]):
        batch[sequence_column_name][index] = [
            alphabet.get(amino_acid) for amino_acid in sequence
        ]

    return batch


def pad_truncate_sequence(example, seq_len, padding, sequence_column_name):
    length = len(example[sequence_column_name])
    if length < seq_len:
        example[sequence_column_name].extend([padding] * (seq_len - length))
    if length > seq_len:
        example[sequence_column_name] = example[sequence_column_name][:seq_len]

    return example


def pad_truncate_sequence_batched(batch, seq_len, padding, sequence_column_name):
    for index, sequence in enumerate(batch[sequence_column_name]):
        length = len(sequence)
        if length < seq_len:
            batch[sequence_column_name][index].extend([padding] * (seq_len - length))
        if length > seq_len:
            batch[sequence_column_name][index] = sequence[:seq_len]

    return batch


def pad_sequence(example, seq_len, padding, sequence_column_name):
    length = len(example[sequence_column_name])
    if length < seq_len:
        example[sequence_column_name].extend([padding] * (seq_len - length))
    return example


def get_num_processors():
    import multiprocessing

    return multiprocessing.cpu_count()
