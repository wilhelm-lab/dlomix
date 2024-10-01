import logging
import urllib.request
import zipfile
from os import makedirs
from os.path import exists, join

import pytest

from dlomix.data.processing import (
    FunctionProcessor,
    SequenceEncodingProcessor,
    SequencePaddingProcessor,
    SequenceParsingProcessor,
    SequencePTMRemovalProcessor,
)

logger = logging.getLogger(__name__)

SEQ_COLUMN = "sequence"

SEQUENCE_UNMODIFIED = "[]-DEL-[]"
PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI = ["D", "E", "L"]

SEQUENCE_MODIFIED = "[]-HC[UNIMOD:4]VD-[]"
PARSED_SEQUENCE_MODIFIED_NO_TERMINI = ["H", "C[UNIMOD:4]", "V", "D"]

SEQUENCE_MODIFIED_WITH_N_MOD = "[UNIMOD:737]-ILC[UNIMOD:4]SIQGFK[UNIMOD:737]D-[]"
PARSED_SEQUENCE_MODIFIED_WITH_N_MOD_NO_TERMINI = [
    "I",
    "L",
    "C[UNIMOD:4]",
    "S",
    "I",
    "Q",
    "G",
    "F",
    "K[UNIMOD:737]",
    "D",
]


def assert_parsed_data(
    parsed_data, new_sequence_column_value, parsed_sequence_value, n_term, c_term
):
    assert SequenceParsingProcessor.PARSED_COL_NAMES["seq"] in parsed_data.keys()
    assert SequenceParsingProcessor.PARSED_COL_NAMES["n_term"] in parsed_data.keys()
    assert SequenceParsingProcessor.PARSED_COL_NAMES["c_term"] in parsed_data
    assert SEQ_COLUMN in parsed_data.keys()

    assert (
        parsed_data[SequenceParsingProcessor.PARSED_COL_NAMES["seq"]]
        == parsed_sequence_value
    )
    assert parsed_data[SequenceParsingProcessor.PARSED_COL_NAMES["n_term"]] == n_term
    assert parsed_data[SequenceParsingProcessor.PARSED_COL_NAMES["c_term"]] == c_term
    assert parsed_data[SEQ_COLUMN] == new_sequence_column_value


def test_sequence_parsing_processor_unmodified():
    p = SequenceParsingProcessor(sequence_column_name=SEQ_COLUMN, with_termini=False)
    input_data = {SEQ_COLUMN: SEQUENCE_UNMODIFIED}

    parsed = p(input_data)
    logger.info(parsed)

    assert_parsed_data(
        parsed,
        PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI,
        PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI,
        "[]-",
        "-[]",
    )


def test_sequence_parsing_processor_batched():
    p = SequenceParsingProcessor(
        sequence_column_name=SEQ_COLUMN, with_termini=False, batched=True
    )
    input_data = {SEQ_COLUMN: [SEQUENCE_UNMODIFIED]}

    parsed = p(input_data)
    logger.info(parsed)

    assert_parsed_data(
        parsed,
        [PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI],
        [PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI],
        ["[]-"],
        ["-[]"],
    )


def test_sequence_parsing_processor_with_termini():
    p = SequenceParsingProcessor(sequence_column_name=SEQ_COLUMN, with_termini=True)
    input_data = {SEQ_COLUMN: SEQUENCE_UNMODIFIED}

    parsed = p(input_data)
    logger.info(parsed)

    assert_parsed_data(
        parsed,
        ["[]-", *PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI, "-[]"],
        PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI,
        "[]-",
        "-[]",
    )


def test_sequence_parsing_processor_with_modifications():
    p = SequenceParsingProcessor(sequence_column_name=SEQ_COLUMN, with_termini=True)
    input_data = {SEQ_COLUMN: SEQUENCE_MODIFIED}

    parsed = p(input_data)
    logger.info(parsed)

    assert_parsed_data(
        parsed,
        ["[]-", *PARSED_SEQUENCE_MODIFIED_NO_TERMINI, "-[]"],
        PARSED_SEQUENCE_MODIFIED_NO_TERMINI,
        "[]-",
        "-[]",
    )


def test_sequence_parsing_processor_with_modifications_and_nterm_mods():
    p = SequenceParsingProcessor(sequence_column_name=SEQ_COLUMN, with_termini=True)
    input_data = {SEQ_COLUMN: SEQUENCE_MODIFIED_WITH_N_MOD}

    parsed = p(input_data)
    logger.info(parsed)

    assert_parsed_data(
        parsed,
        ["[UNIMOD:737]-", *PARSED_SEQUENCE_MODIFIED_WITH_N_MOD_NO_TERMINI, "-[]"],
        PARSED_SEQUENCE_MODIFIED_WITH_N_MOD_NO_TERMINI,
        "[UNIMOD:737]-",
        "-[]",
    )


def test_sequence_padding_processor_keep():
    length = 5
    p = SequencePaddingProcessor(sequence_column_name=SEQ_COLUMN, max_length=length)
    input_data = {SEQ_COLUMN: PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI}
    padded = p(input_data)
    logger.info(padded)

    assert len(padded[SEQ_COLUMN]) == length
    assert padded[SEQ_COLUMN] == ["D", "E", "L", 0, 0]
    assert padded[SequencePaddingProcessor.KEEP_COLUMN_NAME]


def test_sequence_padding_processor_drop():
    length = 2
    p = SequencePaddingProcessor(sequence_column_name=SEQ_COLUMN, max_length=length)
    input_data = {SEQ_COLUMN: PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI}
    padded = p(input_data)
    logger.info(padded)

    assert padded[SEQ_COLUMN] == PARSED_SEQUENCE_UNMODIFIED_NO_TERMINI[:length]
    assert not padded[SequencePaddingProcessor.KEEP_COLUMN_NAME]


def test_sequence_encoding_processor():
    pass


def test_sequence_encoding_processor_with_fixed_alphabet():
    pass


def test_sequence_encoding_processor_with_extend_alphabet_enabled():
    pass


def test_sequence_encoding_processor_with_fallback_enabled():
    pass


def test_sequence_ptm_removal_processor():
    pass


def test_function_processor():
    pass
