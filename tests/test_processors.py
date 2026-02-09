import logging

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


def test_sequence_encoding_processor_fixed_alphabet(basic_alphabet):
    """Test encoding with fixed alphabet (unknown tokens -> unknown_token_index)."""
    alphabet = basic_alphabet.copy()
    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet,
        extend_alphabet=False,
        unknown_token="X",
        unknown_token_index=23,
    )

    # Known sequence
    input_data = {SEQ_COLUMN: ["D", "E", "L"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == [3, 4, 10]  # D=3, E=4, L=10 from alphabet

    # Unknown token should map to unknown_token_index
    # Note: alphabet is a copy from fixture which doesn't have Z
    input_data = {SEQ_COLUMN: ["Z"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == [23]  # Z unknown -> unknown_token_index


def test_sequence_encoding_processor_fixed_alphabet_with_ptm(ptm_alphabet):
    """Test encoding with PTM modifications in fixed alphabet."""
    alphabet = ptm_alphabet.copy()
    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet,
        extend_alphabet=False,
    )

    input_data = {SEQ_COLUMN: ["H", "C[UNIMOD:4]", "V", "D"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == [7, 23, 18, 3]  # H=7, C[UNIMOD:4]=23, V=18, D=3


def test_sequence_encoding_processor_extend_alphabet():
    """Test encoding with alphabet extension (learns new tokens)."""
    initial_alphabet = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "L": 10,
    }

    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=initial_alphabet,
        extend_alphabet=True,
    )

    # First sequence - learns new tokens
    input_data = {SEQ_COLUMN: ["D", "E", "L"]}
    result = p(input_data)
    assert result[SEQ_COLUMN] == [3, 4, 10]

    # Second sequence with unknown token - extends alphabet
    input_data = {SEQ_COLUMN: ["A", "C", "V"]}
    result = p(input_data)
    # V should now be in the alphabet
    assert "V" in p.alphabet
    assert result[SEQ_COLUMN][2] == p.alphabet["V"]


def test_sequence_encoding_processor_extend_alphabet_batched(basic_alphabet):
    """Test alphabet extension in batched mode."""
    alphabet = basic_alphabet.copy()
    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet,
        batched=True,
        extend_alphabet=True,
    )

    # Initialize with unknown amino acids
    initial_alphabet = {"A": 1, "C": 2, "D": 3}
    p.alphabet = initial_alphabet.copy()

    input_data = {SEQ_COLUMN: [["A", "C"], ["D", "E"]]}
    result = p(input_data)

    # "E" should be learned and added to alphabet
    assert "E" in p.alphabet
    assert len(result[SEQ_COLUMN]) == 2
    assert len(result[SEQ_COLUMN][0]) == 2
    assert len(result[SEQ_COLUMN][1]) == 2


def test_sequence_encoding_processor_fallback_mode_unmodified(basic_alphabet):
    """Test fallback mode: unknown PTM -> unmodified amino acid."""
    alphabet = basic_alphabet.copy()
    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet,
        extend_alphabet=False,
        fallback_unmodified=True,
    )

    # Unknown PTM C[UNIMOD:999] should fallback to C
    input_data = {SEQ_COLUMN: ["C[UNIMOD:999]"]}
    result = p(input_data)

    # C[UNIMOD:999] -> C -> 2
    assert result[SEQ_COLUMN][0] == 2


def test_sequence_encoding_processor_fallback_mode_terminal_mods(basic_alphabet):
    """Test fallback mode with unknown terminal modifications."""
    alphabet = basic_alphabet.copy()
    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet,
        extend_alphabet=False,
        fallback_unmodified=True,
    )

    # Unknown N-terminal mod [UNIMOD:999]- should fallback to []-
    input_data = {SEQ_COLUMN: ["[UNIMOD:999]-", "D", "E"]}
    result = p(input_data)

    assert result[SEQ_COLUMN][0] == 21  # []-
    assert result[SEQ_COLUMN][1] == 3  # D
    assert result[SEQ_COLUMN][2] == 4  # E


def test_sequence_encoding_processor_unknown_token_already_in_alphabet():
    """Test that unknown token uses existing index if already in alphabet."""
    alphabet = {
        "A": 1,
        "X": 2,  # X already in alphabet
        "D": 3,
    }

    # Should not raise, but should warn
    with pytest.warns(UserWarning):
        p = SequenceEncodingProcessor(
            sequence_column_name=SEQ_COLUMN,
            alphabet=alphabet,
            unknown_token="X",
            unknown_token_index=1,  # Will be overridden to 2
        )

    # unknown_token_index should be set to existing X index
    assert p.unknown_token_index == 2


def test_sequence_encoding_processor_unknown_token_index_conflict():
    """Test warning when unknown_token_index is already used."""
    alphabet = {
        "A": 1,
        "C": 2,
        "D": 3,
    }

    # Should warn about index conflict
    with pytest.warns(UserWarning):
        p = SequenceEncodingProcessor(
            sequence_column_name=SEQ_COLUMN,
            alphabet=alphabet,
            unknown_token_index=2,  # 2 is already used by C
        )

    # Index should be reassigned
    assert p.unknown_token_index != 2


def test_sequence_encoding_processor_single_vs_batched_consistency(basic_alphabet):
    """Test that single and batched modes produce identical results."""
    alphabet = basic_alphabet.copy()
    p_single = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet.copy(),
        batched=False,
    )
    p_batch = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet=alphabet.copy(),
        batched=True,
    )

    sequences = [["D", "E", "L"], ["A", "C"], ["K", "L", "M"]]

    # Process individually
    results_single = [p_single({SEQ_COLUMN: seq})[SEQ_COLUMN] for seq in sequences]

    # Process in batch
    result_batch = p_batch({SEQ_COLUMN: sequences})[SEQ_COLUMN]

    assert results_single == result_batch


# ============================================================================
# SEQUENCE PTM REMOVAL PROCESSOR TESTS
# ============================================================================


def test_sequence_ptm_removal_processor_basic():
    """Test basic PTM removal from sequence."""
    p = SequencePTMRemovalProcessor(sequence_column_name=SEQ_COLUMN)

    input_data = {SEQ_COLUMN: ["[]-", "C[UNIMOD:4]", "V", "D", "-[]"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == ["[]-", "C", "V", "D", "-[]"]


def test_sequence_ptm_removal_processor_multiple_ptms():
    """Test removal of multiple PTMs from same sequence."""
    p = SequencePTMRemovalProcessor(sequence_column_name=SEQ_COLUMN)

    input_data = {
        SEQ_COLUMN: ["[UNIMOD:737]-", "C[UNIMOD:4]", "K[UNIMOD:737]", "S", "-[]"]
    }
    result = p(input_data)

    assert result[SEQ_COLUMN] == ["[UNIMOD:737]-", "C", "K", "S", "-[]"]


def test_sequence_ptm_removal_processor_no_ptms():
    """Test sequence without PTMs is unchanged."""
    p = SequencePTMRemovalProcessor(sequence_column_name=SEQ_COLUMN)

    input_data = {SEQ_COLUMN: ["[]-", "D", "E", "L", "-[]"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == ["[]-", "D", "E", "L", "-[]"]


def test_sequence_ptm_removal_processor_batched():
    """Test PTM removal in batched mode."""
    p = SequencePTMRemovalProcessor(sequence_column_name=SEQ_COLUMN, batched=True)

    input_data = {
        SEQ_COLUMN: [
            ["[]-", "C[UNIMOD:4]", "V", "-[]"],
            ["[]-", "K[UNIMOD:737]", "D", "-[]"],
        ]
    }
    result = p(input_data)

    assert result[SEQ_COLUMN][0] == ["[]-", "C", "V", "-[]"]
    assert result[SEQ_COLUMN][1] == ["[]-", "K", "D", "-[]"]


def test_sequence_ptm_removal_processor_preserves_terminals():
    """Test that terminal modifications are preserved."""
    p = SequencePTMRemovalProcessor(sequence_column_name=SEQ_COLUMN)

    input_data = {SEQ_COLUMN: ["[UNIMOD:737]-", "C[UNIMOD:4]", "-[UNIMOD:1]"]}
    result = p(input_data)

    # Terminals should be unchanged
    assert result[SEQ_COLUMN][0] == "[UNIMOD:737]-"
    assert result[SEQ_COLUMN][-1] == "-[UNIMOD:1]"


def test_sequence_ptm_removal_processor_non_list_input_raises_error():
    """Test that non-list input raises ValueError."""
    p = SequencePTMRemovalProcessor(sequence_column_name=SEQ_COLUMN)

    # String input should raise error
    input_data = {SEQ_COLUMN: "C[UNIMOD:4]VD"}

    with pytest.raises(ValueError, match="Sequence must be a list"):
        p(input_data)


# ============================================================================
# FUNCTION PROCESSOR TESTS
# ============================================================================


def test_function_processor_basic(sample_custom_function):
    """Test basic function processor application."""
    p = FunctionProcessor(function=sample_custom_function)

    input_data = {SEQ_COLUMN: ["D", "E", "L"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == ["D", "E", "L", "D", "E", "L"]


def test_function_processor_with_kwargs(sample_custom_function_with_kwargs):
    """Test function processor with keyword arguments."""
    p = FunctionProcessor(function=sample_custom_function_with_kwargs)

    input_data = {"feature": 5.0}
    result = p(input_data, scale_factor=2.0)

    assert result["feature"] == 10.0


def test_function_processor_lambda():
    """Test function processor with lambda function."""
    p = FunctionProcessor(
        function=lambda data, **kwargs: {"sequence": data.get("sequence", []) * 2}
    )

    input_data = {"sequence": [1, 2, 3]}
    result = p(input_data)

    assert result["sequence"] == [1, 2, 3, 1, 2, 3]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_sequence_padding_processor_exact_length():
    """Test padding when sequence length equals max_length."""
    length = 3
    p = SequencePaddingProcessor(sequence_column_name=SEQ_COLUMN, max_length=length)

    input_data = {SEQ_COLUMN: ["D", "E", "L"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == ["D", "E", "L"]
    assert result[SequencePaddingProcessor.KEEP_COLUMN_NAME] is True


def test_sequence_padding_processor_custom_padding_index():
    """Test padding with custom padding index."""
    p = SequencePaddingProcessor(
        sequence_column_name=SEQ_COLUMN,
        max_length=5,
        padding_index=-1,
    )

    input_data = {SEQ_COLUMN: ["D", "E", "L"]}
    result = p(input_data)

    assert result[SEQ_COLUMN] == ["D", "E", "L", -1, -1]


def test_sequence_padding_processor_empty_sequence():
    """Test padding with empty sequence."""
    p = SequencePaddingProcessor(sequence_column_name=SEQ_COLUMN, max_length=3)

    input_data = {SEQ_COLUMN: []}
    result = p(input_data)

    assert result[SEQ_COLUMN] == [0, 0, 0]
    assert result[SequencePaddingProcessor.KEEP_COLUMN_NAME] is True


def test_sequence_padding_processor_long_sequence():
    """Test truncation of very long sequence."""
    p = SequencePaddingProcessor(sequence_column_name=SEQ_COLUMN, max_length=5)

    long_sequence = list(range(10))
    input_data = {SEQ_COLUMN: long_sequence}
    result = p(input_data)

    assert len(result[SEQ_COLUMN]) == 5
    assert result[SEQ_COLUMN] == long_sequence[:5]
    assert result[SequencePaddingProcessor.KEEP_COLUMN_NAME] is False


def test_sequence_parsing_processor_invalid_format():
    """Test error on invalid ProForma format (4+ parts)."""
    p = SequenceParsingProcessor(sequence_column_name=SEQ_COLUMN)

    input_data = {SEQ_COLUMN: "[UNIMOD:1]-SEQ-[UNIMOD:2]-EXTRA"}

    with pytest.raises(ValueError, match="Invalid sequence format"):
        p(input_data)


def test_sequence_encoding_processor_empty_alphabet_fixed_mode():
    """Test that fixed mode works with empty alphabet + unknown tokens."""
    p = SequenceEncodingProcessor(
        sequence_column_name=SEQ_COLUMN,
        alphabet={},  # Empty alphabet (will get unknown token added)
        extend_alphabet=False,
        unknown_token="X",
        unknown_token_index=1,
    )

    # Alphabet was empty, but unknown token X was added with index 1
    assert "X" in p.alphabet
    assert p.unknown_token_index == 1

    input_data = {SEQ_COLUMN: ["A", "C", "D"]}
    result = p(input_data)

    # All should map to unknown_token_index since they're not in alphabet
    assert result[SEQ_COLUMN] == [1, 1, 1]
