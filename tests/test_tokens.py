from dlomix.data import extend_alphabet, find_new_tokens, sequence_tokens

SEQUENCES = ["[UNIMOD:1]-AC[UNIMOD:4]K-[]", "AR[UNIMOD:7]K", "AR[UNIMOD:7]K"]
ALPHABET = {"-": 0, "X": 1, "[]-": 2, "-[]": 3, "A": 4, "K": 5, "R": 6}


def test_sequence_tokens_are_what_the_dataset_encodes():
    assert sequence_tokens(SEQUENCES[0]) == [
        "[UNIMOD:1]-",
        "A",
        "C[UNIMOD:4]",
        "K",
        "-[]",
    ]
    # a bare sequence gets the unmodified termini; without termini, residues only
    assert sequence_tokens("AR[UNIMOD:7]K") == ["[]-", "A", "R[UNIMOD:7]", "K", "-[]"]
    assert sequence_tokens(SEQUENCES[0], with_termini=False) == [
        "A",
        "C[UNIMOD:4]",
        "K",
    ]


def test_new_tokens_extend_the_alphabet_without_moving_known_tokens():
    new_tokens = find_new_tokens(SEQUENCES, ALPHABET)
    assert new_tokens == ["C[UNIMOD:4]", "R[UNIMOD:7]", "[UNIMOD:1]-"]
    assert find_new_tokens(SEQUENCES, ALPHABET, with_termini=False) == new_tokens[:2]

    extended = extend_alphabet(ALPHABET, [*new_tokens, "A"])  # "A" is known: skipped
    assert {token: extended[token] for token in ALPHABET} == ALPHABET
    assert [extended[token] for token in new_tokens] == [7, 8, 9]
    assert len(extended) == len(ALPHABET) + 3
    assert find_new_tokens(SEQUENCES, extended) == []
