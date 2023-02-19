import logging

import numpy as np

from dlomix.data.parsers import ProformaParser

NO_MODS_SEQUENCE = "ABC"
MOD_SEQUENCE = "AB[UNIMOD:1]C"
ALL_MODS_SEQUENCE = "[UNIMOD:2]-A[UNIMOD:1]B[UNIMOD:1]C[UNIMOD:1]-[UNIMOD:1]"

logger = logging.getLogger(__name__)

def test_parse_single_sequence():
    parser = ProformaParser()
    parsed_sequence, n_term_mods, c_term_mods = parser._parse_sequence(NO_MODS_SEQUENCE)
    assert parsed_sequence is not None
    assert len(parsed_sequence) == 3
    assert not all([n_term_mods, c_term_mods])

def test_parse_sequence_no_nc_terminals():
    parser = ProformaParser()
    parsed_sequence, n_term_mods, c_term_mods = parser._parse_sequence(MOD_SEQUENCE)
    assert parsed_sequence is not None
    assert len(parsed_sequence) == 3
    # modification exists at Amino Acid position 1
    assert parsed_sequence[1][1] is not None
    # modifications does not exist at other Amino Acid positions
    assert not all([parsed_sequence[0][1], parsed_sequence[2][1]])
    assert not all([n_term_mods, c_term_mods])

def test_parse_sequence_nc_terminals():
    parser = ProformaParser()
    parsed_sequence, n_term_mods, c_term_mods = parser._parse_sequence(ALL_MODS_SEQUENCE)
    logger.info(f"Sequence is: {parsed_sequence}")
    assert parsed_sequence is not None
    assert len(parsed_sequence) == 3
    assert all([parsed_sequence[1][1], parsed_sequence[0][1], parsed_sequence[2][1]])
    assert all([n_term_mods, c_term_mods])

def test_parse_sequences():
    parser = ProformaParser()
    sequences = [NO_MODS_SEQUENCE, MOD_SEQUENCE, ALL_MODS_SEQUENCE]

    seqs, mods, n_terms, c_terms = parser.parse_sequences(sequences)
    assert seqs is not None
    #, mods, n_terms, c_terms])
    print(seqs, mods, n_terms, c_terms)

def test_flatten_proforma_output():
    parser = ProformaParser()
    p, _, _ = parser._parse_sequence(MOD_SEQUENCE)
    seq, mod = parser._flatten_seq_mods(p)
    logger.info(seq)
    logger.info(mod)
    assert seq is not None
    assert mod is not None
