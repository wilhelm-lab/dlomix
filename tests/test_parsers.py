import logging

import numpy as np

from dlomix.data.parsers import ProformaParser

NO_MODS_SEQUENCE = "ABC"
MOD_SEQUENCE = "AB[UNIMOD:1]C"
ALL_MODS_SEQUENCE = "[UNIMOD:2]-A[UNIMOD:1]B[UNIMOD:1]C[UNIMOD:1]-[UNIMOD:1]"
NO_MOD_VALUE = -1

logger = logging.getLogger(__name__)

from dlomix import __version__

logger.warn(__version__)


def test_parse_single_sequence():
    parser = ProformaParser()
    seq, mod, n_term_mods, c_term_mods = parser._parse_sequence(NO_MODS_SEQUENCE)
    assert seq is not None
    assert mod is not None
    assert len(seq) == 3
    assert len(mod) == 3
    assert n_term_mods == NO_MOD_VALUE
    assert c_term_mods is NO_MOD_VALUE


def test_parse_sequence_no_nc_terminals():
    parser = ProformaParser()
    seq, mod, n_term_mods, c_term_mods = parser._parse_sequence(MOD_SEQUENCE)
    assert seq is not None
    assert mod is not None
    assert len(seq) == 3
    assert len(mod) == 3
    # modification exists at Amino Acid position 1
    assert mod[1] is not None
    # modifications does not exist at other Amino Acid positions
    assert mod[0] is not None
    assert mod[2] is not None
    assert n_term_mods is not None
    assert c_term_mods is not None


def test_parse_sequence_nc_terminals():
    parser = ProformaParser()
    seq, mod, n_term_mods, c_term_mods = parser._parse_sequence(ALL_MODS_SEQUENCE)
    logger.info(f"Sequence is: {seq}")
    logger.info(f"Modification is: {mod}")
    logger.info(f"N Term: {n_term_mods}")
    logger.info(f"C Term: {c_term_mods}")
    assert seq is not None
    assert mod is not None
    assert len(seq) == 3
    assert len(mod) == 3
    assert mod[0] is not None
    assert mod[1] is not None
    assert mod[2] is not None
    assert n_term_mods is not None
    assert c_term_mods is not None


def test_parse_sequences():
    parser = ProformaParser()
    sequences = [NO_MODS_SEQUENCE, MOD_SEQUENCE, ALL_MODS_SEQUENCE]

    seqs, mods, n_terms, c_terms = parser.parse_sequences(sequences)
    assert seqs is not None
    # , mods, n_terms, c_terms])
    logger.info(seqs)
    logger.info(mods)
    logger.info(n_terms)
    logger.info(c_terms)


def test_flatten_proforma_output():
    parser = ProformaParser()
    seq, mod, _, _ = parser._parse_sequence(MOD_SEQUENCE)
    assert seq is not None
    assert mod is not None
