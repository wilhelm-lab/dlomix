import abc
from typing import Optional

import numpy as np
from pyteomics.proforma import parse

from .sequence_utils import rebuild_proforma_sequence


class AbstractParser(abc.ABC):
    """
    Abstract class for Parsers that read sequences and split the modification information from the amino acids.
    The abstract method `_parse_sequence(self, sequence)` is to be implemented by child classes.
    """

    @abc.abstractmethod
    def _parse_sequence(self, sequence: str):
        """parse a single sequence and return amino acids and modifications as separate data structures.

        Args:
            sequence (str): a modified sequence
        """
        raise NotImplementedError("Not implemented.")

    def _take_first_modification_proforma_output(self, mods):
        # # take first non-null element (modification only) (applied to all modifications including n and c terminal)
        # # ensure it is a single element and not a string
        # return next(filter(lambda x: x is not None, mods), None)
        return [m[0].id if m is not None else -1 for m in mods]

    def _flatten_seq_mods(self, parsed_sequence: list):
        """helper function to flatten a list of tuples to two lists.

        Args:
            parsed_sequence (list): a list of tuples (Amino Acids, Modification) `[('A', None), ('B', Unimod:1), ('C', None)]`

        Returns:
            list: a list of two lists or tuples (one for Amino acids and the other for modifications). `[['A', 'B', 'C'], [None, Unimod:1, None]]`
        """
        seq, mods = [list(i) for i in zip(*parsed_sequence)]
        return seq, mods

    def parse_sequences(self, sequences):
        """a generic function to apply the implementation of `_parse_sequence` to a list of sequencens.

        Args:
            sequences (list): list of string sequences, possibly with modifications.

        Returns:
            tuple(list, list, list, list): sequences, modifications, n_terminal modifications, c_terminal modifications
        """
        seqs = []
        mods = []
        n_terms = []
        c_terms = []
        for seq in sequences:
            seq, mod, n, c = self._parse_sequence(seq)

            # build sequence as a string from Amino Acid list
            seq = "".join(seq)
            seqs.append(seq)

            mods.append(mod)

            n_terms.append(n)
            c_terms.append(c)
        seqs = np.array(seqs)

        mods = np.array(mods, dtype=object)
        n_terms = np.array(n_terms)
        c_terms = np.array(c_terms)
        return seqs, mods, n_terms, c_terms


class ProformaParser(AbstractParser):
    def __init__(
        self, build_naive_vocab: bool = False, base_vocab: Optional[dict] = None
    ):
        super().__init__()
        self.build_naive_vocab = build_naive_vocab
        if self.build_naive_vocab:
            self.extended_vocab = base_vocab.copy()

    def _parse_sequence(self, sequence):
        """Implementation for parsing sequences according to the Proforma notation based on the Unimod representation.

        Args:
            sequence (str): sequence of amino acids, possibly with modifications.
            N-term and C-term modifications have to be separated with a `-`. Example: `[Unimod:1]-ABC`

        Returns:
            tuple(list, list, list): output of `pyteomics.proforma.parse' with the n-term and c-term modifications
            extracted from the originally returned modifiers dict.
            More information: https://pyteomics.readthedocs.io/en/latest/api/proforma.html#pyteomics.proforma.parse
        """
        # returns tuple (list of tuples (AA, mods), and a dict with properties)
        parsed_sequence, terminal_mods_dict = parse(sequence)

        n_term_mods = terminal_mods_dict.get("n_term")
        c_term_mods = terminal_mods_dict.get("c_term")

        if n_term_mods:
            n_term_mods = n_term_mods.pop().id
        else:
            n_term_mods = -1
        if c_term_mods:
            c_term_mods = c_term_mods.pop().id
        else:
            c_term_mods = -1

        seq, mod = self._flatten_seq_mods(parsed_sequence)
        mod = self._take_first_modification_proforma_output(mod)

        if self.build_naive_vocab:
            proforma_splitted_sequence = rebuild_proforma_sequence(seq, mod)
            for s in proforma_splitted_sequence:
                if self.extended_vocab.get(s) is None:
                    print(
                        f"Adding new key: {s} to the vocabulary with index: {len(self.extended_vocab)}"
                    )

                    # we start from 1 in the alphabet mapping, hence the +1
                    self.extended_vocab[s] = len(self.extended_vocab) + 1

        return seq, mod, n_term_mods, c_term_mods
