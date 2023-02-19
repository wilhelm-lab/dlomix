import abc

import numpy as np
from pyteomics.proforma import parse


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
        pass

    def _take_first_modification_proforma_output(self, mods):
        # take first modification only (applied to all modifications including n and c terminal)
        return next(iter(mods), None)
    
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
            p, n, c = self._parse_sequence(seq)
            seq, mod = self._flatten_seq_mods(p)

            # build sequence as a string from Amino Acid list
            seq = ''.join(seq)
            seqs.append(seq)

            mod = self._take_first_modification_proforma_output(mod)
            mods.append(mod)

            if n is not None:
                n = self._take_first_modification_proforma_output(n)
            n_terms.append(n)

            if c is not None:
                c = self._take_first_modification_proforma_output(c)
            c_terms.append(c)
        return np.array(seqs), np.array(mods), np.array(n_terms), np.array(c_terms)
    

class ProformaParser(AbstractParser):
    def __init__(self):
        super().__init__()

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
        return parsed_sequence, n_term_mods, c_term_mods