"""The tokens of peptide sequences, and alphabets extended by new ones.

A model knows the tokens of its alphabet: residues, modified residues and termini.
These helpers answer what a fine-tuning workflow needs to know about new data
before building a dataset: which tokens its sequences use, which of them the
model has not seen, and the alphabet that adds them without moving a known one.
"""

from typing import Dict, Iterable, List

from .processing.processors import parse_proforma_sequence

__all__ = ["sequence_tokens", "find_new_tokens", "extend_alphabet"]


def sequence_tokens(sequence: str, with_termini: bool = True) -> List[str]:
    """Split a ProForma sequence into its tokens, as the datasets encode it.

    Parameters
    ----------
    sequence:
        A peptide sequence, e.g. ``"[UNIMOD:1]-AC[UNIMOD:4]K-[]"`` or ``"PEPTIDE"``.
    with_termini:
        Whether to include the N- and C-terminal tokens (``"[]-"`` and ``"-[]"``
        when unmodified), as a dataset built with ``with_termini=True`` does.

    Examples
    --------
    >>> sequence_tokens("[UNIMOD:1]-AC[UNIMOD:4]K-[]")
    ['[UNIMOD:1]-', 'A', 'C[UNIMOD:4]', 'K', '-[]']
    """
    n_term, residues, c_term = parse_proforma_sequence(sequence)
    return [n_term, *residues, c_term] if with_termini else residues


def find_new_tokens(
    sequences: Iterable[str], alphabet: Dict[str, int], with_termini: bool = True
) -> List[str]:
    """The tokens used by ``sequences`` that ``alphabet`` does not have, sorted.

    Parameters
    ----------
    sequences:
        ProForma sequences, e.g. a dataset's sequence column.
    alphabet:
        The known tokens, e.g. a pretrained model's ``alphabet``.
    with_termini:
        Whether terminal tokens count; pass the dataset's ``with_termini``.
    """
    found = set()
    for sequence in set(sequences):
        found.update(sequence_tokens(sequence, with_termini))
    return sorted(found - alphabet.keys())


def extend_alphabet(alphabet: Dict[str, int], tokens: Iterable[str]) -> Dict[str, int]:
    """A copy of ``alphabet`` with ``tokens`` appended after its highest index.

    Known tokens keep their index, so a model trained with ``alphabet`` reads data
    encoded with the result as before; tokens the alphabet already has are skipped.
    """
    extended = dict(alphabet)
    for token in tokens:
        if token not in extended:
            extended[token] = max(extended.values(), default=-1) + 1
    return extended
