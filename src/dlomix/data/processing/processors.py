import abc
import re
from typing import Optional


class PeptideDatasetBaseProcessor(abc.ABC):
    """
    Base class for peptide dataset processors.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    batched : bool (default=False)
        Whether to process data in batches.
    """

    def __init__(self, sequence_column_name: str = "", batched: bool = False):
        self.sequence_column_name = sequence_column_name
        self.batched = batched

        if self.batched:
            self._process_fn = self.batch_process
        else:
            self._process_fn = self.single_process

    def process(self, input_data, **kwargs):
        return self._process_fn(input_data, **kwargs)

    def __call__(self, input_data, **kwargs):
        return self.process(input_data, **kwargs)

    def __repr__(self):
        members = [
            attr
            for attr in vars(self)
            if not callable(getattr(self, attr)) and not attr.startswith(("_", "__"))
        ]
        values = [self.__getattribute__(m) for m in members]

        repr_str = ", ".join(
            [
                f"{m}='{v}'" if isinstance(v, str) else f"{m}={v}"
                for m, v in zip(members, values)
            ]
        )

        return f"{self.__class__.__name__}({repr_str})"

    @abc.abstractmethod
    def batch_process(self, input_data, **kwargs):
        pass

    @abc.abstractmethod
    def single_process(self, input_data, **kwargs):
        pass


class SequenceParsingProcessor(PeptideDatasetBaseProcessor):
    """
    Processor for parsing peptide sequences in ProForma format.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    batched : bool (default=False)
        Whether to process data in batches.
    with_termini : bool (default=True)
        Whether to add terminal modifications (also in case they do not exists, []- and -[]) to sequence column and overwrite it.

    Attributes
    ----------
    PARSED_COL_NAMES : dict
        Dictionary mapping the parsed sequence parts to new column names.

    Examples
    --------
    >>> processor = SequenceParsingProcessor("sequence")
    >>> data = {"sequence": "[]-IGGPC[UNIMOD:4]AHC[UNIMOD:4]AAWEGVR-[]"}
    >>> processor(data)
    {'sequence': ['[]', 'I', 'G', 'G', 'P', 'C', 'A', 'H', 'C', 'A', 'A', 'W', 'E', 'G', 'V', 'R', '-[]'],
     '_parsed_sequence': ['I', 'G', 'G', 'P', 'C', 'A', 'H', 'C', 'A', 'A', 'W', 'E', 'G', 'V', 'R'],
     '_n_term_mods': '[]-',
     '_c_term_mods': '-[]'}

    """

    PARSED_COL_NAMES = {
        "seq": "_parsed_sequence",
        "n_term": "_n_term_mods",
        "c_term": "_c_term_mods",
    }

    def __init__(
        self,
        sequence_column_name: str,
        batched: bool = False,
        with_termini: bool = True,
    ):
        super().__init__(sequence_column_name, batched)
        self.with_termini = with_termini

        # decide on sequence update function -> avoid conditional in function/loop
        if self.with_termini:
            self._assign_sequence_column = self.__update_sequence_column_with_termini
        else:
            self._assign_sequence_column = self.__update_sequence_column_without_termini

    def _parse_proforma_sequence(self, sequence_string):
        splitted = sequence_string.split("-")
        seq = ""

        if len(splitted) == 1:
            n_term, seq, c_term = "[]-", splitted[0], "-[]"
        elif len(splitted) == 2:
            if splitted[0].startswith("[UNIMOD:"):
                n_term, seq, c_term = splitted[0] + "-", splitted[1], "-[]"
            else:
                n_term, seq, c_term = "[]-", splitted[0], "-" + splitted[1]
        elif len(splitted) == 3:
            n_term, seq, c_term = splitted
            n_term += "-"
            c_term = "-" + c_term
        else:
            raise ValueError(
                f"Invalid sequence format: {sequence_string}. Expected format: [N-term]-[sequence]-[C-term]."
            )

        seq = re.findall(r"[A-Za-z](?:\[UNIMOD:\d+\])*|[^\[\]]", seq)
        return n_term, seq, c_term

    def __update_sequence_column_with_termini(self, n_terms, seq, c_terms):
        return [n_terms] + seq + [c_terms]

    def __update_sequence_column_without_termini(self, n_terms, seq, c_terms):
        return seq

    def batch_process(self, input_data, **kwargs):
        for new_column in SequenceParsingProcessor.PARSED_COL_NAMES.values():
            input_data[new_column] = []

        for index, sequence in enumerate(input_data[self.sequence_column_name]):
            n_terms, seq, c_terms = self._parse_proforma_sequence(sequence)
            input_data[SequenceParsingProcessor.PARSED_COL_NAMES["seq"]].append(seq)
            input_data[SequenceParsingProcessor.PARSED_COL_NAMES["n_term"]].append(
                n_terms
            )
            input_data[SequenceParsingProcessor.PARSED_COL_NAMES["c_term"]].append(
                c_terms
            )

            # Replace the original sequence with the parsed sequence + terminal mods or parsed sequence only
            input_data[self.sequence_column_name][index] = self._assign_sequence_column(
                n_terms, seq, c_terms
            )

        return input_data

    def single_process(self, input_data, **kwargs):
        n_terms, seq, c_terms = self._parse_proforma_sequence(
            input_data[self.sequence_column_name]
        )
        input_data[SequenceParsingProcessor.PARSED_COL_NAMES["seq"]] = seq
        input_data[SequenceParsingProcessor.PARSED_COL_NAMES["n_term"]] = n_terms
        input_data[SequenceParsingProcessor.PARSED_COL_NAMES["c_term"]] = c_terms

        # Replace the original sequence with the parsed sequence + terminal mods or parsed sequence only
        input_data[self.sequence_column_name] = self._assign_sequence_column(
            n_terms, seq, c_terms
        )

        return input_data


class SequencePaddingProcessor(PeptideDatasetBaseProcessor):
    """
    Processor for padding peptide sequences to a fixed length.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    batched : bool (default=False)
        Whether to process data in batches.
    padding_value : int (default=0)
        Value to use for padding the sequences.
    max_length : int (default=30)
        Maximum length of the sequences.

    Attributes
    ----------
    KEEP_COLUMN_NAME : str
        Name of the column indicating whether to keep the sequence or not (truncated sequences should be dropped from train/val sets).
    """

    KEEP_COLUMN_NAME = "_KEEP"

    def __init__(
        self,
        sequence_column_name: str,
        batched: bool = False,
        padding_value: int = 0,
        max_length: int = 30,
    ):
        super().__init__(sequence_column_name, batched)

        self.padding_value = padding_value
        self.max_length = max_length

    def batch_process(self, input_data, **kwargs):
        input_data[SequencePaddingProcessor.KEEP_COLUMN_NAME] = [True] * len(
            input_data[self.sequence_column_name]
        )

        for index, sequence in enumerate(input_data[self.sequence_column_name]):
            (
                input_data[self.sequence_column_name][index],
                keep_sequence,
            ) = self._pad_sequence(sequence)
            input_data[SequencePaddingProcessor.KEEP_COLUMN_NAME][index] = keep_sequence

        return input_data

    def single_process(self, input_data, **kwargs):
        input_data[self.sequence_column_name], keep_sequence = self._pad_sequence(
            input_data[self.sequence_column_name]
        )

        input_data[SequencePaddingProcessor.KEEP_COLUMN_NAME] = keep_sequence
        return input_data

    def _pad_sequence(self, sequence):
        length = len(sequence)
        if length <= self.max_length:
            return sequence + [self.padding_value] * (self.max_length - length), True
        else:
            return sequence[: self.max_length], False


class SequenceEncodingProcessor(PeptideDatasetBaseProcessor):
    """
    Processor for encoding peptide sequences using an alphabet.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    alphabet : dict (default=None)
        Dictionary mapping amino acids to integers. If None, the alphabet will be learned from the data.
    batched : bool (default=False)
        Whether to process data in batches.
    """

    def __init__(
        self,
        sequence_column_name: str,
        alphabet: Optional[dict] = None,
        batched: bool = False,
        extend_alphabet: bool = False,
        unknown_token: int = 0,
        fallback_unmodified: bool = False,
    ):
        super().__init__(sequence_column_name, batched)

        self.extend_alphabet = extend_alphabet

        #  TODO: consider not starting from 0 when learning the alphabet
        #  TODO: consider adding the padding token to the alphabet
        self.alphabet = {str(unknown_token): 1}
        self.set_alphabet(alphabet)
        self.set_fallback(fallback_unmodified)

        self.unknown_token = unknown_token

    def set_alphabet(self, alphabet):
        if alphabet and not self.extend_alphabet:
            self.alphabet = alphabet
            self._encode = self._encode_with_vocab
        else:
            self._encode = self._encode_learn_vocab

    def set_fallback(self, fallback_unmodified):
        self.fallback_unmodified = fallback_unmodified
        if self.fallback_unmodified:
            self._encode = self._encode_with_vocab_fallback
            if len(self.alphabet) == 0:
                raise ValueError(
                    "Alphabet must be provided if fallback_unmodified is True, to encode unseen modifications with the respective unmodified amino acid token."
                )

    def batch_process(self, input_data, **kwargs):
        return {
            self.sequence_column_name: [
                self._encode(seq) for seq in input_data[self.sequence_column_name]
            ]
        }

    def single_process(self, input_data, **kwargs):
        return {
            self.sequence_column_name: self._encode(
                input_data[self.sequence_column_name]
            )
        }

    def _encode_learn_vocab(self, sequence):
        encoded = []
        for amino_acid in sequence:
            if amino_acid not in self.alphabet:
                self.alphabet[amino_acid] = len(self.alphabet)
            encoded.append(self.alphabet.get(amino_acid))

        return encoded

    def _encode_with_vocab(self, sequence):
        encoded = [
            self.alphabet.get(amino_acid, self.unknown_token) for amino_acid in sequence
        ]
        return encoded

    def _encode_with_vocab_fallback(self, sequence):
        encoded = []
        for amino_acid in sequence:
            if amino_acid not in self.alphabet:
                if amino_acid.startswith(("[")):
                    amino_acid = "[]-"
                elif amino_acid.startswith(("-[")):
                    amino_acid = "-[]"
                else:
                    amino_acid = amino_acid[0]

            encoded.append(self.alphabet.get(amino_acid, self.unknown_token))

        return encoded


class SequencePTMRemovalProcessor(PeptideDatasetBaseProcessor):
    """
    Processor for removing PTMs from peptide sequences.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    batched : bool (default=False)
        Whether to process data in batches.
    """

    def __init__(self, sequence_column_name: str, batched: bool = False):
        super().__init__(sequence_column_name, batched)

    def batch_process(self, input_data, **kwargs):
        return {
            self.sequence_column_name: [
                self._remove_ptms(seq) for seq in input_data[self.sequence_column_name]
            ]
        }

    def single_process(self, input_data, **kwargs):
        return {
            self.sequence_column_name: self._remove_ptms(
                input_data[self.sequence_column_name]
            )
        }

    def _remove_ptms(self, sequence):
        if not isinstance(sequence, list):
            raise ValueError("Sequence must be a list of amino acids")
        n_terms = sequence[0]
        c_terms = sequence[-1]
        aa_sequence = sequence[1:-1]
        ptm_filtered_sequence = re.sub(r"\[UNIMOD:\d+\]", "", "".join(aa_sequence))
        return [n_terms] + list(ptm_filtered_sequence) + [c_terms]


class FunctionProcessor(PeptideDatasetBaseProcessor):
    """
    Processor for applying a function to the input data.

    Parameters
    ----------
    function : callable
        Function to apply to the input data.
    name : str (default="")
        Name of the processor.
    """

    def __init__(self, function, name: str = ""):
        super().__init__(batched=False)
        self.function = function
        self.name = name if name != "" else self.function.__name__

    def batch_process(self, input_data, **kwargs):
        raise NotImplementedError("FunctionProcessor does not support batch processing")

    def single_process(self, input_data, **kwargs):
        return self.function(input_data, **kwargs)
