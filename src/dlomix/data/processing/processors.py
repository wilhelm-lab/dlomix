import abc
import re


class PeptideDatasetBaseProcessor(abc.ABC):
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
    def __init__(
        self,
        sequence_column_name: str,
        batched: bool = False,
        new_column_names: list = ["raw_sequence", "n_terminal_mods", "c_terminal_mods"],
    ):
        super().__init__(sequence_column_name, batched)

        self.new_column_names = new_column_names

    def _parse_proforma_sequence(self, sequence_string):
        splitted = sequence_string.split("-")

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

        seq = re.findall(r"[A-Za-z](?:\[UNIMOD:\d+\])*|[^\[\]]", seq)
        return n_term, seq, c_term

    def _add_terminal_mods(self, seq, n_terms, c_terms):
        return [n_terms] + seq + [c_terms]

    def batch_process(self, input_data, **kwargs):
        for new_column in self.new_column_names:
            input_data[new_column] = []

        for index, sequence in enumerate(input_data[self.sequence_column_name]):
            n_terms, seq, c_terms = self._parse_proforma_sequence(sequence)
            input_data[self.new_column_names[0]].append(seq)
            input_data[self.new_column_names[1]].append(n_terms)
            input_data[self.new_column_names[2]].append(c_terms)

            input_data[self.sequence_column_name][index] = self._add_terminal_mods(
                seq, n_terms, c_terms
            )

        return input_data

    def single_process(self, input_data, **kwargs):
        n_terms, seq, c_terms = self._parse_proforma_sequence(
            input_data[self.sequence_column_name]
        )
        input_data[self.new_column_names[0]] = seq
        input_data[self.new_column_names[1]] = n_terms
        input_data[self.new_column_names[2]] = c_terms

        input_data[self.sequence_column_name] = self._add_terminal_mods(
            seq, n_terms, c_terms
        )

        return input_data


class SequencePaddingProcessor(PeptideDatasetBaseProcessor):
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
    def __init__(
        self, sequence_column_name: str, alphabet: dict, batched: bool = False
    ):
        super().__init__(sequence_column_name, batched)

        self.alphabet = alphabet

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

    def _encode(self, sequence):
        encoded = [self.alphabet.get(amino_acid) for amino_acid in sequence]

        return encoded


class SequencePTMRemovalProcessor(PeptideDatasetBaseProcessor):
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
    def __init__(self, function, name: str = ""):
        super().__init__(batched=False)
        self.function = function
        self.name = name if name != "" else self.function.__name__

    def batch_process(self, input_data, **kwargs):
        raise NotImplementedError("FunctionProcessor does not support batch processing")

    def single_process(self, input_data, **kwargs):
        return self.function(input_data, **kwargs)
