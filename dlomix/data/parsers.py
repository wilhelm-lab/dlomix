import abc

from pyteomics.proforma import parse


class AbstractParser(abc.ABC):
    def __init__(self):
        super(AbstractParser, self).__init__()

    @abc.abstractmethod
    def parse_sequence(self, sequence):
        pass

    def parse_sequences_iterable(self, sequences):
        parsed_sequences = []
        for seq in sequences:
            parsed_sequences.append(self.parse_sequence(seq))


class ProformaParser(AbstractParser):
    def __init__(self):
        super().__init__()

    def parse_sequence(self, sequence):
        parsed_sequence, terminal_mods_dict = parse(
            sequence
        )  # returns tuple (list of tuples (AA, mods), and a dict with properties)
        n_term_mods = terminal_mods_dict.get("n_term")
        c_term_mods = terminal_mods_dict.get("c_term")
        return parsed_sequence, n_term_mods, c_term_mods
