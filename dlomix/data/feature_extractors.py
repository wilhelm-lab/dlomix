import abc

from ..utils import get_constructor_call_object_creation

ALPHABET_PTMS = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,  # amino acids
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "[]-": 21,
    "-[]": 22,  # termini
    "M[UNIMOD:35]": 23,
    "S[UNIMOD:21]": 24,
    "T[UNIMOD:21]": 25,
    "Y[UNIMOD:21]": 26,
    "R[UNIMOD:7]": 27,
    "K[UNIMOD:1]": 28,
    "K[UNIMOD:121]": 29,
    "Q(gl)": 30,
    "R[UNIMOD:34]": 31,
    "K[UNIMOD:34]": 32,
    "T(ga)": 33,
    "S(ga)": 34,
    "T(gl)": 35,
    "S(gl)": 36,
    "C[UNIMOD:4]": 37,
    "E(gl)": 39,
    "[ac]-": 38,
    "K[UNIMOD:36]": 40,
    "K[UNIMOD:37]": 41,
    "K[UNIMOD:122]": 42,
    "K[UNIMOD:58]": 43,
    "K[UNIMOD:1289]": 44,
    "K[UNIMOD:747]": 45,
    "K[UNIMOD:64]": 46,
    "K[UNIMOD:1848]": 47,
    "K[UNIMOD:1363]": 48,
    "K[UNIMOD:1849]": 49,
    "K[UNIMOD:3]": 50,
    "R[UNIMOD:36]": 51,
    "R[UNIMOD:36a]": 52,
    "P[UNIMOD:35]": 53,
    "Y[UNIMOD:354]": 54,
}


class SequenceFeatureExtractor(abc.ABC):
    def __init__(self, pad_to_seq_length=False, padding_element=-1):
        super(SequenceFeatureExtractor, self).__init__()
        self.pad_to_seq_length = pad_to_seq_length
        self.padding_element = padding_element

    @abc.abstractmethod
    def extract(self, seq, mods, **kwargs):
        pass

    def extract_all(self, sequences, modifications, seq_length=0):
        features = []
        for seq, mods in zip(sequences, modifications):
            feature = self.extract(seq, mods, seq_length=seq_length)
            if seq_length:
                feature = self.pad_feature_to_seq_length(feature, seq_length)
            features.append(feature)
        return features

    def pad_feature_to_seq_length(self, single_feature, seq_length=0):
        feature_length = len(single_feature)

        if feature_length > seq_length:
            raise ValueError(
                f"Feature length ({len(single_feature)}) is longer than sequence length provided ({seq_length})."
            )

        padding_length = seq_length - feature_length
        single_feature += [self.padding_element] * padding_length

        return single_feature

    def __repr__(self) -> str:
        return get_constructor_call_object_creation(self)


class LengthFeature(SequenceFeatureExtractor):
    def __init__(self):
        super(LengthFeature, self).__init__()

    def extract(self, seq, mods, **kwargs):
        return len(seq)


class ModificationLocationFeature(SequenceFeatureExtractor):

    DICT_PTM_MOD_ATOM = {
        "M[UNIMOD:35]": 4,
        "S[UNIMOD:21]": 3,
        "T[UNIMOD:21]": 3,
        "Y[UNIMOD:21]": 3,
        "R[UNIMOD:7]": 1,
        "K[UNIMOD:1]": 2,
        "K[UNIMOD:121]": 2,
        "Q(gl)": 1,
        "R[UNIMOD:34]": 2,
        "K[UNIMOD:34]": 2,
        "T(ga)": 3,
        "S(ga)": 3,
        "T(gl)": 3,
        "S(gl)": 3,
        "C[UNIMOD:4]": 4,
        "[ac]-": 2,
        "E(gl)": 1,
        "K[UNIMOD:36]": 2,
        "K[UNIMOD:37]": 2,
        "K[UNIMOD:122]": 2,
        "K[UNIMOD:58]": 2,
        "K[UNIMOD:1289]": 2,
        "K[UNIMOD:747]": 2,
        "K[UNIMOD:64]": 2,
        "K[UNIMOD:1848]": 2,
        "K[UNIMOD:1363]": 2,
        "K[UNIMOD:1849]": 2,
        "K[UNIMOD:3]": 2,
        "unknown": 1,
        "R[UNIMOD:36]": 2,
        "P[UNIMOD:35]": 1,
        "Y[UNIMOD:354]": 1,
    }

    def __init__(self):
        super(ModificationLocationFeature, self).__init__(pad_to_seq_length=True)

    def extract(self, seq, mods, seq_length):
        modified_aas = [f"{s}[UNIMOD:{m}]" for s, m in zip(seq, mods)]
        feature = [
            ModificationLocationFeature.DICT_PTM_MOD_ATOM.get(i, 0)
            for i in modified_aas
        ]

        return feature


class ModificationLossFeature(SequenceFeatureExtractor):

    PTM_LOSS_LOOKUP = {
        "M[UNIMOD:35]": [0, 0, 0, 0, 0, 0],
        "S[UNIMOD:21]": [1, 0, 0, 0, 0, 0],
        "T[UNIMOD:21]": [1, 0, 0, 0, 0, 0],
        "Y[UNIMOD:21]": [1, 0, 0, 0, 0, 0],
        "R[UNIMOD:7]": [1, 0, 1, 0, 0, 0],
        "K[UNIMOD:1]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:121]": [1, 0, 0, 0, 0, 0],
        "Q(gl)": [9, 4, 2, 1, 0, 0],
        "R[UNIMOD:34]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:34]": [1, 0, 0, 0, 0, 0],
        "T(ga)": [1, 0, 0, 0, 0, 0],
        "S(ga)": [1, 0, 0, 0, 0, 0],
        "T(gl)": [1, 0, 0, 0, 0, 0],
        "S(gl)": [1, 0, 0, 0, 0, 0],
        "C[UNIMOD:4]": [1, 0, 0, 0, 0, 0],
        "[ac]-": [1, 0, 0, 0, 0, 0],
        "E(gl)": [8, 4, 1, 2, 0, 0],
        "K[UNIMOD:36]": [2, 0, 0, 0, 0, 0],
        "K[UNIMOD:37]": [3, 0, 0, 0, 0, 0],
        "K[UNIMOD:122]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:58]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1289]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:747]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:64]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1848]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1363]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:1849]": [1, 0, 0, 0, 0, 0],
        "K[UNIMOD:3]": [1, 0, 0, 0, 0, 0],
        "unknown": [3, 0, 2, 0, 0, 0],
        "R[UNIMOD:36]": [2, 0, 0, 0, 0, 0],
        "P[UNIMOD:35]": [1, 0, 0, 0, 0, 0],
        "Y[UNIMOD:354]": [1, 0, 0, 0, 0, 0],
    }

    def __init__(self):
        super(ModificationLossFeature, self).__init__(
            pad_to_seq_length=True, padding_element=[0, 0, 0, 0, 0, 0]
        )

    def extract(self, seq, mods, seq_length):
        modified_aas = [f"{s}[UNIMOD:{m}]" for s, m in zip(seq, mods)]
        feature = [
            ModificationLossFeature.PTM_LOSS_LOOKUP.get(i, [0] * 6)
            for i in modified_aas
        ]

        return feature


class ModificationGainFeature(SequenceFeatureExtractor):

    PTM_GAIN_LOOKUP = {
        "M[UNIMOD:35]": [0, 0, 0, 1, 0, 0],
        "S[UNIMOD:21]": [2, 0, 0, 3, 1, 0],
        "T[UNIMOD:21]": [2, 0, 0, 3, 1, 0],
        "Y[UNIMOD:21]": [2, 0, 0, 3, 1, 0],
        "R[UNIMOD:7]": [0, 0, 0, 1, 0, 0],
        "K[UNIMOD:1]": [3, 2, 0, 1, 0, 0],
        "K[UNIMOD:121]": [7, 4, 2, 2, 0, 0],
        "Q(gl)": [6, 4, 1, 1, 0, 0],
        "R[UNIMOD:34]": [3, 1, 0, 0, 0, 0],
        "K[UNIMOD:34]": [3, 1, 0, 0, 0, 0],
        "T(ga)": [14, 8, 1, 5, 0, 0],
        "S(ga)": [14, 8, 1, 5, 0, 0],
        "T(gl)": [14, 8, 1, 5, 0, 0],
        "S(gl)": [14, 8, 1, 5, 0, 0],
        "C[UNIMOD:4]": [4, 2, 1, 1, 0, 0],
        "[ac]-": [3, 2, 0, 1, 0, 0],
        "E(gl)": [6, 4, 1, 1, 0, 0],
        "K[UNIMOD:36]": [6, 2, 0, 0, 0, 0],
        "K[UNIMOD:37]": [9, 3, 0, 0, 0, 0],
        "K[UNIMOD:122]": [0, 1, 0, 1, 0, 0],
        "K[UNIMOD:58]": [5, 3, 0, 1, 0, 0],
        "K[UNIMOD:1289]": [7, 4, 0, 1, 0, 0],
        "K[UNIMOD:747]": [3, 3, 0, 3, 0, 0],
        "K[UNIMOD:64]": [5, 4, 0, 3, 0, 0],
        "K[UNIMOD:1848]": [7, 5, 0, 3, 0, 0],
        "K[UNIMOD:1363]": [5, 4, 0, 1, 0, 0],
        "K[UNIMOD:1849]": [7, 4, 0, 2, 0, 0],
        "K[UNIMOD:3]": [15, 10, 2, 2, 0, 1],
        "unknown": [7, 2, 2, 0, 0, 0],
        "R[UNIMOD:36]": [6, 2, 0, 0, 0, 0],
        "P[UNIMOD:35]": [1, 0, 0, 1, 0, 0],
        "Y[UNIMOD:354]": [0, 0, 1, 2, 0, 0],
    }

    def __init__(self):
        super(ModificationGainFeature, self).__init__(
            pad_to_seq_length=True, padding_element=[0, 0, 0, 0, 0, 0]
        )

    def extract(self, seq, mods, seq_length):
        modified_aas = [f"{s}[UNIMOD:{m}]" for s, m in zip(seq, mods)]
        feature = [
            ModificationGainFeature.PTM_GAIN_LOOKUP.get(i, [0] * 6)
            for i in modified_aas
        ]

        return feature
