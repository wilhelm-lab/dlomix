import abc

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
    "R[UNIMOD:36]": 52,
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


class LengthFeature(SequenceFeatureExtractor):
    def __init__(self):
        super(LengthFeature, self).__init__()

    def extract(self, seq, mods, **kwargs):
        return len(seq)


class ModificationLocationFeature(SequenceFeatureExtractor):

    DICT_PTM_MOD_ATOM = {
        23: 4,  # M(ox)  - S
        24: 3,  # S(ph)  - O
        25: 3,  # T(ph)  - O
        26: 3,  # Y(ph)  - O
        27: 1,  # R(ci)  - C
        28: 2,  # K(ac)  - N
        29: 2,  # K(gl)  - N
        30: 1,  # Q(gl)  - C
        31: 2,  # R(me)  - N
        32: 2,  # K(me)  - N
        33: 3,  # T(ga)  - O
        34: 3,  # S(ga)  - O
        35: 3,  # T(gl)  - O
        36: 3,  # S(gl)  - O
        37: 4,  # C(cam) - S
        38: 2,  # [ac]-  - N
        39: 1,  # E(gl)  - C
        40: 2,  # K(dme)   - N
        41: 2,  # K(tme)   - N
        42: 2,  # K(f)     - N
        43: 2,  # K(p)     - N
        44: 2,  # K(b)     - N
        45: 2,  # K(m)     - N
        46: 2,  # K(s)     - N
        47: 2,  # K(glu)   - N
        48: 2,  # K(cr)    - N
        49: 2,  # K(hib)   - N
        50: 2,  # K(bi)    - N
        51: 1,  # R(sdime) - C
        52: 2,  # R(adime) - N
        53: 1,  # P(h)     - C
        54: 1,  # Y(n)     - C
    }

    def __init__(self):
        super(ModificationLocationFeature, self).__init__(pad_to_seq_length=True)

    def extract(self, seq, mods, seq_length):
        modified_aas = [f"{s}[UNIMOD:{m}]" for s, m in zip(seq, mods)]
        feature = [
            ModificationLocationFeature.DICT_PTM_MOD_ATOM.get(
                ALPHABET_PTMS.get(i, 0), 0
            )
            for i in modified_aas
        ]

        return feature


class ModificationLossFeature(SequenceFeatureExtractor):

    DICT_PTM_ATOM_COUNT_LOSS = {
        #        H   C   N   O   P   S
        23: [0, 0, 0, 0, 0, 0],  # M(ox)    -
        24: [1, 0, 0, 0, 0, 0],  # S(ph)    -  H
        25: [1, 0, 0, 0, 0, 0],  # T(ph)    -  H
        26: [1, 0, 0, 0, 0, 0],  # Y(ph)    -  H
        27: [1, 0, 1, 0, 0, 0],  # R(ci)    -  H N
        28: [1, 0, 0, 0, 0, 0],  # K(ac)    -  H
        29: [1, 0, 0, 0, 0, 0],  # K(gl)    -  H
        30: [
            9,
            4,
            2,
            1,
            0,
            0,
        ],  # Q(gl)    -  C(4) H(9) N(2) O                    * fixed
        31: [1, 0, 0, 0, 0, 0],  # R(me)    -  H
        32: [1, 0, 0, 0, 0, 0],  # K(me)    -  H
        33: [1, 0, 0, 0, 0, 0],  # T(ga)    -  H
        34: [1, 0, 0, 0, 0, 0],  # S(ga)    -  H
        35: [1, 0, 0, 0, 0, 0],  # T(gl)    -  H
        36: [1, 0, 0, 0, 0, 0],  # S(gl)    -  H
        37: [1, 0, 0, 0, 0, 0],  # C(cam)   -  H
        38: [1, 0, 0, 0, 0, 0],  # [ac]-    -  H
        39: [8, 4, 1, 2, 0, 0],  # E(gl)    -  C(4) H(8) N O(2)     * new
        40: [2, 0, 0, 0, 0, 0],  # K(dme)   -  H(2)
        41: [3, 0, 0, 0, 0, 0],  # K(tme)   -  H(3)
        42: [1, 0, 0, 0, 0, 0],  # K(f)     -  H
        43: [1, 0, 0, 0, 0, 0],  # K(p)     -  H
        44: [1, 0, 0, 0, 0, 0],  # K(b)     -  H
        45: [1, 0, 0, 0, 0, 0],  # K(m)     -  H
        46: [1, 0, 0, 0, 0, 0],  # K(s)     -  H
        47: [1, 0, 0, 0, 0, 0],  # K(glu)   -  H
        48: [1, 0, 0, 0, 0, 0],  # K(cr)    -  H
        49: [1, 0, 0, 0, 0, 0],  # K(hib)   -  H
        50: [1, 0, 0, 0, 0, 0],  # K(bi)    -  H
        51: [3, 0, 2, 0, 0, 0],  # R(sdime) -  N(2) H(3)
        52: [2, 0, 0, 0, 0, 0],  # R(adime) -  H(2)
        53: [1, 0, 0, 0, 0, 0],  # P(h)     -  H
        54: [1, 0, 0, 0, 0, 0],  # Y(n)     -  H
    }

    def __init__(self):
        super(ModificationLossFeature, self).__init__(
            pad_to_seq_length=True, padding_element=[0, 0, 0, 0, 0, 0]
        )

    def extract(self, seq, mods, seq_length):
        modified_aas = [f"{s}[UNIMOD:{m}]" for s, m in zip(seq, mods)]
        feature = [
            ModificationLossFeature.DICT_PTM_ATOM_COUNT_LOSS.get(
                ALPHABET_PTMS.get(i, 0), [0] * 6
            )
            for i in modified_aas
        ]

        return feature
