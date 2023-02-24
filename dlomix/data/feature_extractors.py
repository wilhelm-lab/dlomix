import abc


class SequenceFeatureExtractor(abc.ABC):
    def __init__(self, pad_to_seq_length = False, padding_int=-1):
        super(SequenceFeatureExtractor, self).__init__()
        self.pad_to_seq_length = pad_to_seq_length
        self.padding_int = padding_int


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
            raise ValueError(f"Feature length ({len(single_feature)}) is longer than sequence length provided ({seq_length}).")
        
        padding_length = seq_length - feature_length
        single_feature += [self.padding_int] * padding_length

        return single_feature

class LengthFeature(SequenceFeatureExtractor):
    def __init__(self):
        super(LengthFeature, self).__init__()
    
    def extract(self, seq, mods, **kwargs):
        return len(seq)

class ModificationLocationFeature(SequenceFeatureExtractor):

    ALPHABET_PTMS = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10, "M": 11,     # amino acids
            "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, 
            "[]-": 21, "-[]": 22,                                                                         # termini
            "M[UNIMOD:35]": 23, "S[UNIMOD:21]": 24,  "T[UNIMOD:21]": 25, "Y[UNIMOD:21]": 26, "R[UNIMOD:7]": 27, 
            "K[UNIMOD:1]": 28, "K[UNIMOD:121]": 29,  "Q(gl)": 30, "R[UNIMOD:34]": 31, "K[UNIMOD:34]": 32, 
            "T(ga)": 33, "S(ga)": 34, "T(gl)": 35, "S(gl)": 36, "C[UNIMOD:4]": 37, 'E(gl)':39, "[ac]-": 38,
            "K[UNIMOD:36]": 40, "K[UNIMOD:37]": 41, "K[UNIMOD:122]": 42, "K[UNIMOD:58]": 43, "K[UNIMOD:1289]": 44, 
            "K[UNIMOD:747]": 45, "K[UNIMOD:64]": 46, "K[UNIMOD:1848]": 47, "K[UNIMOD:1363]": 48, "K[UNIMOD:1849]": 49,
            "K[UNIMOD:3]": 50, "R[UNIMOD:36]": 52, "P[UNIMOD:35]": 53, "Y[UNIMOD:354]": 54}

    DICT_PTM_MOD_ATOM = {
            23: 4,   # M(ox)  - S
            24: 3,   # S(ph)  - O
            25: 3,   # T(ph)  - O
            26: 3,   # Y(ph)  - O
            27: 1,   # R(ci)  - C
            28: 2,   # K(ac)  - N
            29: 2,   # K(gl)  - N
            30: 1,   # Q(gl)  - C
            31: 2,   # R(me)  - N
            32: 2,   # K(me)  - N
            33: 3,   # T(ga)  - O
            34: 3,   # S(ga)  - O
            35: 3,   # T(gl)  - O
            36: 3,   # S(gl)  - O
            37: 4,   # C(cam) - S
            38: 2,   # [ac]-  - N
            39: 1,   # E(gl)  - C
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
        print("mods AA: ", modified_aas)
        feature = [
            ModificationLocationFeature.DICT_PTM_MOD_ATOM
            .get(ModificationLocationFeature.ALPHABET_PTMS
                 .get(i, 0), 0) 
                 for i in modified_aas]
        
        return feature