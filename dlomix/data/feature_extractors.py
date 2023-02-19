import abc


class SequenceFeatureExtractor(abc.ABC):
    def __init__(self):
        super(SequenceFeatureExtractor, self).__init__()

    @abc.abstractmethod
    def extract(self, seq, mods, **kwargs):
        pass

    def extract_all(self, sequences, modifications):
        features = []
        for seq, mods in zip(sequences, modifications):
            feature = self.extract(seq, mods)
            features.append(feature)
        return features
