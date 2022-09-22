import numpy as np

class SequenceEncoder:
    def __init__(self, vocab_dict):
        super(SequenceEncoder, self).__init__()
        self.vocab_dict = vocab_dict
    
    def encode(self, sequences):
        encoded_sequences = []
        for seq in sequences:
            encoded_seq = np.array([self.vocab_dict.get(s, 0) for s in seq])
            encoded_sequences.append(encoded_seq)
        encoded_sequences = np.array(encoded_sequences)
        return encoded_sequences
