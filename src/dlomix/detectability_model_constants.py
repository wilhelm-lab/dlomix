import numpy as np

CLASSES_LABELS = ["Non-Flyer", "Weak Flyer", "Intermediate Flyer", "Strong Flyer"]

alphabet = [
    "0",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

aa_to_int_dict = dict((aa, i) for i, aa in enumerate(alphabet))

int_to_aa_dict = dict((i, aa) for i, aa in enumerate(alphabet))

padding_char = np.zeros(len(alphabet))
padding_char[0] = 1
