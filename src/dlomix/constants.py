DEFAULT_PARQUET_ENGINE = "pyarrow"

retention_time_pipeline_parameters = {
    "model_params": {"seq_length": 30},
    "data_params": {
        "seq_length": 30,
    },
    "trained_model_path": "../pretrained_models/retention_time/example_rtmodel/",
    "trained_model_zipfile_name": "rtmodel.zip",
    "trained_model_stats": [0.0, 1.0],
}

retention_time_pipeline_parameters.update(
    {
        "trained_model_url": "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop"
        + retention_time_pipeline_parameters["trained_model_path"].strip("..")
        + retention_time_pipeline_parameters["trained_model_zipfile_name"]
    }
)

_ALPHABET_UNMOD_ORDERED = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_UNMOD = {k: v for v, k in enumerate(_ALPHABET_UNMOD_ORDERED, start=1)}
ALPHABET_UNMOD.update({"[]-": 21, "-[]": 22})
ALPHABET_UNMOD.update({"[UNIMOD:737]-": 21})

_ALPHABET_NAIVE_MODS_ORDERED = ["C[UNIMOD:4]", "M[UNIMOD:35]"]
ALPHABET_NAIVE_MODS = {
    k: v
    for v, k in enumerate(_ALPHABET_NAIVE_MODS_ORDERED, start=len(ALPHABET_UNMOD) + 1)
}
ALPHABET_NAIVE_MODS.update(ALPHABET_UNMOD)

PTMS_ALPHABET = {
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
    "M": 11,
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
    "-[]": 22,
    "[UNIMOD:737]-": 56,
    "M[UNIMOD:35]": 23,
    "S[UNIMOD:21]": 24,
    "T[UNIMOD:21]": 25,
    "Y[UNIMOD:21]": 26,
    "R[UNIMOD:7]": 27,
    "Q[UNIMOD:7]": 4,
    "N[UNIMOD:7]": 3,
    "K[UNIMOD:1]": 28,
    "K[UNIMOD:121]": 29,
    "Q[UNIMOD:28]": 30,
    "R[UNIMOD:34]": 31,
    "K[UNIMOD:34]": 32,
    "T[UNIMOD:43]": 35,
    "S[UNIMOD:43]": 36,
    "C[UNIMOD:4]": 37,
    "[UNIMOD:1]-": 38,
    "E[UNIMOD:27]": 39,
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
    "K[UNIMOD:737]": 55,
    "R[UNIMOD:36]": 51,
    "R[UNIMOD:36a]": 52,
    "P[UNIMOD:35]": 53,
    "Y[UNIMOD:354]": 54,
}
