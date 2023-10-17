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

ALPHABET_UNMOD = {
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
}

ALPHABET_MOD = {
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
    "^": 21,
    "}": 22,
}

MCD_pipeline_parameters = {
    # params for full dataset with modified alphabet 
    #"model_params": {"seq_length": 30, "vocab_dict": ALPHABET_MOD.copy()},
    #"data_params": {"seq_length": 30, 'sequence_col': 'modified_sequence_single_letter', 'target_col': 'median'},
    "model_params": {"seq_length": 30, "vocab_dict": ALPHABET_UNMOD.copy()},
    "data_params": {"seq_length": 30,},
    #TODO upload model in dlomix and change path
    "base_model_path": "../data/models/rtmodel_prosit_epoch20",
    "test_set_path": "https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_test.csv"
}


QR_pipeline_parameters = {
    "model_params": {"seq_length": 30, "vocab_dict": ALPHABET_UNMOD.copy()},
    "data_params": {"seq_length": 30,},
    #specify path of desired model here
    "base_model_path": "../../data/models/rtmodel_prosit_epoch20",
    "train_val_path": 'https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_train_val.csv',
    "test_path": 'https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_test.csv',
}