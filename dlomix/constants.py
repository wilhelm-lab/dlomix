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

ALPHABET_ORDERED = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_UNMOD = {k: v for v, k in enumerate(ALPHABET_ORDERED, start=1)}
