import json
import pandas as pd


def read_parquet_file_pandas(filepath, parquet_engine):
    try:
        df = pd.read_parquet(filepath, engine=parquet_engine)
    except ImportError:
        raise ImportError(
            "Parquet engine is missing, please install fastparquet using pip or conda."
        )
    return df


def read_json_file(filepath):
    with open(filepath, "r") as j:
        json_dict = json.loads(j.read())
    return json_dict
