import json

import pandas as pd


def read_parquet_file_pandas(filepath, parquet_engine):
    """
    Reads a Parquet file located at the given filepath using pandas and the specified Parquet engine.

    Parameters:
    -----------
    filepath : str
        The file path of the Parquet file to read.
    parquet_engine : str
        The name of the Parquet engine to use for reading the file.

    Returns:
    --------
    pandas.DataFrame
        A pandas DataFrame containing the data from the Parquet file.

    Raises:
    -------
    ImportError
        If the specified Parquet engine is missing, fastparquet must be installed.
    """
    try:
        df = pd.read_parquet(filepath, engine=parquet_engine)
    except ImportError:
        raise ImportError(
            "Parquet engine is missing, please install fastparquet using pip or conda."
        )
    return df


def read_json_file(filepath):
    """
    Reads a JSON file located at the given filepath and returns its contents as a dictionary.

    Parameters:
    -----------
    filepath : str
        The file path of the JSON file to read.

    Returns:
    --------
    dict
        A dictionary containing the contents of the JSON file.
    """
    with open(filepath, "r") as j:
        json_dict = json.loads(j.read())
    return json_dict
