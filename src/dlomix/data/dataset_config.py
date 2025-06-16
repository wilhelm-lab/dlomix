import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from .dataset_utils import EncodingScheme


@dataclass
class DatasetConfig:
    """
    Configuration class for the dataset.
    """

    data_source: Union[str, List]
    val_data_source: Union[str, List]
    test_data_source: Union[str, List]
    data_format: str
    sequence_column: str
    label_column: List[str]
    val_ratio: float
    max_seq_len: int
    dataset_type: str
    batch_size: int
    model_features: List[str]
    dataset_columns_to_keep: Optional[List[str]]
    features_to_extract: Optional[List[Union[Callable, str]]]
    pad: bool
    padding_value: int
    alphabet: Dict
    with_termini: bool
    encoding_scheme: Union[str, EncodingScheme]
    processed: bool
    enable_tf_dataset_cache: bool
    disable_cache: bool
    auto_cleanup_cache: bool
    num_proc: Optional[int]
    batch_processing_size: int
    _additional_data: dict = field(default_factory=dict, compare=False)

    def save_config_json(self, path: str):
        """
        Save the configuration to a json file.

        Args:
            path (str): Path to the json file.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, default=repr)

    @staticmethod
    def load_config_json(path: str):
        """
        Load the configuration from a json file.

        Args:
            path (str): Path to the json file.

        Returns:
            DatasetConfig: The configuration object.
        """

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return DatasetConfig(**config)
