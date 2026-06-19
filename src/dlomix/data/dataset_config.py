import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from .dataset_utils import EncodingScheme, validate_num_proc_value


@dataclass
class DatasetConfig:
    """
    Configuration class for the dataset.

    Splitting Parameters
    --------------------
    val_ratio : Optional[float]
        Fraction of data for the validation split. None or 0 means no val split. Default None.
    split_strategy : Optional[str]
        Strategy for splitting the dataset. Options: 'random', 'sequence_unique', 'stratified'.
        Default is 'random'. Only used when automatic splitting is performed.
    split_seed : Optional[int]
        Random seed for reproducible splits. Default is None.
    test_ratio : Optional[float]
        Ratio of test data for three-way splits (0 < test_ratio < 1).
        If None, only train/val split is performed. Default is None.
    stratify_by_column : Optional[str]
        Column name for stratified splitting. Can be a label column or any feature column.
        Only used when split_strategy='stratified'. Default is None.
    """

    data_source: Union[str, List]
    val_data_source: Union[str, List]
    test_data_source: Union[str, List]
    data_format: str
    sequence_column: str
    label_column: List[str]
    max_seq_len: int
    dataset_type: str
    batch_size: int
    shuffle: bool
    model_features: List[str]
    dataset_columns_to_keep: Optional[List[str]]
    features_to_extract: Optional[List[Union[Callable, str]]]
    pad: bool
    padding_value: str
    alphabet: Dict
    with_termini: bool
    encoding_scheme: Union[str, EncodingScheme]
    processed: bool
    enable_tf_dataset_cache: bool
    disable_cache: bool
    auto_cleanup_cache: bool
    num_proc: Optional[int]
    batch_processing_size: int
    torch_dataloader_kwargs: Optional[Dict] = field(default_factory=dict)
    # Splitting parameters
    val_ratio: Optional[float] = None
    split_strategy: Optional[str] = "random"
    split_seed: Optional[int] = None
    test_ratio: Optional[float] = None
    stratify_by_column: Optional[str] = None

    # validate input parameters
    def __post_init__(self):
        # sequence length validation
        if self.max_seq_len <= 0:
            raise ValueError(
                f"Max sequence length provided is an integer but not a valid value: {self.max_seq_len}, only positive non-zero values are allowed."
            )

        # label column validation, either a string or a list of strings
        if not isinstance(self.label_column, (str, list)):
            raise ValueError(
                "The label_column parameter should be a string or a list of strings."
            )
        elif isinstance(self.label_column, str):
            self.label_column = [self.label_column]

        validate_num_proc_value(self.num_proc)

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
