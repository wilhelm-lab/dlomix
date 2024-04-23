from typing import Callable, Dict, List, Optional, Union

from ..constants import ALPHABET_UNMOD
from .dataset import PeptideDataset
from .dataset_utils import EncodingScheme


class ChargeStateDataset(PeptideDataset):
    DEFAULT_SPLIT_NAMES = ["train", "val", "test"]

    def __init__(
        self,
        data_source: Optional[Union[str, List]] = None,
        val_data_source: Optional[Union[str, List]] = None,
        test_data_source: Optional[Union[str, List]] = None,
        data_format: str = "parquet",
        sequence_column: str = "modified_sequence",
        label_column: str = "most_abundant_charge_by_count",
        val_ratio: float = 0.2,
        max_seq_len: Union[int, str] = 30,
        dataset_type: str = "tf",
        batch_size: int = 256,
        model_features: Optional[List[str]] = None,
        dataset_columns_to_keep: Optional[List[str]] = None,
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        alphabet: Dict = ALPHABET_UNMOD,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.UNMOD,
        processed: bool = False,
    ):
        super().__init__(
            data_source,
            val_data_source,
            test_data_source,
            data_format,
            sequence_column,
            label_column,
            val_ratio,
            max_seq_len,
            dataset_type,
            batch_size,
            model_features,
            dataset_columns_to_keep,
            features_to_extract,
            pad,
            padding_value,
            alphabet,
            encoding_scheme,
            processed,
        )
