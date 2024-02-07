from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset

from ..constants import ALPHABET_UNMOD
from .base import AbstractPeptideDataset
from .dataset_utils import EncodingScheme


class RetentionTimeDataset(AbstractPeptideDataset):
    DEFAULT_SPLIT_NAMES = ["train", "val", "test"]

    def __init__(
        self,
        data_source: Optional[Union[str, List]] = None,
        val_data_source: Optional[Union[str, List]] = None,
        test_data_source: Optional[Union[str, List]] = None,
        data_format: str = "parquet",
        sequence_column: str = "modified_sequence",
        label_column: str = "indexed_retention_time",
        val_ratio: float = 0.2,
        max_seq_len: Union[int, str] = 30,
        dataset_type: str = "tf",
        batch_size: int = 256,
        model_features: Optional[List[str]] = None,
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        vocab: Dict = ALPHABET_UNMOD,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.NO_MODS,
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
            features_to_extract,
            pad,
            padding_value,
            vocab,
            encoding_scheme,
        )

    @staticmethod
    def load_processed_dataset(
        dataset: Dataset,
        batch_size,
        sequence_column,
        columns: Optional[List] = None,
        label: Optional[str] = None,
    ):
        """
        For convenience, load hugging face dataset that is previously processed and ready to be used for training/inference
        """

        rt_dataset = RetentionTimeDataset(
            batch_size=batch_size,
            sequence_column=sequence_column,
            model_features=columns,
            label_column=label,
        )
        rt_dataset.dataset = dataset
        return rt_dataset
