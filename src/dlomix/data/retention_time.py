from typing import Callable, Dict, List, Optional, Union

from ..constants import ALPHABET_UNMOD
from .dataset import PeptideDataset
from .dataset_utils import EncodingScheme


class RetentionTimeDataset(PeptideDataset):
    """
    A dataset class for handling retention time data.

    Args:
        data_source (Optional[Union[str, List]]): The data source for the dataset. Defaults to None.
        val_data_source (Optional[Union[str, List]]): The validation data source for the dataset. Defaults to None.
        test_data_source (Optional[Union[str, List]]): The test data source for the dataset. Defaults to None.
        data_format (str): The format of the data source. Defaults to "parquet".
        sequence_column (str): The column name for the peptide sequence in the dataset. Defaults to "modified_sequence".
        label_column (str): The column name for the retention time label in the dataset. Defaults to "indexed_retention_time".
        val_ratio (float): The ratio of validation data to split from the main dataset. Defaults to 0.2.
        max_seq_len (Union[int, str]): The maximum sequence length allowed in the dataset. Defaults to 30.
        dataset_type (str): The type of dataset to use. Defaults to "tf".
        batch_size (int): The batch size for the dataset. Defaults to 256.
        model_features (Optional[List[str]]): The features to use in the model. Defaults to None.
        dataset_columns_to_keep (Optional[List[str]]): The columns to keep in the dataset. Defaults to None.
        features_to_extract (Optional[List[Union[Callable, str]]]): The features to extract from the dataset. Defaults to None.
        pad (bool): Whether to pad sequences to the maximum length. Defaults to True.
        padding_value (int): The value to use for padding sequences. Defaults to 0.
        alphabet (Dict): The alphabet used for encoding sequences. Defaults to ALPHABET_UNMOD.
        encoding_scheme (Union[str, EncodingScheme]): The encoding scheme to use for sequences. Defaults to EncodingScheme.UNMOD.
        processed (bool): Whether the dataset has been preprocessed. Defaults to False.
        enable_tf_dataset_cache (bool): Flag to indicate whether to enable TensorFlow Dataset caching (call `.cahce()` on the generate TF Datasets).
        disable_cache (bool): Whether to disable Hugging Face datasets caching. Default is False.
    """

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
        dataset_columns_to_keep: Optional[List[str]] = None,
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        alphabet: Dict = ALPHABET_UNMOD,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.UNMOD,
        processed: bool = False,
        enable_tf_dataset_cache: bool = False,
        disable_cache: bool = False,
        auto_cleanup_cache: bool = True,
        num_proc: Optional[int] = None,
        batch_processing_size: int = 1000,
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
            enable_tf_dataset_cache,
            disable_cache,
            auto_cleanup_cache,
            num_proc,
            batch_processing_size,
        )
