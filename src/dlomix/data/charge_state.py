from typing import Callable, Dict, List, Optional, Union

from ..constants import ALPHABET_UNMOD
from .dataset import PeptideDataset
from .dataset_utils import EncodingScheme


class ChargeStateDataset(PeptideDataset):
    """
    A dataset class for handling charge state prediction data.

    Args:
        data_source (Optional[Union[str, List]]): The path or list of paths to the data source file(s).
        val_data_source (Optional[Union[str, List]]): The path or list of paths to the validation data source file(s).
        test_data_source (Optional[Union[str, List]]): The path or list of paths to the test data source file(s).
        data_format (str): The format of the data source file(s). Default is "parquet".
        sequence_column (str): The name of the column containing the peptide sequences. Default is "modified_sequence".
        label_column (str): The name of the column containing the charge state labels. Default is "most_abundant_charge_by_count".
        val_ratio (float): The ratio of validation data to split from the training data. Default is 0.2.
        max_seq_len (Union[int, str]): The maximum length of the peptide sequences. Default is 30.
        dataset_type (str): The type of dataset to use. Default is "tf".
        batch_size (int): The batch size for training and evaluation. Default is 256.
        model_features (Optional[List[str]]): The list of features to use for the model. Default is None.
        dataset_columns_to_keep (Optional[List[str]]): The list of columns to keep in the dataset. Default is None.
        features_to_extract (Optional[List[Union[Callable, str]]]): The list of features to extract from the dataset. Default is None.
        pad (bool): Whether to pad the sequences to the maximum length. Default is True.
        padding_value (int): The value to use for padding. Default is 0.
        alphabet (Dict): The mapping of characters to integers for encoding the sequences. Default is ALPHABET_UNMOD.
        encoding_scheme (Union[str, EncodingScheme]): The encoding scheme to use for encoding the sequences. Default is EncodingScheme.UNMOD.
        processed (bool): Whether the data has been preprocessed. Default is False.
        enable_tf_dataset_cache (bool): Flag to indicate whether to enable TensorFlow Dataset caching (call `.cahce()` on the generate TF Datasets).
        disable_cache (bool): Whether to disable Hugging Face datasets caching. Default is True.
    """

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
        enable_tf_dataset_cache: bool = False,
        disable_cache: bool = True,
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
        )
