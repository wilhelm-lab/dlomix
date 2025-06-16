from typing import Callable, Dict, List, Optional, Union

from ..constants import ALPHABET_UNMOD
from .dataset import PeptideDataset
from .dataset_config import DatasetConfig
from .dataset_utils import EncodingScheme


class DetectabilityDataset(PeptideDataset):
    """
    A dataset class for handling Detectability prediction data.

    Args:
        data_source (Optional[Union[str, List]]): The path or list of paths to the data source file(s).
        val_data_source (Optional[Union[str, List]]): The path or list of paths to the validation data source file(s).
        test_data_source (Optional[Union[str, List]]): The path or list of paths to the test data source file(s).
        data_format (str): The format of the data source file(s). Default is "parquet".
        sequence_column (str): The name of the column containing the peptide sequences. Default is "Sequences".
        label_column (str): The name of the column containing the class labels. Default is "Classes".
        val_ratio (float): The ratio of validation data to split from the training data. Default is 0.2.
        max_seq_len (Union[int, str]): The maximum length of the peptide sequences. Default is 30.
        dataset_type (str): The type of dataset to use. Default is "tf". Fallback is to TensorFlow dataset tensors.
        batch_size (int): The batch size for training and evaluation. Default is 256.
        model_features (Optional[List[str]]): The list of features to use for the model. Default is None.
        dataset_columns_to_keep (Optional[List[str]]): The list of columns to keep in the dataset. Default is ["Proteins"].
        features_to_extract (Optional[List[Union[Callable, str]]]): The list of features to extract from the dataset. Default is None.
        pad (bool): Whether to pad the sequences to the maximum length. Default is True.
        padding_value (int): The value to use for padding. Default is 0.
        alphabet (Dict): The mapping of characters to integers for encoding the sequences. Default is ALPHABET_UNMOD.
        with_termini (bool): Whether to add the N- and C-termini in the sequence column, even if they do not exist. Defaults to True.
        encoding_scheme (Union[str, EncodingScheme]): The encoding scheme to use for encoding the sequences. Default is EncodingScheme.UNMOD.
        processed (bool): Whether the data has been preprocessed. Default is False.
        enable_tf_dataset_cache (bool): Flag to indicate whether to enable TensorFlow Dataset caching (call `.cahce()` on the generate TF Datasets).
        disable_cache (bool): Whether to disable Hugging Face datasets caching. Default is False.
    """

    def __init__(
        self,
        data_source: Optional[Union[str, List]] = None,
        val_data_source: Optional[Union[str, List]] = None,
        test_data_source: Optional[Union[str, List]] = None,
        data_format: str = "csv",
        sequence_column: str = "Sequences",
        label_column: str = "Classes",
        val_ratio: float = 0.2,
        max_seq_len: Union[int, str] = 40,
        dataset_type: str = "tf",
        batch_size: int = 256,
        model_features: Optional[List[str]] = None,
        dataset_columns_to_keep: Optional[List[str]] = ["Proteins"],
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        alphabet: Dict = ALPHABET_UNMOD,
        with_termini: bool = True,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.UNMOD,
        processed: bool = False,
        enable_tf_dataset_cache: bool = False,
        disable_cache: bool = False,
        auto_cleanup_cache: bool = True,
        num_proc: Optional[int] = None,
        batch_processing_size: int = 1000,
        **kwargs,
    ):
        config_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "kwargs"]
        }
        super().__init__(DatasetConfig(**config_kwargs), **kwargs)
