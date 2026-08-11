from typing import Callable, Dict, List, Optional, Union

from .dataset import PeptideDataset
from .dataset_config import DatasetConfig
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
        val_ratio (Optional[float]): Fraction of data for the validation split. None or 0 means no val split. Defaults to None.
        max_seq_len (Union[int, str]): The maximum sequence length allowed in the dataset. Defaults to 30.
        dataset_type (str): The type of dataset to use. Defaults to None, which resolves to "pt" or "tf" based on the active DLOMIX_BACKEND.
        batch_size (int): The batch size for the dataset. Defaults to 256.
        shuffle (bool): Whether to shuffle the data. Default is False.
        model_features (Optional[List[str]]): The features to use in the model. Defaults to None.
        dataset_columns_to_keep (Optional[List[str]]): The columns to keep in the dataset. Defaults to None.
        features_to_extract (Optional[List[Union[Callable, str]]]): The features to extract from the dataset. Defaults to None.
        pad (bool): Whether to pad sequences to the maximum length. Defaults to True.
        padding_value (str): The value to use for padding sequences. Defaults to '-'.
        alphabet (Optional[Dict]): The alphabet used for encoding sequences. Defaults to None to trigger learning the alphabet.
        with_termini (bool): Whether to add the N- and C-termini in the sequence column, even if they do not exist. Defaults to True.
        encoding_scheme (Union[str, EncodingScheme]): The encoding scheme to use for sequences. Defaults to EncodingScheme.UNMOD.
        processed (bool): Whether the dataset has been preprocessed. Defaults to False.
        enable_tf_dataset_cache (bool): Flag to indicate whether to enable TensorFlow Dataset caching (call `.cache()` on the generated TF Datasets).
        disable_cache (bool): Whether to disable Hugging Face datasets caching. Default is False.
        auto_cleanup_cache (bool): Whether to automatically clean up the cache. Default is True.
        num_proc (Optional[int]): Number of processes to use for dataset processing. Use -1 for all available processors, None for single-process mode, or a positive integer. Default is -1.
        batch_processing_size (int): Size of batches for processing. Default is 1000.
        torch_dataloader_kwargs (Optional[Dict]): Additional keyword arguments to pass to PyTorch DataLoader. Default is None.
        split_strategy (Optional[str]): Strategy for splitting the dataset. Options: 'random', 'sequence_unique', 'stratified'. Default is 'random'.
        split_seed (Optional[int]): Random seed for reproducible splits. Default is None.
        test_ratio (Optional[float]): Ratio of test data for three-way splits. If None, only train/val split is performed. Default is None.
        stratify_by_column (Optional[str]): Column name for stratified splitting. Can be a label column or feature column. Default is None.
    """

    def __init__(
        self,
        data_source: Optional[Union[str, List]] = None,
        val_data_source: Optional[Union[str, List]] = None,
        test_data_source: Optional[Union[str, List]] = None,
        data_format: str = "parquet",
        sequence_column: str = "modified_sequence",
        label_column: str = "indexed_retention_time",
        val_ratio: Optional[float] = None,
        max_seq_len: Union[int, str] = 30,
        dataset_type: Optional[str] = None,
        batch_size: int = 256,
        shuffle: bool = False,
        model_features: Optional[List[str]] = None,
        dataset_columns_to_keep: Optional[List[str]] = None,
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: str = "-",
        alphabet: Optional[Dict] = None,
        with_termini: bool = True,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.UNMOD,
        processed: bool = False,
        enable_tf_dataset_cache: bool = False,
        disable_cache: bool = False,
        auto_cleanup_cache: bool = True,
        num_proc: Optional[int] = -1,
        batch_processing_size: int = 1000,
        torch_dataloader_kwargs: Optional[Dict] = None,
        # New splitting parameters
        split_strategy: Optional[str] = "random",
        split_seed: Optional[int] = None,
        test_ratio: Optional[float] = None,
        stratify_by_column: Optional[str] = None,
        **kwargs,
    ):
        config_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "kwargs"]
        }
        super().__init__(DatasetConfig(**config_kwargs), **kwargs)
