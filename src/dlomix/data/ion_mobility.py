from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..constants import ALPHABET_UNMOD
from .dataset import PeptideDataset
from .dataset_config import DatasetConfig
from .dataset_utils import EncodingScheme


class IonMobilityDataset(PeptideDataset):
    """
    A dataset class for handling ion mobility data.

    Args:
        data_source (Optional[Union[str, List]]): The data source for the dataset. Defaults to None.
        val_data_source (Optional[Union[str, List]]): The validation data source for the dataset. Defaults to None.
        test_data_source (Optional[Union[str, List]]): The test data source for the dataset. Defaults to None.
        data_format (str): The format of the data source. Defaults to "parquet".
        sequence_column (str): The column name for the peptide sequence in the dataset. Defaults to "sequence_modified".
        label_column (str): The column name for ion mobility in the dataset. Defaults to ["ccs", "ccs_std"].
        val_ratio (float): The ratio of validation data to split from the main dataset. Defaults to 0.2.
        max_seq_len (Union[int, str]): The maximum sequence length allowed in the dataset. Defaults to 30.
        dataset_type (str): The type of dataset to use. Defaults to "tf". Fallback is to TensorFlow dataset tensors.
        batch_size (int): The batch size for the dataset. Defaults to 256.
        model_features (Optional[List[str]]): The features to use in the model. Defaults to ["charge", "mz"].
        dataset_columns_to_keep (Optional[List[str]]): The columns to keep in the dataset. Defaults to None.
        features_to_extract (Optional[List[Union[Callable, str]]]): The features to extract from the dataset. Defaults to None.
        pad (bool): Whether to pad sequences to the maximum length. Defaults to True.
        padding_value (int): The value to use for padding sequences. Defaults to 0.
        alphabet (Dict): The alphabet used for encoding sequences. Defaults to ALPHABET_UNMOD.
        with_termini (bool): Whether to add the N- and C-termini in the sequence column, even if they do not exist. Defaults to True.
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
        sequence_column: str = "sequence_modified",
        label_column: Union[str, List] = ["ccs", "ccs_std"],
        val_ratio: float = 0.1,
        max_seq_len: Union[int, str] = 50,
        dataset_type: str = "tf",
        batch_size: int = 256,
        model_features: Optional[List[str]] = ["charge", "mz"],
        dataset_columns_to_keep: Optional[List[str]] = None,
        features_to_extract: Optional[List[Union[Callable, str]]] = None,
        pad: bool = True,
        padding_value: int = 0,
        alphabet: Optional[Dict] = None,
        with_termini: bool = True,
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.NAIVE_MODS,
        processed: bool = False,
        enable_tf_dataset_cache: bool = False,
        disable_cache: bool = False,
        auto_cleanup_cache: bool = True,
        num_proc: Optional[int] = None,
        batch_processing_size: int = 1000,
    ):
        kwargs = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super().__init__(DatasetConfig(**kwargs))


def reduced_mobility_to_ccs(
    one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15
):
    """
    convert reduced ion mobility (1/k0) to CCS
    :param one_over_k0: reduced ion mobility
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (
        np.sqrt(reduced_mass * (temp + t_diff)) * 1 / one_over_k0
    )


def ccs_to_one_over_reduced_mobility(
    ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15
):
    """
    convert CCS to 1 over reduced ion mobility (1/k0)
    :param ccs: collision cross-section
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas (N2)
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (
        SUMMARY_CONSTANT * charge
    )


def get_sqrt_weights_and_biases(
    mz: NDArray,
    charge: NDArray,
    ccs: NDArray,
    fit_charge_state_one: bool = True,
    max_charge: int = 4,
) -> Tuple[NDArray, NDArray]:
    """
    Fit a sqrt function to the data and return the weights and biases,
    used to parameterize the init layer for the CCS prediction model.
    Args:
        mz: Array of mass-over-charge values
        charge: Array of charge states
        ccs: Array of collision cross-section values
        fit_charge_state_one: Whether to fit the charge state 1 or not (should be set to false if
        your data does not contain charge state 1)
        max_charge: Maximum charge state to consider

    Returns:
        Tuple of weights and biases the initial projection layer can be parameterized with
    """
    from scipy.optimize import curve_fit

    if fit_charge_state_one:
        slopes, intercepts = [], []
    else:
        slopes, intercepts = [0.0], [0.0]

    c_begin = 1 if fit_charge_state_one else 2

    for c in range(c_begin, max_charge + 1):

        def fit_func(x, a, b):
            return a * np.sqrt(x) + b

        mask = charge == c
        mz_tmp = mz[mask]
        ccs_tmp = ccs[mask]

        popt, _ = curve_fit(fit_func, mz_tmp, ccs_tmp)

        slopes.append(popt[0])
        intercepts.append(popt[1])

    return np.array(slopes, np.float32), np.array(intercepts, np.float32)
