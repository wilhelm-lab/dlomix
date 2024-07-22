import warnings
import os
import requests
import importlib.resources as pkg_resources
from copy import deepcopy
import tensorflow as tf
import pyarrow.parquet as pq
from pathlib import Path

import dlomix
from dlomix.losses import masked_spectral_distance
from dlomix.data.fragment_ion_intensity import FragmentIonIntensityDataset
from dlomix.models.prosit import PrositIntensityPredictor

MODEL_FILENAME = 'prosit_baseline_model.keras'
MODEL_DIR = Path.home() / '.dlomix' / 'models'


def get_model_url():
    with pkg_resources.open_text(dlomix, 'prosit_baseline_model.txt') as url_file:
        return url_file.read().strip()
    

def download_model_from_github():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILENAME

    if model_path.exists():
        print('Using cached model.')
        return model_path
    
    print('Start downloading model from GitHub...')
    model_url = get_model_url()
    response = requests.get(model_url)
    response.raise_for_status()

    with open(model_path, 'wb') as f:
        f.write(response.content)
    
    print('Model downloaded successfully.')
    return model_path



def process_dataset(
        parquet_file_path: str,
        model_file_path: str = 'baseline',
        modifications: list = None,
        ion_types: list = None,
        label_column: str = 'intensities_raw',
        val_ratio: float = 0.2,
        test_ratio: float = 0.0
        ) -> tuple[FragmentIonIntensityDataset, PrositIntensityPredictor]:
    """Interface function for Oktoberfest package to correcly process a dataset and load a baseline model

    Processes the parquet file to a FragmentIonIntensityDataset, which is ready to be used for prediction and/or refinement/transfer learning
    The data splits can be investigated with dataset.hf_dataset.keys().
    If the label column is not present in the given parquet_file_path, the dataset can only be used for prediction.
    If the format of the given parquet file does not match the model format -> the user is warned that there may be refinement/transfer learning steps
    necessary. 

    Args:
        parquet_file_path (str): Path to the .parquet file which has the necessary data stored. 
            Necessary columns are: ['modified_sequence', 'precursor_charge_onehot', 'collision_energy_aligned_normed', 'method_nbr']
            Optional columns are: ['intensities_raw']
        model_file_path (str, optional): Either download the pre defined baseline model from github, or specify own local model path
            or a path to a PrositIntensityPredictor model saved with the .keras file format. Defaults to 'unmod_ext'.
        modifications (list, optional): A list of all modifications which are present in the dataset. Defaults to None.
        ion_types (list, optional): A list of the ion types which are present in the dataset. Defaults to ['y', 'b'].
        label_column (str, optional): The column identifier for where the intensity labels are, if there are any. Defaults to 'intensities_raw'.
        val_ratio (float, optional): A validation split ratio. Defaults to 0.2.
        test_ratio (float, optional): A test split ratio. Defaults to 0.0.

    Raises:
        ValueError: If the model_file_path does not have the .keras extension
        FileNotFoundError: if the model_file_path does not exist
        ValueError: If the parquet_file_path does not have the .parquet extension
        FileNotFoundError: If the parquet_file_path does not exist

    Returns:
        (FragmentIonIntensityDataset, PrositIntensityPredictor): 
            FragmentIonIntensityDataset: The fully processed dataset, which is ready to be used for prediction or transfer/refinement learning
            PrositIntensityPredictor: The loaded baseline model. Can only be used for prediction, if the data is compatible -> see warnings
    """
    
    
    modifications = [] if modifications is None else modifications
    ion_types = ['y', 'b'] if ion_types is None else ion_types

    # download the model file from github if the baseline model should be used, otherwise a model path can be specified
    if model_file_path == 'baseline':
        model_file_path = download_model_from_github()
    
    model = tf.keras.models.load_model(model_file_path)

    if not parquet_file_path.endswith('.parquet'):
        raise ValueError('The specified file is not a parquet file! Please specify a path with the .parquet extension.')
    if not Path(parquet_file_path).exists():
        raise FileNotFoundError('Specified parquet file was not found. Please specify a valid parquet file.')
    
    # check if intensities_raw column is in the parquet file
    inference_only = True    
    col_names = pq.read_schema(parquet_file_path).names
    if label_column in col_names:
        inference_only = False
    
    # get the differences between the model and the datset tokens 
    difference = set(modifications) - set(model.alphabet.keys())
    if not difference:
        new_alphabet = model.alphabet
    else:
        warnings.warn(
            """
            There are new tokens in the dataset, which are not supported by the loaded model.
            Either load a different model or transfer learning needs to be done.
            """)
        new_alphabet = deepcopy(model.alphabet)
        new_alphabet.update({k: i for i, k in enumerate(difference, start=len(new_alphabet) + 1)})

    # check for new ion types
    if any([ion_type in ['c', 'z', 'a', 'x'] for ion_type in ion_types]):
        if len(ion_types) == 2:
            warnings.warn(
                """
                Number of ions is the same as the loaded model supports, but the ion types are different.
                The model probably needs to be refined to achieve a better performance on these new ion types.
                """)
        if len(ion_types) > 2:
            if 'y' in ion_types and 'b' in ion_types:
                warnings.warn(
                    """
                    New Ion types in addition to y and b ions detected.
                    A new output layer is necessary, but it can keep trained weights for y and b ions.
                    """)
            else:
                warnings.warn(
                    """
                    Only new ion types are detected. A totally new output layer is necessary.
                    """
                )

    print('Start processing the dataset...')
    dataset = FragmentIonIntensityDataset(
        data_source=parquet_file_path,
        data_format='parquet',
        label_column=label_column,
        inference_only=inference_only,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        alphabet=new_alphabet,
        encoding_scheme='naive-mods',
        model_features=['precursor_charge_onehot', 'collision_energy_aligned_normed', 'method_nbr'],
        ion_types=ion_types,
    )

    print(f'The available data splits are: {", ".join(list(dataset.hf_dataset.keys()))}')

    return dataset, model
