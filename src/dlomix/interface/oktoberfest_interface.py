import logging
from pathlib import Path
import importlib.resources as pkg_resources
from copy import deepcopy
import requests
from tensorflow.keras.models import load_model
import pyarrow.parquet as pq


import dlomix
from dlomix.losses import masked_spectral_distance
from dlomix.data.fragment_ion_intensity import FragmentIonIntensityDataset
from dlomix.models.prosit import PrositIntensityPredictor

logger = logging.getLogger(__name__)
logger.propagate = False

MODEL_FILENAME = 'prosit_baseline_model.keras'
MODEL_DIR = Path.home() / '.dlomix' / 'models'


def get_model_url():
    with pkg_resources.open_text(dlomix, 'prosit_baseline_model.txt') as url_file:
        return url_file.read().strip()
    

def download_model_from_github() -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILENAME

    if model_path.exists():
        logger.info(f'Using cached model: {str(model_path)}')
        return model_path
    
    logger.info('Start downloading model from GitHub...')
    model_url = get_model_url()
    response = requests.get(model_url)
    response.raise_for_status()

    with open(model_path, 'wb') as f:
        f.write(response.content)
    
    print(f'Model downloaded successfully under {str(model_path)}')
    return str(model_path)


def load_keras_model(model_file_path: str = 'baseline') -> PrositIntensityPredictor:
    """Load a PrositIntensityPredictor model given a model file path. 

    Args:
        model_file_path (str): Path to a saved PrositIntensityPredictor model (.keras format). 
            If no path is given, automatically downloads the baseline model. Defaults to 'baseline'

    Raises:
        ValueError: If the model_file_path does not end with the .keras extension
        FileNotFoundError: If the given model_file_path does not exist

    Returns:
        PrositIntensityPredictor: A loaded PrositIntensityPredictor model, that can be used for predictions, refinement or transfer learning purposes.
    """

    # download the model file from github if the baseline model should be used, otherwise a model path can be specified
    if model_file_path == 'baseline':
        model_file_path = download_model_from_github()
        return load_model(model_file_path, compile=False)

    if not str(model_file_path).endswith('.keras'):
        raise ValueError('The given model file is not saved with the .keras format! Please specify a path with the .keras extension.')
    if not Path(model_file_path).exists():
        raise FileNotFoundError('Given model file was not found. Please specify an existing saved model file.')
    return load_model(model_file_path, compile=False)


def save_keras_model(model: PrositIntensityPredictor, path_to_model: str) -> None:
    """Saves a given keras model to the path_to_model path. 
    Automatically adds the .keras extension, if the given path does not end in it. This is important, so that
    the model is saved correctly to be loaded again.

    Args:
        model (PrositIntensityPredictor): The model object which should be saved
        path_to_model (str): Path to the model where the model should be saved

    Raises:
        FileExistsError: If the model file already exists -> Raise Error
    """
    if Path(path_to_model).exists():
        raise FileExistsError('This model file already exists. Specify a file, which does not exist yet.')
    if not path_to_model.endswith('.keras'):
        path_to_model += '.keras'
    model.save(path_to_model)


def process_dataset(
        parquet_file_path: str,
        model: PrositIntensityPredictor = None,
        modifications: list = None,
        ion_types: list = None,
        label_column: str = 'intensities_raw',
        batch_size: int = 1024,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        additional_columns: list[str] = None
        ) -> FragmentIonIntensityDataset:
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
        model (PrositIntensityPredictor, optional): Specify a loaded model of the PrositIntensityPredictor class. If None is given,
            the baseline model will be automatically downloaded from GitHub and loaded. Defaults to 'None'
        modifications (list, optional): A list of all modifications which are present in the dataset. Defaults to None.
        ion_types (list, optional): A list of the ion types which are present in the dataset. Defaults to ['y', 'b'].
        label_column (str, optional): The column identifier for where the intensity labels are, if there are any. Defaults to 'intensities_raw'.
        val_ratio (float, optional): A validation split ratio. Defaults to 0.2.
        test_ratio (float, optional): A test split ratio. Defaults to 0.0.
        additional_columns (list[str], optional): List of additional columns to keep in dataset for downstream analysis (will not be returned as tensors).

    Raises:
        ValueError: If the parquet_file_path does not have the .parquet extension
        FileNotFoundError: If the parquet_file_path does not exist

    Returns:
        FragmentIonIntensityDataset: 
            FragmentIonIntensityDataset: The fully processed dataset, which is ready to be used for prediction or transfer/refinement learning
    """
    
    
    modifications = [] if modifications is None else modifications
    ion_types = ['y', 'b'] if ion_types is None else ion_types

    # load the baseline model if None is given
    if model is None:
        model = load_keras_model('baseline')

    val_data_source, test_data_source = None, None
    if not parquet_file_path.endswith('.parquet'):
        # check if dataset is already split
        train_path = parquet_file_path + '_train.parquet'
        val_data_path = parquet_file_path + '_val.parquet'
        test_data_path = parquet_file_path + '_test.parquet'

        # check if the train split exists, if not -> raise ValueError (val and test split are not necessary)
        if not Path(train_path).exists():
            raise ValueError('The specified file is not a parquet file! Please specify a path with the .parquet extension.')
        else:
            parquet_file_path = train_path

        # check if validation split exists
        if Path(val_data_path).exists():
            val_data_source = val_data_path 
        # check if test split exists
        if Path(test_data_path).exists():
            test_data_source = test_data_path

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
        logger.warning(
            """
            There are new tokens in the dataset, which are not supported by the loaded model.
            Either load a different model or transfer learning needs to be done.
            """)
        new_alphabet = deepcopy(model.alphabet)
        new_alphabet.update({k: i for i, k in enumerate(difference, start=len(new_alphabet) + 1)})

    # check for new ion types
    if any([ion_type in ['c', 'z', 'a', 'x'] for ion_type in ion_types]):
        if len(ion_types) == 2:
            logger.warning(
                """
                Number of ions is the same as the loaded model supports, but the ion types are different.
                The model probably needs to be refined to achieve a better performance on these new ion types.
                """)
        if len(ion_types) > 2:
            if 'y' in ion_types and 'b' in ion_types:
                logger.warning(
                    """
                    New Ion types in addition to y and b ions detected.
                    A new output layer is necessary, but it can keep trained weights for y and b ions.
                    """)
            else:
                logger.warning(
                    """
                    Only new ion types are detected. A totally new output layer is necessary.
                    """
                )

    # put additional columns in lower case TODO: remove if CAPS issue is fixed on Oktoberfest side
    if additional_columns is not None:
        additional_columns = [c.lower() for c in additional_columns]

    logger.info('Start processing the dataset...')
    dataset = FragmentIonIntensityDataset(
        data_source=parquet_file_path,
        val_data_source=val_data_source,
        test_data_source=test_data_source,
        data_format='parquet',
        label_column=label_column,
        inference_only=inference_only,
        val_ratio=val_ratio,
        batch_size=batch_size,
        test_ratio=test_ratio,
        alphabet=new_alphabet,
        encoding_scheme='naive-mods',
        model_features=['precursor_charge_onehot', 'collision_energy_aligned_normed', 'method_nbr'],
        ion_types=ion_types,
        dataset_columns_to_keep=additional_columns
    )

    logger.info(f'The available data splits are: {", ".join(list(dataset.hf_dataset.keys()))}')

    return dataset
