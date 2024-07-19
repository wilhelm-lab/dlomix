import warnings
from copy import deepcopy
import tensorflow as tf
import pyarrow.parquet as pq
from pathlib import Path
from dlomix.losses import masked_spectral_distance
from dlomix.data.fragment_ion_intensity import FragmentIonIntensityDataset
from dlomix.models.prosit import PrositIntensityPredictor


def process_dataset(
        parquet_file_path: str,
        model_file_path: str = 'unmod_ext',
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
        model_file_path (str, optional): Either a predefined baseline model identifier. Options are ['unmod_ext', 'naive', 'ptm']
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

    # give the option to load different baseline models
    model_base_path = '/cmnfs/proj/bmpc_dlomix/models/baseline_models/'
    match model_file_path:
        case 'unmod_ext':
            model_file_path = model_base_path + 'noptm_baseline_full_bs1024_unmod_extended/7ef3360f-2349-46c0-a905-01187d4899e2.keras'
        case 'naive':
            model_file_path = model_base_path + 'noptm_baseline_full_bs1024_naivemods/d961f940-d142-4102-9775-c1f8b4373c91.keras'
        case 'ptm':
            model_file_path = model_base_path + 'noptm_baseline_full_bs1024/4bc7bc69-bbf4-4366-90fc-474b1946c588.keras'
        case _:
            if not model_file_path.endswith('.keras'):
                raise ValueError('Given model needs to be saved in the .keras format in order to be loaded correctly.')
            if not Path(model_file_path).exists():
                raise FileNotFoundError('The model was not found. Please specify a valid model.')
    
    # load the model
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
