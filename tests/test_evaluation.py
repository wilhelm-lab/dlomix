from os.path import join

import keras
import numpy as np
import pytest

from dlomix.data import FragmentIonIntensityDataset
from dlomix.reports import evaluate_intensity, score_intensity_predictions
from dlomix.reports.postprocessing import normalize_intensity_predictions

pytestmark = pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="builds a TensorFlow model, which needs the TensorFlow Keras backend",
)

FEATURES = ["precursor_charge_onehot", "collision_energy_aligned_normed"]


@pytest.fixture(scope="module")
def dataset(download_path_for_assets):
    return FragmentIonIntensityDataset(
        data_format="parquet",
        data_source=join(download_path_for_assets, "file_3.parquet"),
        sequence_column="sequence",
        label_column="intensities",
        model_features=FEATURES,
        with_termini=False,
        val_ratio=0.1,
        batch_size=16,
        shuffle=True,  # the training split's batches are shuffled; the result is not
        dataset_type="tf",
    )


def test_evaluate_intensity_scores_every_spectrum_in_split_order(dataset):
    from dlomix.models.prosit import PrositIntensityPredictor

    model = PrositIntensityPredictor(
        seq_length=30,
        alphabet=dataset.extended_alphabet,
        with_termini=False,
        use_meta_data=True,
        meta_data_keys=FEATURES,
        recurrent_layers_sizes=(8, 16),
        regressor_layer_size=16,
    )

    result = evaluate_intensity(model, dataset, split="train")

    # the same, from the split's rows fed to the model directly, in their order
    split = dataset.hf_dataset["train"]
    inputs = split.with_format("numpy", columns=["sequence", *FEATURES])[:]
    expected = split.select_columns(
        ["_parsed_sequence", "intensities", "precursor_charge_onehot"]
    ).to_pandas()
    expected["intensities_pred"] = list(
        model.predict(inputs, batch_size=dataset.batch_size, verbose=0)
    )
    expected = normalize_intensity_predictions(
        expected,
        sequence_column_name="_parsed_sequence",
        labels_column_name="intensities",
    )

    assert len(result) == len(split)
    predictions = np.stack(result["intensities_pred"])
    np.testing.assert_allclose(
        predictions, np.stack(expected["intensities_pred"]), atol=1e-6
    )
    np.testing.assert_allclose(
        result["spectral_angle"], expected["spectral_angle"], atol=1e-6
    )
    # ions the peptides cannot produce are masked, as in Prosit's post-processing
    assert (predictions == -1).any()


def test_score_intensity_predictions_scores_predictions_made_elsewhere(dataset):
    split = dataset.hf_dataset["val"]
    observed = np.array(split["intensities"], dtype="float32")
    predictions = np.where(observed < 0, 0.0, observed)  # the observed spectra
    predictions[0] = predictions[0, ::-1]  # one wrong prediction, in the first row

    result = score_intensity_predictions(predictions, dataset, split="val")

    assert len(result) == len(split)
    np.testing.assert_allclose(result["spectral_angle"][1:], 1.0, atol=1e-3)
    assert result["spectral_angle"][0] < 0.9

    with pytest.raises(ValueError, match="one prediction per row"):
        score_intensity_predictions(predictions[:-1], dataset, split="val")
