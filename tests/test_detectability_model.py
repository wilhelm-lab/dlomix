"""Weight-loading contract for the Detectability (Pfly) model.

Keras 3's default ``Model.build(input_shape)`` only flips the ``built`` flag; it
does not create the weights of sub-layers built in ``__init__``. That made
``build(...)`` followed by ``load_weights(...)`` a silent no-op: no error, zero
variables, and the first forward pass then produced fresh random weights. The
fine-tuning walkthrough looked like it resumed from a checkpoint while actually
training from scratch, which is exactly the failure these tests pin down.

The shipped pretrained weights are also covered, so the committed `.weights.h5`
files cannot silently stop loading.
"""

import logging
import pathlib

import keras
import numpy as np
import pytest

from dlomix.constants import CLASSES_LABELS
from dlomix.models.detectability import DetectabilityModel

logger = logging.getLogger(__name__)

NUM_UNITS = 8  # small: these tests are about wiring, not capacity
SEQ_LEN = 40
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PRETRAINED = REPO_ROOT / "pretrained_models"
SHIPPED_WEIGHTS = [
    PRETRAINED / name / f"{name}.weights.h5"
    for name in (
        "original_detectability_base_model",
        "original_detectability_fine_tuned_model_FINAL",
    )
]


def _model(num_units=NUM_UNITS):
    return DetectabilityModel(num_units=num_units, num_classes=len(CLASSES_LABELS))


@pytest.fixture
def sequences():
    return np.random.default_rng(0).integers(1, 20, (4, SEQ_LEN))


def test_build_creates_the_sublayer_weights():
    """``build(input_shape)`` must materialise variables, not just set a flag."""
    model = _model()
    model.build(input_shape=(None, SEQ_LEN))

    assert model.built
    assert len(model.weights) > 0, "build() left the model with no variables"


def test_build_matches_what_a_forward_pass_creates(sequences):
    """Explicit build and the call-driven build must agree on the structure."""
    built = _model()
    built.build(input_shape=(None, SEQ_LEN))

    called = _model()
    called(sequences)

    assert [tuple(v.shape) for v in built.weights] == [
        tuple(v.shape) for v in called.weights
    ]


def test_build_then_load_weights_actually_loads(tmp_path, sequences):
    """The regression: this combination used to load nothing, silently."""
    source = _model()
    source(sequences)  # real build, so the weights are initialised
    path = tmp_path / "detectability.weights.h5"
    source.save_weights(path)

    target = _model()
    target.build(input_shape=(None, SEQ_LEN))
    target.load_weights(path)

    np.testing.assert_allclose(
        source(sequences).numpy(), target(sequences).numpy(), atol=1e-6
    )


def test_load_weights_into_an_unbuilt_model_is_rejected(tmp_path, sequences):
    """An unbuilt model must fail loudly rather than quietly load nothing."""
    source = _model()
    source(sequences)
    path = tmp_path / "detectability.weights.h5"
    source.save_weights(path)

    with pytest.raises(ValueError, match="has not yet been built"):
        _model().load_weights(path)


def test_keras_round_trip_preserves_predictions(tmp_path, sequences):
    source = _model()
    before = source(sequences).numpy()
    path = tmp_path / "detectability.keras"
    source.save(path)

    reloaded = keras.saving.load_model(path)

    assert isinstance(reloaded, DetectabilityModel)
    np.testing.assert_allclose(before, reloaded(sequences).numpy(), atol=1e-6)


@pytest.mark.parametrize("weights_path", SHIPPED_WEIGHTS, ids=lambda p: p.parent.name)
def test_shipped_pretrained_weights_load(weights_path, sequences):
    """The committed `.weights.h5` files must stay loadable and deterministic.

    They are conversions of the original TensorFlow checkpoints, which Keras 3
    cannot read at all; see scripts/convert_detectability_checkpoints.py.
    """
    if not weights_path.is_file():
        pytest.skip(f"{weights_path.name} not present")

    model = DetectabilityModel(num_units=64, num_classes=len(CLASSES_LABELS))
    model.build(input_shape=(None, SEQ_LEN))
    model.load_weights(weights_path)

    predictions = model(sequences).numpy()
    assert predictions.shape == (len(sequences), len(CLASSES_LABELS))
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-5)

    # loading twice must give the same answer -- guards against partial restores
    again = DetectabilityModel(num_units=64, num_classes=len(CLASSES_LABELS))
    again.build(input_shape=(None, SEQ_LEN))
    again.load_weights(weights_path)
    np.testing.assert_allclose(predictions, again(sequences).numpy(), atol=0)
