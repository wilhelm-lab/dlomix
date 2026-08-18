import logging

import pytest
import tensorflow as tf

from dlomix.constants import ALPHABET_UNMOD, PTMS_ALPHABET
from dlomix.models.chargestate import ChargeStatePredictor
from dlomix.models.deepLC import DeepLCRetentionTimePredictor
from dlomix.models.prosit import PrositIntensityPredictor, PrositRetentionTimePredictor

logger = logging.getLogger(__name__)


def _zeros_for(shapes, batch_size=2):
    """Materialise a ``{key: shape}`` spec into zero tensors for a forward pass.

    Keras 3 builds a subclassed model on its first call, not from a shape, so
    the tests exercise the model the way callers do.
    """
    return {key: tf.zeros((batch_size, *shape[1:])) for key, shape in shapes.items()}


def test_prosit_retention_time_model():
    model = PrositRetentionTimePredictor()
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model():
    model = PrositIntensityPredictor(
        input_keys={
            "SEQUENCE_KEY": "sequence",
        },
        meta_data_keys=["collision_energy", "precursor_charge"],
    )

    seq_len = model.seq_length
    model(
        _zeros_for(
            {
                "sequence": (None, seq_len),
                "collision_energy": (None, 1),
                "precursor_charge": (None, 6),
            }
        )
    )
    model.summary(print_fn=logger.info)
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model_ptm_on_input():
    model = PrositIntensityPredictor(
        input_keys={
            "SEQUENCE_KEY": "sequence",
        },
        meta_data_keys=["collision_energy", "precursor_charge", "fragmentation_type"],
        use_prosit_ptm_features=True,
    )

    seq_len = model.seq_length
    model(
        _zeros_for(
            {
                "sequence": (None, seq_len),
                "collision_energy": (None, 1),
                "precursor_charge": (None, 6),
                "fragmentation_type": (None, 1),
                PrositIntensityPredictor.PTM_INPUT_KEYS[0]: (None, seq_len, 6),
                PrositIntensityPredictor.PTM_INPUT_KEYS[1]: (None, seq_len, 6),
                PrositIntensityPredictor.PTM_INPUT_KEYS[2]: (None, seq_len, 1),
            }
        )
    )
    model.summary(print_fn=logger.info)
    logger.info(model.input_keys)
    logger.info(model)
    assert model is not None


def test_prosit_intensity_model_ptm_on_missing():
    model = PrositIntensityPredictor(use_prosit_ptm_features=True)
    seq_len = model.seq_length
    with pytest.raises(ValueError, match="PTM"):
        model(
            _zeros_for(
                {
                    "sequence": (None, seq_len),
                    "collision_energy": (None, 1),
                    "precursor_charge": (None, 6),
                    "fragmentation_type": (None, 1),
                    # no PTM features
                }
            )
        )


def test_prosit_intensity_model_encoding_metadata_missing():
    model = PrositIntensityPredictor(
        meta_data_keys=["meta_data_1", "meta_data_2"], use_meta_data=True
    )
    seq_len = model.seq_length

    with pytest.raises(ValueError, match="metadata"):
        model(
            _zeros_for(
                {
                    "sequence": (None, seq_len),
                    # no meta-data while expected
                }
            )
        )


def test_prosit_intensity_model_no_metadata():
    model = PrositIntensityPredictor(
        input_keys={
            "SEQUENCE_KEY": "sequence",
        },
        meta_data_keys=None,
    )

    seq_len = model.seq_length

    assert model.meta_data_keys == []

    model(_zeros_for({"sequence": (None, seq_len)}))

    assert model is not None
    assert model.meta_encoder is None


def _make_deeplc_inputs(
    model,
    batch_size=2,
    sequence_rank=2,
    include_global_features=False,
    invalid_sequence_depth=None,
):
    seq_length = model.seq_length
    alphabet_size = len(model.alphabet)

    if sequence_rank == 2:
        sequence = tf.zeros((batch_size, seq_length), dtype=tf.int32)
    elif sequence_rank == 3:
        depth = (
            alphabet_size if invalid_sequence_depth is None else invalid_sequence_depth
        )
        sequence = tf.one_hot(
            tf.zeros((batch_size, seq_length), dtype=tf.int32), depth=depth
        )
    elif sequence_rank == 1:
        sequence = tf.zeros((batch_size,), dtype=tf.int32)
    else:
        sequence = tf.zeros((batch_size, seq_length, 1), dtype=tf.float32)

    inputs = {
        model.sequence_input_key: sequence,
        model.counts_input_key: tf.zeros((batch_size, seq_length, 6), dtype=tf.float32),
        model.di_counts_input_key: tf.zeros(
            (batch_size, seq_length // 2, 6), dtype=tf.float32
        ),
    }

    if include_global_features:
        inputs[model.global_features_input_key] = tf.zeros(
            (batch_size, 6), dtype=tf.float32
        )

    return inputs


def test_deeplc_retention_time_model():
    model = DeepLCRetentionTimePredictor()
    inputs = _make_deeplc_inputs(model)

    outputs = model(inputs)

    logger.info(model)
    logger.info(outputs)
    assert outputs.shape == (2, 1)
    assert len(model.trainable_variables) > 0


# Test that the model can handle both rank 2 (token ids) and rank 3 (one-hot encoded) sequence inputs
@pytest.mark.parametrize("sequence_rank", [2, 3])
def test_deeplc_retention_time_model_sequence_input_modes(sequence_rank):
    model = DeepLCRetentionTimePredictor()
    inputs = _make_deeplc_inputs(model, sequence_rank=sequence_rank)

    outputs = model(inputs)

    assert outputs.shape == (2, 1)


def test_deeplc_retention_time_model_with_global_features():
    model = DeepLCRetentionTimePredictor(use_global_features=True)
    inputs = _make_deeplc_inputs(model, include_global_features=True)

    outputs = model(inputs)

    assert outputs.shape == (2, 1)


def test_deeplc_retention_time_model_missing_required_input_key():
    model = DeepLCRetentionTimePredictor()
    inputs = _make_deeplc_inputs(model)
    inputs.pop(model.di_counts_input_key)

    with pytest.raises(KeyError):
        model(inputs)


def test_deeplc_retention_time_model_invalid_sequence_rank():
    model = DeepLCRetentionTimePredictor()
    inputs = _make_deeplc_inputs(model, sequence_rank=1)

    with pytest.raises(ValueError, match="rank 2 .* rank 3"):
        model(inputs)


def test_deeplc_retention_time_model_invalid_sequence_depth():
    model = DeepLCRetentionTimePredictor()
    inputs = _make_deeplc_inputs(model, sequence_rank=3, invalid_sequence_depth=3)

    with pytest.raises(ValueError, match="alphabet size"):
        model(inputs)


def test_deeplc_retention_time_model_get_config_round_trip():
    model = DeepLCRetentionTimePredictor(use_global_features=True)
    clone = DeepLCRetentionTimePredictor.from_config(model.get_config())

    inputs = _make_deeplc_inputs(clone, include_global_features=True)

    outputs = clone(inputs)

    assert clone.use_global_features is True
    assert outputs.shape == (2, 1)


def basic_model_existence_test(model):
    logger.info(model)
    assert model is not None

    # Keras 3 builds a subclassed model on its first call, not from a shape.
    model(tf.zeros((2, 30)))
    assert len(model.trainable_variables) > 0


def test_dominant_chargestate_model():
    model = ChargeStatePredictor(model_flavour="dominant")
    basic_model_existence_test(model)


def test_observed_chargestate_model():
    model = ChargeStatePredictor(model_flavour="observed")
    basic_model_existence_test(model)


def test_chargestate_distribution_model():
    model = ChargeStatePredictor(model_flavour="relative")
    basic_model_existence_test(model)


# ---------------------------------------------------------------------------
# Alphabet sizing
# ---------------------------------------------------------------------------


def test_raw_alphabet_with_out_of_range_indices_is_rejected():
    """A raw alphabet whose top index exceeds its length cannot be embedded.

    PTMS_ALPHABET maps 56 tokens onto indices up to 56, so it needs 57 embedding
    rows. Without the guard the extra token would index out of range and gather
    zeros silently on GPU. The dataset's extended_alphabet is the supported input.
    """
    assert max(PTMS_ALPHABET.values()) >= len(PTMS_ALPHABET)

    with pytest.raises(ValueError, match="extended_alphabet"):
        PrositIntensityPredictor(seq_length=30, alphabet=PTMS_ALPHABET)


def test_alphabet_carrying_padding_and_unknown_is_accepted():
    """The vocabulary a dataset produces sizes the embedding exactly."""
    extended = {**ALPHABET_UNMOD, "": 0, "X": max(ALPHABET_UNMOD.values()) + 1}

    model = PrositIntensityPredictor(seq_length=30, alphabet=extended)

    assert model.embeddings_count == len(extended)
    assert model.embeddings_count > max(extended.values())
