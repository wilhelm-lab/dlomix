"""
Tests for model utility functions in dlomix.models.model_utils.

Tests cover vocabulary expansion, embedding transfer, and model adaptation
for transfer learning scenarios.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from datasets import Dataset

from dlomix.constants import ALPHABET_UNMOD
from dlomix.models import PrositIntensityPredictor, PrositRetentionTimePredictor
from dlomix.models.model_utils import (
    expand_embedding_vocabulary,
    get_alphabet_from_model,
    load_and_adapt_pretrained_model,
)

logger = logging.getLogger(__name__)

DUMMY_INPUT = {
    "sequence": tf.zeros((2, 10), dtype=tf.int32),
    "collision_energy": tf.ones((2, 1)),
    "precursor_charge": tf.ones((2, 1)),
}


@pytest.fixture
def base_alphabet():
    """Simple base alphabet for testing."""
    return {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4}


@pytest.fixture
def expanded_alphabet(base_alphabet):
    """Expanded alphabet with additional tokens."""
    expanded = dict(base_alphabet)
    expanded.update(
        {
            "M[UNIMOD:1]": 5,
            "S[UNIMOD:2]": 6,
            "T[UNIMOD:3]": 7,
        }
    )
    return expanded


@pytest.fixture
def intensity_model(base_alphabet):
    """Create a PrositIntensityPredictor for testing."""
    model = PrositIntensityPredictor(
        embedding_output_dim=8,
        seq_length=10,
        alphabet=base_alphabet,
        dropout_rate=0.1,
        # no meta_data_keys: these tests only exercise the embedding, and the
        # extra metadata inputs are ignored
    )
    model(DUMMY_INPUT)
    return model


def assert_shared_rows_preserved(old_model, new_model, old_alphabet, new_alphabet):
    old = old_model.embedding.get_weights()[0]
    new = new_model.embedding.get_weights()[0]
    for token, idx in old_alphabet.items():
        np.testing.assert_array_equal(new[new_alphabet[token]], old[idx], err_msg=token)


class TestExpandEmbeddingVocabulary:
    """Tests for expand_embedding_vocabulary function."""

    def test_expansion(self, intensity_model, base_alphabet, expanded_alphabet):
        """The embedding grows to the new alphabet, keeps the rows of the shared
        tokens, updates alphabet and embeddings_count, and the model still runs."""
        adapted = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=None,  # extracted from the model
            initialization_strategy="random",
            random_seed=42,
        )

        assert adapted.embedding.input_dim == len(expanded_alphabet)
        assert adapted.embedding.output_dim == intensity_model.embedding.output_dim
        assert adapted.alphabet == expanded_alphabet
        assert adapted.embeddings_count == len(expanded_alphabet)
        assert_shared_rows_preserved(
            intensity_model, adapted, base_alphabet, expanded_alphabet
        )
        assert adapted(DUMMY_INPUT) is not None

    def test_mean_initialization(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that new tokens are initialized with mean when strategy='mean'."""
        old_weights = intensity_model.embedding.get_weights()[0]
        mean_embedding = np.mean(old_weights, axis=0)

        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
            initialization_strategy="mean",
        )

        new_weights = adapted_model.embedding.get_weights()[0]

        # Check that new tokens have mean embeddings
        new_tokens = set(expanded_alphabet.keys()) - set(base_alphabet.keys())
        for token in new_tokens:
            new_idx = expanded_alphabet[token]
            embedding = new_weights[new_idx]
            assert np.allclose(
                embedding, mean_embedding, atol=1e-5
            ), f"New token '{token}' should have mean embedding"

    def test_random_initialization_with_seed(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that random initialization is reproducible with seed."""
        # Expand twice with same seed
        model1 = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
            initialization_strategy="random",
            random_seed=42,
        )

        # Need a fresh model for second expansion
        model2 = PrositRetentionTimePredictor(
            embedding_output_dim=8,
            seq_length=10,
            alphabet=base_alphabet,
            dropout_rate=0.1,
        )
        _ = model2(tf.zeros((2, 10), dtype=tf.int32))

        model2 = expand_embedding_vocabulary(
            model=model2,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
            initialization_strategy="random",
            random_seed=42,
        )

        weights1 = model1.embedding.get_weights()[0]
        weights2 = model2.embedding.get_weights()[0]

        # Check that new token embeddings are identical (reproducible)
        new_tokens = set(expanded_alphabet.keys()) - set(base_alphabet.keys())
        for token in new_tokens:
            new_idx = expanded_alphabet[token]
            assert np.allclose(
                weights1[new_idx], weights2[new_idx]
            ), f"Random initialization with seed should be reproducible for '{token}'"

    def test_invalid_embedding_layer_name(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that invalid embedding layer name raises error."""
        with pytest.raises(AttributeError, match="Embedding layer.*not found"):
            expand_embedding_vocabulary(
                model=intensity_model,
                new_alphabet=expanded_alphabet,
                old_alphabet=base_alphabet,
                embedding_layer_name="nonexistent_layer",
            )

    def test_invalid_initialization_strategy(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that invalid initialization strategy raises error."""
        with pytest.raises(ValueError, match="Unknown initialization_strategy"):
            expand_embedding_vocabulary(
                model=intensity_model,
                new_alphabet=expanded_alphabet,
                old_alphabet=base_alphabet,
                initialization_strategy="invalid_strategy",
            )

    def test_retention_time_model_with_real_alphabet(self):
        """Expansion of a PrositRetentionTimePredictor built on ALPHABET_UNMOD."""
        model = PrositRetentionTimePredictor(
            embedding_output_dim=16,
            seq_length=30,
            alphabet=ALPHABET_UNMOD,
        )
        _ = model(tf.zeros((2, 30), dtype=tf.int32))
        next_idx = len(ALPHABET_UNMOD)
        expanded_alphabet = {
            **ALPHABET_UNMOD,
            "M[UNIMOD:1]": next_idx,
            "S[UNIMOD:2]": next_idx + 1,
            "C[UNIMOD:3]": next_idx + 2,
        }

        adapted_model = expand_embedding_vocabulary(
            model=model,
            new_alphabet=expanded_alphabet,
            old_alphabet=ALPHABET_UNMOD,
            initialization_strategy="mean",
            embedding_layer_name="embedding",
        )

        assert adapted_model.embedding.input_dim == len(expanded_alphabet)


class TestGetAlphabetFromModel:
    """Tests for get_alphabet_from_model function."""

    def test_extract_from_model(self, intensity_model, base_alphabet):
        assert get_alphabet_from_model(intensity_model) == base_alphabet

    def test_model_without_alphabet(self):
        """Test that models without alphabet return None."""
        # Create a generic Keras model without alphabet attribute
        simple_keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(10, 8),
                tf.keras.layers.Dense(1),
            ]
        )

        alphabet = get_alphabet_from_model(simple_keras_model)
        assert alphabet is None


class TestLoadAndAdaptPretrainedModel:
    """Tests for load_and_adapt_pretrained_model function."""

    @pytest.mark.parametrize("explicit_old_alphabet", [False, True])
    def test_save_load_adapt(
        self, intensity_model, base_alphabet, expanded_alphabet, explicit_old_alphabet
    ):
        """Save, then load and adapt: the shared rows survive and the model runs,
        whether the old alphabet is given or read from the saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.keras"
            intensity_model.save(model_path)

            adapted = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                old_alphabet=base_alphabet if explicit_old_alphabet else None,
                initialization_strategy="mean",
                random_seed=42,
            )

        assert adapted.embedding.input_dim == len(expanded_alphabet)
        assert adapted.alphabet == expanded_alphabet
        assert_shared_rows_preserved(
            intensity_model, adapted, base_alphabet, expanded_alphabet
        )
        assert adapted(DUMMY_INPUT) is not None

    def test_invalid_model_path(self, expanded_alphabet):
        """Test that invalid model path raises error."""
        with pytest.raises(ValueError, match="Failed to load model"):
            load_and_adapt_pretrained_model(
                model_path="/nonexistent/path/model.keras",
                new_alphabet=expanded_alphabet,
            )


def test_best_fit_copies_the_chosen_rows_and_never_picks_padding_or_unknown():
    """Best-fit runs on its defaults (n_examples_for_eval, eval_metric), copies the
    chosen pretrained row into each new token, and never picks the padding or
    unknown token."""
    rng = np.random.default_rng(0)
    data = Dataset.from_dict(
        {
            "sequence": ["AM[UNIMOD:1]CDEF", "ES[UNIMOD:2]CDEF", "CT[UNIMOD:3]CDEF"],
            "collision_energy": [0.25, 0.3, 0.35],
            "precursor_charge": [1.0, 2.0, 3.0],
            "label": rng.random((3, 54)).tolist(),  # (10 - 1) positions x 2 x 3
        }
    )
    old_alphabet = {"-": 0, "X": 1, "A": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    new_alphabet = {
        **old_alphabet,
        "M[UNIMOD:1]": 7,
        "S[UNIMOD:2]": 8,
        "T[UNIMOD:3]": 9,
    }
    model = PrositIntensityPredictor(
        embedding_output_dim=8,
        seq_length=10,
        alphabet=old_alphabet,
        use_meta_data=True,
        meta_data_keys=["collision_energy", "precursor_charge"],
    )
    model(DUMMY_INPUT)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "intensity_model.keras"
        model.save(model_path)
        adapted, fit_info = load_and_adapt_pretrained_model(
            model_path=str(model_path),
            new_alphabet=new_alphabet,
            initialization_strategy="best-fit",
            best_fit_kwargs={
                "new_hf_data": data,
                "sequence_column": "sequence",
                "label_column": "label",
                "return_fit_info": True,
                "dataset_kwargs": {
                    "encoding_scheme": "naive-mods",
                    "max_seq_len": 10,
                    "with_termini": False,
                    "num_proc": None,
                    "model_features": ["collision_energy", "precursor_charge"],
                },
            },
        )

    old_weights = model.embedding.get_weights()[0]
    new_weights = adapted.embedding.get_weights()[0]
    assert set(fit_info) == {"M[UNIMOD:1]", "S[UNIMOD:2]", "T[UNIMOD:3]"}
    for token, info in fit_info.items():
        assert info, "every new token has examples, so each must get a fit"
        assert info["old_token"] not in ("-", "X")
        np.testing.assert_array_equal(
            new_weights[new_alphabet[token]], old_weights[info["old_token_idx"]]
        )
