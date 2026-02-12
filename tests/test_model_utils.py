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
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictor, PrositRetentionTimePredictor
from dlomix.models.model_utils import (
    expand_embedding_vocabulary,
    get_alphabet_from_model,
    load_and_adapt_pretrained_model,
)

logger = logging.getLogger(__name__)


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
        meta_data_keys=["collision_energy", "precursor_charge"],
    )
    # Build the model
    dummy_input = {
        "sequence": tf.zeros((2, 10), dtype=tf.int32),
        "collision_energy": tf.ones((2, 1)),
        "precursor_charge": tf.ones((2, 1)),
    }
    _ = model(dummy_input)
    return model


@pytest.fixture
def hf_dataset_with_mods():
    """Fixture to provide a Hugging Face dataset with modified sequences."""

    data = {
        "sequence": ["AM[UNIMOD:1]CDEF", "ES[UNIMOD:2]CDEF", "CT[UNIMOD:3]CDEF"],
        "collision_energy": [
            np.zeros((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
            np.full((2, 1), 2, dtype=np.float32),
        ],
        "precursor_charge": [
            np.zeros((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
            np.full((2, 1), 3, dtype=np.float32),
        ],
        "label": [
            [0.1, 0.2, 0.3] * 9 * 2,  # (sequence length - 1) * 2 ions
            [0.4, 0.5, 0.6] * 9 * 2,
            [0.7, 0.8, 0.9] * 9 * 2,
        ],
    }

    # Create dataset
    ds = Dataset.from_dict(data)
    return ds


class TestExpandEmbeddingVocabulary:
    """Tests for expand_embedding_vocabulary function."""

    def test_basic_expansion(self, intensity_model, base_alphabet, expanded_alphabet):
        """Test basic vocabulary expansion with default random initialization."""
        old_vocab_size = intensity_model.embedding.input_dim
        old_embedding_dim = intensity_model.embedding.output_dim

        # Expand vocabulary
        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
            initialization_strategy="random",
            random_seed=42,
        )

        # Check new vocabulary size
        expected_new_vocab_size = len(expanded_alphabet)
        assert adapted_model.embedding.input_dim == expected_new_vocab_size
        assert adapted_model.embedding.output_dim == old_embedding_dim

        # Verify size increased
        assert adapted_model.embedding.input_dim > old_vocab_size

    def test_embedding_transfer_preservation(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that common token embeddings are preserved after expansion."""
        # Get original embeddings
        old_weights = intensity_model.embedding.get_weights()[0]

        # Expand vocabulary
        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
            initialization_strategy="random",
            random_seed=42,
        )

        new_weights = adapted_model.embedding.get_weights()[0]

        # Check that common amino acid embeddings were preserved
        for token, old_idx in base_alphabet.items():
            # In embedding matrix: +1 for padding token at index 0
            old_embedding_idx = old_idx + 1
            new_idx = expanded_alphabet[token]
            new_embedding_idx = new_idx + 1

            # Only check if the index is valid in the old weights
            if old_embedding_idx < old_weights.shape[0]:
                # Embeddings should be identical
                assert np.allclose(
                    old_weights[old_embedding_idx], new_weights[new_embedding_idx]
                ), f"Embedding for token '{token}' was not preserved"

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

    def test_alphabet_attribute_updated(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that model's alphabet attribute is updated after expansion."""
        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
        )

        # Check alphabet was updated
        assert hasattr(adapted_model, "alphabet")
        assert adapted_model.alphabet == expanded_alphabet

    def test_embeddings_count_updated(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that model's embeddings_count attribute is updated."""
        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
        )

        expected_count = len(expanded_alphabet)
        assert hasattr(adapted_model, "embeddings_count")
        assert adapted_model.embeddings_count == expected_count

    def test_auto_extract_alphabet_from_model(self, intensity_model, expanded_alphabet):
        """Test that old_alphabet can be auto-extracted from model."""
        # Don't provide old_alphabet - should be extracted automatically
        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=None,  # Should auto-extract
        )

        assert adapted_model is not None
        assert adapted_model.embedding.input_dim == len(expanded_alphabet)

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

    def test_intensity_model_expansion(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test vocabulary expansion on PrositIntensityPredictor."""
        adapted_model = expand_embedding_vocabulary(
            model=intensity_model,
            new_alphabet=expanded_alphabet,
            old_alphabet=base_alphabet,
            initialization_strategy="mean",
        )

        # Verify model still works with expanded vocabulary
        assert adapted_model is not None
        assert adapted_model.embedding.input_dim == len(expanded_alphabet)

        # Test forward pass still works
        dummy_input = {
            "sequence": tf.zeros((2, 10), dtype=tf.int32),
            "collision_energy": tf.ones((2, 1)),
            "precursor_charge": tf.ones((2, 1)),
        }
        output = adapted_model(dummy_input)
        assert output is not None


class TestGetAlphabetFromModel:
    """Tests for get_alphabet_from_model function."""

    def test_extract_from_model_attribute(self, intensity_model):
        """Test extracting alphabet from model.alphabet attribute."""
        alphabet = get_alphabet_from_model(intensity_model)
        assert alphabet is not None
        assert isinstance(alphabet, dict)
        assert len(alphabet) > 0

    def test_extract_from_intensity_model(self, intensity_model):
        """Test extracting alphabet from PrositIntensityPredictor."""
        alphabet = get_alphabet_from_model(intensity_model)
        assert alphabet is not None
        assert isinstance(alphabet, dict)

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

    def test_full_workflow(self, intensity_model, base_alphabet, expanded_alphabet):
        """Test complete workflow: save, load, and adapt model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the model
            model_path = Path(tmpdir) / "test_model.keras"
            intensity_model.save(model_path)

            # Load and adapt
            adapted_model = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                old_alphabet=None,
                initialization_strategy="mean",
                random_seed=42,
            )

            # Verify adaptation
            assert adapted_model is not None
            assert adapted_model.embedding.input_dim == len(expanded_alphabet)

    def test_with_explicit_old_alphabet(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test workflow with explicitly provided old_alphabet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.keras"
            intensity_model.save(model_path)

            adapted_model = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                old_alphabet=base_alphabet,
                initialization_strategy="mean",
            )

            assert adapted_model.embedding.input_dim == len(expanded_alphabet)

    def test_embeddings_preserved_through_save_load(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test that embeddings are preserved through save/load/adapt cycle."""
        # Get original embeddings for a common token
        old_weights = intensity_model.embedding.get_weights()[0]
        token = "A"
        original_embedding = old_weights[base_alphabet[token]].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.keras"
            intensity_model.save(model_path)

            adapted_model = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                initialization_strategy="random",
                random_seed=42,
            )

            # Check that common token embedding was preserved
            new_weights = adapted_model.embedding.get_weights()[0]
            new_embedding = new_weights[expanded_alphabet[token]]

            assert np.allclose(
                original_embedding, new_embedding
            ), "Embedding should be preserved through save/load/adapt"

    def test_invalid_model_path(self, expanded_alphabet):
        """Test that invalid model path raises error."""
        with pytest.raises(ValueError, match="Failed to load model"):
            load_and_adapt_pretrained_model(
                model_path="/nonexistent/path/model.keras",
                new_alphabet=expanded_alphabet,
            )

    def test_intensity_model_full_workflow(
        self, intensity_model, base_alphabet, expanded_alphabet
    ):
        """Test full workflow with PrositIntensityPredictor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "intensity_model.keras"
            intensity_model.save(model_path)

            adapted_model = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                initialization_strategy="mean",
            )

            # Verify model still works
            dummy_input = {
                "sequence": tf.zeros((2, 10), dtype=tf.int32),
                "collision_energy": tf.ones((2, 1)),
                "precursor_charge": tf.ones((2, 1)),
            }
            output = adapted_model(dummy_input)
            assert output is not None

    def test_intensity_model_best_fit_initialization_workflow(
        self, intensity_model, base_alphabet, expanded_alphabet, hf_dataset_with_mods
    ):
        """Test full workflow with PrositIntensityPredictor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "intensity_model.keras"
            intensity_model.save(model_path)

            adapted_model, fit_info = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                old_alphabet=base_alphabet,
                initialization_strategy="best-fit",
                best_fit_kwargs={
                    "new_hf_data": hf_dataset_with_mods,  # Use the dataset fixture with modified sequences
                    "sequence_column": "sequence",
                    "label_column": "label",
                    "n_examples_for_eval": 1,
                    "eval_metric": masked_spectral_distance,
                    "return_fit_info": True,
                    "dataset_kwargs": {
                        "encoding_scheme": "naive-mods",
                        "max_seq_len": 10,
                        "with_termini": False,
                        "model_features": ["collision_energy", "precursor_charge"],
                    },
                },
            )

            # Verify model still works
            dummy_input = {
                "sequence": tf.zeros((2, 10), dtype=tf.int32),
                "collision_energy": tf.ones((2, 1)),
                "precursor_charge": tf.ones((2, 1)),
            }
            output = adapted_model(dummy_input)
            assert output is not None

            logger.info(fit_info)

            assert fit_info is not None

            old_weights = intensity_model.get_layer("embedding").get_weights()[0]
            new_weights = adapted_model.get_layer("embedding").get_weights()[0]

            for new_token, fit_info in fit_info.items():
                new_idx = expanded_alphabet[new_token]
                assert np.allclose(
                    old_weights[fit_info["old_token_idx"]], new_weights[new_idx]
                ), f"Best-fit initialization failed for token '{new_token}'"


class TestIntegrationWithRealAlphabet:
    """Integration tests using real ALPHABET_UNMOD."""

    def test_expand_with_real_alphabet(self):
        """Test expansion with actual ALPHABET_UNMOD."""
        # Create model with real alphabet
        model = PrositRetentionTimePredictor(
            embedding_output_dim=16,
            seq_length=30,
            alphabet=ALPHABET_UNMOD,
        )
        _ = model(tf.zeros((2, 30), dtype=tf.int32))

        # Create expanded alphabet with PTMs
        expanded_alphabet = dict(ALPHABET_UNMOD)
        next_idx = len(expanded_alphabet)
        expanded_alphabet.update(
            {
                "M[UNIMOD:1]": next_idx,
                "S[UNIMOD:2]": next_idx + 1,
                "C[UNIMOD:3]": next_idx + 2,
            }
        )

        # Expand vocabulary
        adapted_model = expand_embedding_vocabulary(
            model=model,
            new_alphabet=expanded_alphabet,
            old_alphabet=ALPHABET_UNMOD,
            initialization_strategy="mean",
            embedding_layer_name="embedding",
        )

        assert adapted_model.embedding.input_dim == len(expanded_alphabet)

    def test_real_model_save_load_adapt(self):
        """Test complete workflow with real alphabet and model saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            model = PrositIntensityPredictor(
                embedding_output_dim=16,
                seq_length=30,
                alphabet=ALPHABET_UNMOD,
            )
            dummy_input = {
                "sequence": tf.zeros((2, 30), dtype=tf.int32),
                "collision_energy": tf.ones((2, 1)),
                "precursor_charge": tf.ones((2, 1)),
            }
            _ = model(dummy_input)

            model_path = Path(tmpdir) / "real_model.keras"
            model.save(model_path)

            # Create expanded alphabet
            expanded_alphabet = dict(ALPHABET_UNMOD)
            next_idx = len(expanded_alphabet)
            expanded_alphabet["M[UNIMOD:1]"] = next_idx
            expanded_alphabet["S[UNIMOD:2]"] = next_idx + 1

            # Load and adapt
            adapted_model = load_and_adapt_pretrained_model(
                model_path=str(model_path),
                new_alphabet=expanded_alphabet,
                initialization_strategy="random",
                random_seed=42,
            )

            # Verify functionality
            output = adapted_model(dummy_input)
            assert output is not None
            assert adapted_model.alphabet == expanded_alphabet
