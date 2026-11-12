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
from dlomix.data import FragmentIonIntensityDataset
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

    def test_mean_initialization_averages_residue_rows_only(self):
        """Padding and the unknown token carry no residue information."""
        old_alphabet = {"-": 0, "X": 1, "A": 2, "C": 3, "D": 4}
        new_alphabet = {**old_alphabet, "M[UNIMOD:35]": 5}
        model = PrositIntensityPredictor(
            embedding_output_dim=8, seq_length=10, alphabet=old_alphabet
        )
        model(DUMMY_INPUT)
        old_weights = model.embedding.get_weights()[0]

        adapted = expand_embedding_vocabulary(
            model=model,
            new_alphabet=new_alphabet,
            old_alphabet=old_alphabet,
            initialization_strategy="mean",
        )

        np.testing.assert_allclose(
            adapted.embedding.get_weights()[0][5], old_weights[2:].mean(axis=0)
        )
        assert_shared_rows_preserved(model, adapted, old_alphabet, new_alphabet)

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


# --------------------------------- best-fit -----------------------------------

BEST_FIT_OLD_ALPHABET = {"-": 0, "X": 1, "A": 2, "C": 3, "D": 4, "E": 5, "F": 6}
BEST_FIT_DATASET_KWARGS = {
    "encoding_scheme": "naive-mods",
    "max_seq_len": 10,
    "with_termini": False,
    "num_proc": None,
    "model_features": ["collision_energy", "precursor_charge"],
}


def _spectra(sequences):
    rng = np.random.default_rng(0)
    return Dataset.from_dict(
        {
            "sequence": sequences,
            "collision_energy": [0.3] * len(sequences),
            "precursor_charge": [2.0] * len(sequences),
            # (10 - 1) positions x 2 x 3
            "label": rng.random((len(sequences), 54)).tolist(),
        }
    )


@pytest.fixture(scope="module")
def best_fit_model_path(tmp_path_factory):
    model = PrositIntensityPredictor(
        embedding_output_dim=8,
        seq_length=10,
        alphabet=BEST_FIT_OLD_ALPHABET,
        use_meta_data=True,
        meta_data_keys=["collision_energy", "precursor_charge"],
    )
    model(DUMMY_INPUT)
    path = tmp_path_factory.mktemp("best_fit") / "intensity_model.keras"
    model.save(path)
    return str(path)


def _best_fit(model_path, data, new_tokens, random_seed=None, **options):
    """Adapt the model to ``new_tokens`` by best-fit: (old model, new model, info)."""
    new_alphabet = dict(BEST_FIT_OLD_ALPHABET)
    for token in new_tokens:
        new_alphabet[token] = len(new_alphabet)
    adapted, fit_info = load_and_adapt_pretrained_model(
        model_path=model_path,
        new_alphabet=new_alphabet,
        initialization_strategy="best-fit",
        random_seed=random_seed,
        best_fit_kwargs={
            "new_hf_data": data,
            "sequence_column": "sequence",
            "label_column": "label",
            "return_fit_info": True,
            "dataset_kwargs": BEST_FIT_DATASET_KWARGS,
            **options,
        },
    )
    return tf.keras.models.load_model(model_path), adapted, fit_info


def _scores_by_substitution(model, data, token, substitutions=None):
    """What the search must find: per candidate, the pretrained model's median
    spectral angle on the sequences with ``token`` written as the candidate."""
    from dlomix.losses import masked_spectral_distance

    scores = {}
    for candidate in BEST_FIT_OLD_ALPHABET.keys() - {"-", "X"}:
        replacements = {token: candidate, **(substitutions or {})}

        def substitute(row):
            for old, new in replacements.items():
                row["sequence"] = row["sequence"].replace(old, new)
            return row

        dataset = FragmentIonIntensityDataset(
            data_format="hf",
            test_data_source=data.map(substitute),
            sequence_column="sequence",
            label_column="label",
            alphabet=BEST_FIT_OLD_ALPHABET,
            **BEST_FIT_DATASET_KWARGS,
        )
        inputs, labels = next(iter(dataset.tensor_test_data))
        distance = masked_spectral_distance(labels, model.predict(inputs, verbose=0))
        scores[candidate] = float(np.median(1 - np.asarray(distance)))
    return scores


def test_best_fit_picks_the_candidate_that_predicts_the_token_best(
    best_fit_model_path,
):
    data = _spectra(["AM[UNIMOD:1]CDEF", "EM[UNIMOD:1]CDEF", "CM[UNIMOD:1]CAAF"])

    model, adapted, fit_info = _best_fit(best_fit_model_path, data, ["M[UNIMOD:1]"])

    info = fit_info["M[UNIMOD:1]"]
    expected = _scores_by_substitution(model, data, "M[UNIMOD:1]")
    assert info["scores"] == pytest.approx(expected, abs=1e-6)
    assert info["old_token"] == max(info["scores"], key=info["scores"].get)
    assert info["eval"] == info["scores"][info["old_token"]]
    assert (info["initialization"], info["examples"]) == ("best-fit", "sole-token")
    np.testing.assert_array_equal(
        adapted.embedding.get_weights()[0][7],
        model.embedding.get_weights()[0][info["old_token_idx"]],
    )


def test_best_fit_with_several_new_tokens(best_fit_model_path):
    sole_a = ["CA[UNIMOD:1]CDEF", "EA[UNIMOD:1]CDEF", "CA[UNIMOD:1]CAAF"]
    a_and_b = ["CA[UNIMOD:1]CD[UNIMOD:2]F", "FA[UNIMOD:1]D[UNIMOD:2]EF"]
    sole_b = ["ACD[UNIMOD:2]EF"]
    data = _spectra(sole_a + a_and_b + sole_b)

    model, adapted, fit_info = _best_fit(
        best_fit_model_path,
        data,
        ["A[UNIMOD:1]", "D[UNIMOD:2]", "T[UNIMOD:3]"],
        min_examples_for_eval=2,
    )

    # enough spectra with A[UNIMOD:1] alone: those with D[UNIMOD:2] are left out
    info = fit_info["A[UNIMOD:1]"]
    assert info["examples"] == "sole-token"
    assert (info["n_with_token"], info["n_left_out"], info["n_searched"]) == (5, 2, 3)
    expected = _scores_by_substitution(model, data.select(range(3)), "A[UNIMOD:1]")
    assert info["scores"] == pytest.approx(expected, abs=1e-6)

    # too few with D[UNIMOD:2] alone: all its spectra, A[UNIMOD:1] encoded as A
    info = fit_info["D[UNIMOD:2]"]
    assert info["examples"] == "unmodified-fallback"
    assert (info["n_with_token"], info["n_left_out"], info["n_searched"]) == (3, 0, 3)
    expected = _scores_by_substitution(
        model, data.select(range(3, 6)), "D[UNIMOD:2]", {"A[UNIMOD:1]": "A"}
    )
    assert info["scores"] == pytest.approx(expected, abs=1e-6)

    # no spectra: the mean initialization stays, and the fit info says so
    info = fit_info["T[UNIMOD:3]"]
    assert (info["initialization"], info["old_token"]) == ("mean", None)
    assert (info["scores"], info["n_searched"]) == ({}, 0)
    np.testing.assert_allclose(
        adapted.embedding.get_weights()[0][9],
        model.embedding.get_weights()[0][2:].mean(axis=0),
        rtol=1e-6,
    )


def test_best_fit_samples_examples_with_the_seed(best_fit_model_path):
    data = _spectra(
        [f"{first}M[UNIMOD:1]CD{last}" for first in "ACDEF" for last in "AF"]
    )

    def scores(seed):
        _, _, fit_info = _best_fit(
            best_fit_model_path,
            data,
            ["M[UNIMOD:1]"],
            random_seed=seed,
            n_examples_for_eval=2,
        )
        assert fit_info["M[UNIMOD:1]"]["n_searched"] == 2
        return fit_info["M[UNIMOD:1]"]["scores"]

    assert scores(seed=0) == scores(seed=0)
    assert any(scores(seed) != scores(seed=0) for seed in (1, 2, 3))


def test_best_fit_enough_examples_is_capped_by_the_number_searched(
    best_fit_model_path,
):
    # three spectra with A[UNIMOD:1] alone are all that a search of three would
    # use: asking for more of them does not bring in the spectra with D[UNIMOD:2]
    sole_a = ["CA[UNIMOD:1]CDEF", "EA[UNIMOD:1]CDEF", "CA[UNIMOD:1]CAAF"]
    a_and_b = ["CA[UNIMOD:1]CD[UNIMOD:2]F", "FA[UNIMOD:1]D[UNIMOD:2]EF"]
    data = _spectra(sole_a + a_and_b)

    model, _, fit_info = _best_fit(
        best_fit_model_path,
        data,
        ["A[UNIMOD:1]", "D[UNIMOD:2]"],
        n_examples_for_eval=3,
        min_examples_for_eval=20,
    )

    info = fit_info["A[UNIMOD:1]"]
    assert (info["examples"], info["n_searched"]) == ("sole-token", 3)
    expected = _scores_by_substitution(model, data.select(range(3)), "A[UNIMOD:1]")
    assert info["scores"] == pytest.approx(expected, abs=1e-6)


def test_best_fit_fallback_keeps_the_spectra_with_the_token_alone(
    best_fit_model_path,
):
    sole_a = ["CA[UNIMOD:1]CDEF", "EA[UNIMOD:1]CDEF"]
    a_and_b = ["CA[UNIMOD:1]CD[UNIMOD:2]F", "FA[UNIMOD:1]D[UNIMOD:2]EF"]
    a_and_b.append("EA[UNIMOD:1]D[UNIMOD:2]CF")
    data = _spectra(sole_a + a_and_b)

    # too few alone (2 < 3): both are searched, plus one of the three others
    model = tf.keras.models.load_model(best_fit_model_path)
    possible = [
        _scores_by_substitution(
            model, data.select([0, 1, other]), "A[UNIMOD:1]", {"D[UNIMOD:2]": "D"}
        )
        for other in (2, 3, 4)
    ]
    for seed in range(4):
        _, _, fit_info = _best_fit(
            best_fit_model_path,
            data,
            ["A[UNIMOD:1]", "D[UNIMOD:2]"],
            random_seed=seed,
            n_examples_for_eval=3,
            min_examples_for_eval=3,
        )
        info = fit_info["A[UNIMOD:1]"]
        assert (info["examples"], info["n_searched"]) == ("unmodified-fallback", 3)
        assert any(info["scores"] == pytest.approx(p, abs=1e-6) for p in possible)
