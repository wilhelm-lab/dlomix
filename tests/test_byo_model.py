"""Bring-your-own-model contract.

dlomix models are ordinary ``keras.Model`` subclasses -- nothing in the framework
requires a user model to inherit from a dlomix base class. These tests pin that
promise for the paths a third-party model actually travels: training on data the
dataset modules produce, the ``.keras`` save/load that the inference pipeline
uses, and the vocabulary expansion behind the fine-tuning pipeline.

They deliberately use a hand-written model with no dlomix inheritance, so the
contract cannot narrow without a test failing.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dlomix.models.model_utils import expand_embedding_vocabulary

ALPHABET = {token: index for index, token in enumerate("ACDEF")}
SEQ_LENGTH = 10


@keras.saving.register_keras_serializable(package="tests")
class UserModel(keras.Model):
    """A minimal third-party model: plain Keras, no dlomix base class.

    ``alphabet`` is a constructor argument, which is all that is needed for
    vocabulary expansion -- Keras 3 captures ``__init__`` arguments into
    ``get_config()`` automatically, so no ``get_config`` override is required.
    """

    def __init__(self, alphabet=ALPHABET, embedding_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.alphabet = dict(alphabet)
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(
            len(self.alphabet), embedding_dim, name="embedding"
        )
        self.encoder = keras.layers.GRU(16)
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs):
        return self.output_layer(self.encoder(self.embedding(inputs)))


@pytest.fixture
def sequences():
    return np.random.randint(0, len(ALPHABET), (8, SEQ_LENGTH))


@pytest.fixture
def trained_model(sequences):
    model = UserModel()
    model.compile(optimizer="adam", loss="mse")
    labels = np.random.rand(len(sequences), 1)
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels)).batch(4)
    model.fit(dataset, epochs=1, verbose=0)
    return model


def test_trains_on_a_tf_dataset(trained_model, sequences):
    """A user model trains on the tf.data pipeline the dataset modules produce."""
    assert trained_model.predict(sequences, verbose=0).shape == (len(sequences), 1)


def test_keras_save_load_round_trip(trained_model, sequences, tmp_path):
    """The .keras round-trip used by InferencePipeline preserves predictions."""
    path = tmp_path / "user_model.keras"
    trained_model.save(path)

    reloaded = keras.saving.load_model(path)

    assert isinstance(reloaded, UserModel)
    np.testing.assert_allclose(
        trained_model.predict(sequences, verbose=0),
        reloaded.predict(sequences, verbose=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_vocabulary_expansion(trained_model, sequences):
    """Vocabulary expansion works on a model with no dlomix inheritance.

    Only the embedding grows; the other weights carry over untouched.
    """
    new_alphabet = {**ALPHABET, "G": 5, "H": 6}
    old_encoder_weights = [w.numpy() for w in trained_model.encoder.weights]

    adapted = expand_embedding_vocabulary(
        model=trained_model, new_alphabet=new_alphabet, old_alphabet=ALPHABET
    )

    assert adapted.embedding.input_dim == len(new_alphabet)
    assert adapted.predict(sequences, verbose=0).shape == (len(sequences), 1)
    for before, after in zip(old_encoder_weights, adapted.encoder.weights):
        np.testing.assert_allclose(before, after.numpy(), rtol=1e-6)


def test_vocabulary_expansion_requires_a_built_model():
    """An unbuilt model is rejected with an actionable message, not an IndexError."""
    with pytest.raises(ValueError, match="Build the model"):
        expand_embedding_vocabulary(
            model=UserModel(),
            new_alphabet={**ALPHABET, "G": 5},
            old_alphabet=ALPHABET,
        )


def test_vocabulary_expansion_requires_an_alphabet_in_the_config(sequences):
    """A model that does not expose its alphabet is rejected explicitly."""

    class VocabSizeModel(keras.Model):
        def __init__(self, vocab_size=len(ALPHABET), **kwargs):
            super().__init__(**kwargs)
            self.embedding = keras.layers.Embedding(vocab_size, 8, name="embedding")
            self.output_layer = keras.layers.Dense(1)

        def call(self, inputs):
            return self.output_layer(self.embedding(inputs)[:, -1, :])

    model = VocabSizeModel()
    model.predict(sequences, verbose=0)

    with pytest.raises(ValueError, match="alphabet"):
        expand_embedding_vocabulary(
            model=model, new_alphabet={**ALPHABET, "G": 5}, old_alphabet=ALPHABET
        )
