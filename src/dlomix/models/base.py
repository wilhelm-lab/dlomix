import tensorflow as tf

from ..constants import ALPHABET_UNMOD


class RetentionTimePredictor(tf.keras.Model):
    """
    A simple class for Retention Time prediction models.

    Parameters
    ----------

    embedding_dim: int, optional
        Dimensionality of the embeddings to be used for representing the Amino Acids. Defaults to ``16``.
    seq_length: int, optional
        Sequence length of the peptide sequences. Defaults to ``30``.
    encoder: str, optional
        String for specifying the decoder to use, either based on 1D conv-layers or LSTMs. Defaults to ``conv1d``.
    alphabet: dict, optional
        Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ``ALPHABET_UNMOD``.
    """

    def __init__(
        self,
        embedding_dim=16,
        seq_length=30,
        encoder="conv1d",
        alphabet=ALPHABET_UNMOD,
    ):
        super(RetentionTimePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(alphabet) + 2

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_dim,
            input_length=seq_length,
        )

        self._build_encoder(encoder)

        self.flatten = tf.keras.layers.Flatten()
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        self.output_layer = tf.keras.layers.Dense(1)

    def _build_encoder(self, encoder_type):
        if encoder_type.lower() == "conv1d":
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=256, kernel_size=3, padding="same", activation="relu"
                    ),
                    tf.keras.layers.Conv1D(
                        filters=512, kernel_size=3, padding="valid", activation="relu"
                    ),
                    tf.keras.layers.MaxPooling1D(pool_size=2),
                ]
            )
        else:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(256, return_sequences=True),
                    tf.keras.layers.LSTM(256),
                ]
            )

    def call(self, inputs, **kwargs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.regressor(x)
        x = self.output_layer(x)

        return x
