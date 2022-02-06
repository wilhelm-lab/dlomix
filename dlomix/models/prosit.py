import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from dlomix.constants import ALPHABET_UNMOD
from dlomix.layers.attention import AttentionLayer


class PrositRetentionTimePredictor(tf.keras.Model):
    """Implementation of the Prosit model for retention time prediction.

    Parameters
    -----------
        embedding_output_dim (int, optional): Size of the embeddings to use. Defaults to 16.
        seq_length (int, optional): Sequence length of the peptide sequences. Defaults to 30.
        vocab_dict (dict, optional): Dictionary mapping for the vocabulary (the amino acids in this case). Defaults to ALPHABET_UNMOD.
        dropout_rate (float, optional): Probability to use for dropout layers in the encoder. Defaults to 0.5.
        latent_dropout_rate (float, optional): Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
        recurrent_layers_sizes (tuple, optional): A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
        regressor_layer_size (int, optional): Size of the dense layer in the regressor after the encoder. Defaults to 512.
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        vocab_dict=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
    ):
        super(PrositRetentionTimePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(vocab_dict.keys())
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            input_length=seq_length,
        )
        self._build_encoder()

        self.attention = AttentionLayer()

        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.regressor_layer_size, activation="relu"),
                tf.keras.layers.Dropout(rate=self.latent_dropout_rate),
            ]
        )

        self.output_layer = tf.keras.layers.Dense(1)

    def _build_encoder(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(
                        units=self.recurrent_layers_sizes[0], return_sequences=True
                    )
                ),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
                tf.keras.layers.GRU(
                    units=self.recurrent_layers_sizes[1], return_sequences=True
                ),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
            ]
        )

    def call(self, inputs, **kwargs):
        x = self.string_lookup(inputs)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x
