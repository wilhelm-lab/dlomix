import tensorflow as tf

from ..constants import ALPHABET_UNMOD
from ..layers.attention import AttentionLayer, DecoderAttentionLayer


class PrositRetentionTimePredictor(tf.keras.Model):
    r"""Implementation of the Prosit model for retention time prediction.

    Parameters
    -----------
    embedding_output_dim: int, optional
        Size of the embeddings to use. Defaults to 16.
    seq_length: int, optional
        Sequence length of the peptide sequences. Defaults to 30.
    alphabet: dict, optional
        Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
    dropout_rate: float, optional
        Probability to use for dropout layers in the encoder. Defaults to 0.5.
    latent_dropout_rate: float, optional
        Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
    recurrent_layers_sizes: tuple, optional
        A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
    regressor_layer_size: int, optional
        Size of the dense layer in the regressor after the encoder. Defaults to 512.
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        alphabet=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
    ):
        super(PrositRetentionTimePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(alphabet) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

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
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x


class PrositIntensityPredictor(tf.keras.Model):
    r"""Implementation of the Prosit model for intensity prediction.

    Parameters
    ----------
    embedding_output_dim : int, optional
        Size of the embeddings to use. Defaults to 16.
    seq_length : int, optional
        Sequence length of the peptide sequences. Defaults to 30.
    alphabet : dict, optional
        Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
    dropout_rate : float, optional
        Probability to use for dropout layers in the encoder. Defaults to 0.5.
    latent_dropout_rate : float, optional
        Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
    recurrent_layers_sizes : tuple, optional
        A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
    regressor_layer_size : int, optional
        Size of the dense layer in the regressor after the encoder. Defaults to 512.
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        len_fion=6,
        alphabet=ALPHABET_UNMOD,
        sequence_input_name="sequence",
        collision_energy_input_name="collision_energy_aligned_normed",
        precursor_charge_input_name="precursor_charge_onehot",
        fragmentation_method_input_name="method_nbr",
        dropout_rate=0.2,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
    ):
        super(PrositIntensityPredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(alphabet) + 2

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes
        self.seq_length = seq_length
        self.len_fion = len_fion

        self.sequence_input_name = sequence_input_name
        self.collision_energy_input_name = collision_energy_input_name
        self.precursor_charge_input_name = precursor_charge_input_name
        self.fragmentation_method_input_name = fragmentation_method_input_name

        # maximum number of fragment ions
        self.max_ion = self.seq_length - 1

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            input_length=seq_length,
        )

        self._build_encoders()
        self._build_decoder()

        self.attention = AttentionLayer(name="encoder_att")

        self.fusion_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Multiply(name="add_meta"),
                tf.keras.layers.RepeatVector(self.max_ion, name="repeat"),
            ]
        )

        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(self.len_fion), name="time_dense"
                ),
                tf.keras.layers.LeakyReLU(name="activation"),
                tf.keras.layers.Flatten(name="out"),
            ]
        )

    def _build_encoders(self):
        self.meta_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Concatenate(name="meta_in"),
                tf.keras.layers.Dense(
                    self.recurrent_layers_sizes[1], name="meta_dense"
                ),
                tf.keras.layers.Dropout(self.dropout_rate, name="meta_dense_do"),
            ]
        )

        self.sequence_encoder = tf.keras.Sequential(
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

    def _build_decoder(self):
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    units=self.regressor_layer_size,
                    return_sequences=True,
                    name="decoder",
                ),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
                DecoderAttentionLayer(self.max_ion),
            ]
        )

    def call(self, inputs, **kwargs):
        peptides_in = inputs[self.sequence_input_name]
        collision_energy_in = inputs[self.collision_energy_input_name]
        precursor_charge_in = inputs[self.precursor_charge_input_name]
        # fragmentation_method_in = inputs[self.fragmentation_method_input_name]

        collision_energy_in = tf.expand_dims(collision_energy_in, axis=1)
        precursor_charge_in = tf.cast(precursor_charge_in, tf.float32)
        # fragmentation_method_in = tf.expand_dims(fragmentation_method_in, axis=1)
        # fragmentation_method_in = tf.cast(fragmentation_method_in, tf.float32)

        encoded_meta = self.meta_encoder(
            [collision_energy_in, precursor_charge_in]  # , fragmentation_method_in]
        )

        x = self.embedding(peptides_in)
        x = self.sequence_encoder(x)
        x = self.attention(x)

        x = self.fusion_layer([x, encoded_meta])

        x = self.decoder(x)

        x = self.regressor(x)

        return x
