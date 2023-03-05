import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from dlomix.constants import ALPHABET_UNMOD
from dlomix.layers.attention import AttentionLayer, DecoderAttentionLayer

from ..data.feature_extractors import (
    ModificationGainFeature,
    ModificationLocationFeature,
    ModificationLossFeature,
)


class PrositRetentionTimePredictor(tf.keras.Model):
    """Implementation of the Prosit model for retention time prediction.

    Parameters
    -----------
        embedding_output_dim (int, optional): Size of the embeddings to use. Defaults to 16.
        seq_length (int, optional): Sequence length of the peptide sequences. Defaults to 30.
        vocab_dict (dict, optional): Dictionary mapping for the vocabulary (the amino acids in this case).  Defaults to None, which is mapped to `ALPHABET_UNMOD`.
        dropout_rate (float, optional): Probability to use for dropout layers in the encoder. Defaults to 0.5.
        latent_dropout_rate (float, optional): Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
        recurrent_layers_sizes (tuple, optional): A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
        regressor_layer_size (int, optional): Size of the dense layer in the regressor after the encoder. Defaults to 512.
    """

    DEFAULT_INPUT_KEYS = {
        "SEQUENCE_KEY": "sequence",
    }

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        vocab_dict=None,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
    ):
        super(PrositRetentionTimePredictor, self).__init__()

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        if vocab_dict:
            self.vocab_dict = vocab_dict
        else:
            self.vocab_dict = ALPHABET_UNMOD

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(self.vocab_dict) + 2

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(self.vocab_dict.keys())
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
        if isinstance(inputs, dict):
            x = inputs.get(
                PrositRetentionTimePredictor.DEFAULT_INPUT_KEYS["SEQUENCE_KEY"]
            )
        else:
            x = inputs
        x = self.string_lookup(x)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x


class PrositIntensityPredictor(tf.keras.Model):
    """Implementation of the Prosit model for intensity prediction.

    Parameters
    -----------
        embedding_output_dim (int, optional): Size of the embeddings to use. Defaults to 16.
        seq_length (int, optional): Sequence length of the peptide sequences. Defaults to 30.
        vocab_dict (dict, optional): Dictionary mapping for the vocabulary (the amino acids in this case). Defaults to None, which is mapped to `ALPHABET_UNMOD`.
        dropout_rate (float, optional): Probability to use for dropout layers in the encoder. Defaults to 0.5.
        latent_dropout_rate (float, optional): Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
        recurrent_layers_sizes (tuple, optional): A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
        regressor_layer_size (int, optional): Size of the dense layer in the regressor after the encoder. Defaults to 512.
        use_ptm_counts (boolean, optional): Whether to use PTM counts and create corresponding layers, has to be aligned with input_keys. Defaults to False.
        input_keys (dict, optional): dict of string keys and values mapping a fixed key to a value key in the inputs dict from the dataset class. Defaults to None, which corresponds then to the required default input keys `DEFAULT_INPUT_KEYS`.
        meta_data_keys (list, optional): list of string values corresponding to fixed keys in the inputs dict that are considered meta data. Defaults to None, which corresponds then to the default meta data keys `META_DATA_KEYS`.
    """

    # consider using kwargs in the call function instead !

    DEFAULT_INPUT_KEYS = {
        "SEQUENCE_KEY": "sequence",
        "COLLISION_ENERGY_KEY": "collision_energy",
        "PRECURSOR_CHARGE_KEY": "precursor_charge",
        "FRAGMENTATION_TYPE_KEY": "fragmentation_type",
    }

    # can be extended to include all possible meta data
    META_DATA_KEYS = [
        "COLLISION_ENERGY_KEY",
        "PRECURSOR_CHARGE_KEY",
        "FRAGMENTATION_TYPE_KEY",
    ]
    PTM_INPUT_KEYS = [
        ModificationLossFeature.__name__.lower(),
        ModificationGainFeature.__name__.lower(),
        # ModificationLocationFeature.__name__.lower()
    ]

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        len_fion=6,
        vocab_dict=None,
        dropout_rate=0.2,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        use_ptm_counts=False,
        input_keys=None,
        meta_data_keys=None,
    ):
        super(PrositIntensityPredictor, self).__init__()

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes
        self.seq_length = seq_length
        self.len_fion = len_fion
        self.use_ptm_counts = use_ptm_counts
        self.input_keys = input_keys
        self.meta_data_keys = meta_data_keys

        # maximum number of fragment ions
        self.max_ion = self.seq_length - 1

        if vocab_dict:
            self.vocab_dict = vocab_dict
        else:
            self.vocab_dict = ALPHABET_UNMOD

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(self.vocab_dict) + 2

        self.string_lookup = preprocessing.StringLookup(
            vocabulary=list(self.vocab_dict.keys())
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_output_dim,
            input_length=seq_length,
        )

        if self.input_keys is None:
            self.input_keys = PrositIntensityPredictor.DEFAULT_INPUT_KEYS

        if self.meta_data_keys is None:
            self.meta_data_keys = PrositIntensityPredictor.META_DATA_KEYS

        self._build_encoders()
        self._build_decoder()

        self.attention = AttentionLayer(name="encoder_att")

        self.meta_data_fusion_layer = tf.keras.Sequential(
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
        if not self.use_ptm_counts:
            self.ptm_encoder, self.ptm_aa_fusion = None, None
        else:
            self.ptm_encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Concatenate(name="ptm_ac_loss_gain"),
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

            self.ptm_aa_fusion = tf.keras.layers.Multiply(name="aa_ptm_in")

    def _build_decoder(self):
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    units=self.recurrent_layers_sizes[1],
                    return_sequences=True,
                    name="decoder",
                ),
                tf.keras.layers.Dropout(rate=self.dropout_rate),
                DecoderAttentionLayer(self.max_ion),
            ]
        )

    def call(self, inputs, **kwargs):
        peptides_in = inputs.get(self.input_keys["SEQUENCE_KEY"])

        # read meta data from the input dict
        meta_data = []
        # note that the value here is the key to use in the inputs dict passed from the dataset
        for meta_key, key_in_inputs in self.input_keys.items():
            if meta_key in PrositIntensityPredictor.META_DATA_KEYS:
                # get the input under the specified key if exists
                meta_in = inputs.get(key_in_inputs, None)
                if meta_in is not None:
                    # add the input to the list of meta data inputs
                    meta_data.append(meta_in)

        if self.meta_encoder and len(meta_data) > 0:
            encoded_meta = self.meta_encoder(meta_data)
        else:
            raise ValueError(
                f"Following metadata keys are expected in the model for Prosit Intesity: {PrositIntensityPredictor.META_DATA_KEYS}. The actual input passed to the model contains the following keys: {list(inputs.keys())}"
            )

        # read PTM atom count features from the input dict
        ptm_ac_features = []
        for ptm_key in PrositIntensityPredictor.PTM_INPUT_KEYS:
            ptm_ac_f = inputs.get(ptm_key, None)
            if ptm_ac_f is not None:
                ptm_ac_features.append(ptm_ac_f)

        if self.ptm_encoder and len(ptm_ac_features) > 0:
            encoded_ptm = self.ptm_encoder(ptm_ac_features)
        elif self.use_ptm_counts:
            raise ValueError(
                f"PTM features enabled and following PTM features are expected in the model for Prosit Intesity: {PrositIntensityPredictor.PTM_INPUT_KEYS}. The actual input passed to the model contains the following keys: {list(inputs.keys())}"
            )
        else:
            encoded_ptm = None

        x = self.string_lookup(peptides_in)
        x = self.embedding(x)
        x = self.sequence_encoder(x)

        if self.use_ptm_counts and self.ptm_aa_fusion and encoded_ptm is not None:
            x = self.ptm_aa_fusion([x, encoded_ptm])

        x = self.attention(x)

        x = self.meta_data_fusion_layer([x, encoded_meta])

        x = self.decoder(x)
        x = self.regressor(x)

        return x
