import warnings

import tensorflow as tf

from ..constants import ALPHABET_UNMOD
from ..data.processing.feature_extractors import FEATURE_EXTRACTORS_PARAMETERS
from ..layers.attention import AttentionLayer, DecoderAttentionLayer


class PrositRetentionTimePredictor(tf.keras.Model):
    """
    Implementation of the Prosit model for retention time prediction.

    Parameters
    -----------
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
        self.embedding_output_dim = embedding_output_dim

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=self.embedding_output_dim,
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
    """
    Prosit model for intensity prediction.

    Parameters
    ----------
    embedding_output_dim : int, optional
        Size of the embeddings to use. Defaults to 16.
    seq_length : int, optional
        Sequence length of the peptide sequences. Defaults to 30.
    alphabet : dict, optional
        Dictionary mapping for the vocabulary (the amino acids in this case). Defaults to None, which is mapped to `ALPHABET_UNMOD`.
    dropout_rate : float, optional
        Probability to use for dropout layers in the encoder. Defaults to 0.5.
    latent_dropout_rate : float, optional
        Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
    recurrent_layers_sizes : tuple, optional
        A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
    regressor_layer_size : int, optional
        Size of the dense layer in the regressor after the encoder. Defaults to 512.
    use_prosit_ptm_features : boolean, optional
        Whether to use PTM features and create corresponding layers, has to be aligned with input_keys. Defaults to False.
    input_keys : dict, optional
        Dict of string keys and values mapping a fixed key to a value key in the inputs dict from the dataset class. Defaults to None, which corresponds then to the required default input keys `DEFAULT_INPUT_KEYS`.
    meta_data_keys : list, optional
        List of string values corresponding to fixed keys in the inputs dict that are considered meta data. Defaults to None, which corresponds then to the default meta data keys `META_DATA_KEYS`.
    with_termini : boolean, optional
        Whether to consider the termini in the sequence. Defaults to True.

    Attributes
    ----------
    DEFAULT_INPUT_KEYS : dict
        Default keys for the input dict.
    META_DATA_KEYS : list
        List of meta data keys.
    PTM_INPUT_KEYS : list
        List of PTM feature keys.
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

    # retrieve the Lookup PTM feature keys
    PTM_INPUT_KEYS = [*FEATURE_EXTRACTORS_PARAMETERS.keys()]

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        len_fion=6,
        alphabet=None,
        dropout_rate=0.2,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        use_prosit_ptm_features=False,
        input_keys=None,
        meta_data_keys=None,
        with_termini=True,
    ):
        super(PrositIntensityPredictor, self).__init__()

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes
        self.embedding_output_dim = embedding_output_dim
        self.seq_length = seq_length
        self.len_fion = len_fion
        self.use_prosit_ptm_features = use_prosit_ptm_features
        self.input_keys = input_keys
        self.meta_data_keys = meta_data_keys

        # maximum number of fragment ions
        self.max_ion = self.seq_length - 1

        # account for encoded termini
        if with_termini:
            self.max_ion = self.max_ion - 2

        if alphabet:
            self.alphabet = alphabet
        else:
            self.alphabet = ALPHABET_UNMOD

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(self.alphabet) + 2

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=self.embedding_output_dim,
            input_length=seq_length,
        )

        self._build_encoders()
        self._build_decoder()

        self.attention = AttentionLayer(name="encoder_att")

        self.meta_data_fusion_layer = None
        if self.meta_data_keys:
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
        # sequence encoder -> always present
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

        # meta data encoder -> optional, only if meta data keys are provided
        self.meta_encoder = None
        if self.meta_data_keys:
            self.meta_encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Concatenate(name="meta_in"),
                    tf.keras.layers.Dense(
                        self.recurrent_layers_sizes[1], name="meta_dense"
                    ),
                    tf.keras.layers.Dropout(self.dropout_rate, name="meta_dense_do"),
                ]
            )

        # ptm encoder -> optional, only if ptm flag is provided
        self.ptm_input_encoder, self.ptm_aa_fusion = None, None
        if self.use_prosit_ptm_features:
            self.ptm_input_encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Concatenate(name="ptm_features_concat"),
                    tf.keras.layers.Dense(self.regressor_layer_size // 2),
                    tf.keras.layers.Dropout(rate=self.dropout_rate),
                    tf.keras.layers.Dense(self.embedding_output_dim * 4),
                    tf.keras.layers.Dropout(rate=self.dropout_rate),
                    tf.keras.layers.Dense(self.embedding_output_dim),
                    tf.keras.layers.Dropout(rate=self.dropout_rate),
                ],
                name="ptm_input_encoder",
            )

            self.ptm_aa_fusion = tf.keras.layers.Concatenate(name="aa_ptm_in")

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
        encoded_meta = None
        encoded_ptm = None

        if not isinstance(inputs, dict):
            # when inputs has (seq, target), it comes as tuple
            peptides_in = inputs
        else:
            peptides_in = inputs.get(self.input_keys["SEQUENCE_KEY"])

            # read meta data from the input dict
            # note that the value here is the key to use in the inputs dict passed from the dataset
            meta_data = self._collect_values_from_inputs_if_exists(
                inputs, self.meta_data_keys
            )

            if self.meta_encoder and len(meta_data) > 0:
                encoded_meta = self.meta_encoder(meta_data)
            else:
                raise ValueError(
                    f"Following metadata keys were specified when creating the model: {self.meta_data_keys}, but the corresponding values do not exist in the input. The actual input passed to the model contains the following keys: {list(inputs.keys())}"
                )

            # read PTM features from the input dict
            ptm_ac_features = self._collect_values_from_inputs_if_exists(
                inputs, PrositIntensityPredictor.PTM_INPUT_KEYS
            )

            if self.ptm_input_encoder and len(ptm_ac_features) > 0:
                encoded_ptm = self.ptm_input_encoder(ptm_ac_features)
            elif self.use_prosit_ptm_features:
                warnings.warn(
                    f"PTM features enabled and following PTM features are expected in the model for Prosit Intesity: {PrositIntensityPredictor.PTM_INPUT_KEYS}. The actual input passed to the model contains the following keys: {list(inputs.keys())}. Falling back to no PTM features."
                )

        x = self.embedding(peptides_in)

        # fusion of PTMs (before going into the GRU sequence encoder)
        if self.ptm_aa_fusion and encoded_ptm is not None:
            x = self.ptm_aa_fusion([x, encoded_ptm])

        x = self.sequence_encoder(x)

        x = self.attention(x)

        if self.meta_data_fusion_layer and encoded_meta is not None:
            x = self.meta_data_fusion_layer([x, encoded_meta])
        else:
            # no metadata -> add a dimension to comply with the shape
            x = tf.expand_dims(x, axis=1)

        x = self.decoder(x)
        x = self.regressor(x)

        return x

    def _collect_values_from_inputs_if_exists(self, inputs, keys_mapping):
        collected_values = []

        keys = []
        if isinstance(keys_mapping, dict):
            keys = keys_mapping.values()

        elif isinstance(keys_mapping, list):
            keys = keys_mapping

        for key_in_inputs in keys:
            # get the input under the specified key if exists
            single_input = inputs.get(key_in_inputs, None)
            if single_input is not None:
                if single_input.ndim == 1:
                    single_input = tf.expand_dims(single_input, axis=-1)
                collected_values.append(single_input)
        return collected_values
