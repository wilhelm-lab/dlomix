import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from ..constants import ALPHABET_UNMOD
from ..data.processing.feature_extractors import FEATURE_EXTRACTORS_PARAMETERS
from ..layers.attention_torch import AttentionLayer
from ..layers.bi_gru_seq_encoder_torch import BiGRUSequentialEncoder
from ..layers.gru_seq_decoder_torch import GRUSequentialDecoder

logger = logging.getLogger("dlomix.models.prosit_torch")


class PrositRetentionTimePredictor(nn.Module):
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
        self.embeddings_count = len(alphabet)

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes
        self.embedding_output_dim = embedding_output_dim
        self.seq_length = seq_length

        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=embedding_output_dim,
            padding_idx=0,  # TODO check this
        )

        self.encoder = BiGRUSequentialEncoder(
            embedding_output_dim, self.recurrent_layers_sizes, self.dropout_rate
        )

        self.attention = AttentionLayer(
            feature_dim=self.recurrent_layers_sizes[1], seq_len=self.seq_length
        )

        self.regressor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense",
                        nn.Linear(
                            in_features=self.recurrent_layers_sizes[1],
                            out_features=self.regressor_layer_size,
                        ),
                    ),
                    ("activation_relu", nn.ReLU()),
                    ("regressor_dropout", nn.Dropout(self.latent_dropout_rate)),
                ]
            )
        )

        self.output_layer = nn.Linear(
            in_features=self.regressor_layer_size, out_features=1
        )

    def forward(self, inputs, **kwargs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x


class PrositIntensityPredictor(nn.Module):
    """
    Implementation of the Prosit model for fragment ion intensity prediction.

    Parameters
    -----------





    """

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
        self.embeddings_count = len(self.alphabet) + 1

        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=self.embedding_output_dim,
        )

        self._build_encoders()
        self._build_decoder()

        self.attention = AttentionLayer(
            feature_dim=regressor_layer_size, seq_len=seq_length
        )

        self.meta_data_fusion_layer = None
        if self.meta_data_keys:
            self.meta_data_fusion_layer = MetaDataFusionBlock(max_ion=self.max_ion)

        self.regressor = nn.Sequential(
            OrderedDict(
                [
                    ("time_dense", nn.LazyLinear(out_features=len_fion)),
                    ("activation", nn.LeakyReLU()),
                    ("output", nn.Flatten()),
                ]
            )
        )

    def _build_encoders(self):
        # sequence encoder -> always present
        gru_features_input_size = self.embedding_output_dim

        if self.use_prosit_ptm_features:
            gru_features_input_size = self.embedding_output_dim * 2

        self.sequence_encoder = BiGRUSequentialEncoder(
            embedding_output_dim=gru_features_input_size,
            recurrent_layers_sizes=self.recurrent_layers_sizes,
            dropout_rate=self.dropout_rate,
        )

        # meta data encoder -> optional, only if meta data keys are provided
        self.meta_encoder = None
        if self.meta_data_keys:
            self.meta_encoder = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "meta_dense",
                            nn.LazyLinear(out_features=self.recurrent_layers_sizes[1]),
                        ),
                        ("dropout", nn.Dropout(p=self.dropout_rate)),
                    ]
                )
            )

        # ptm encoder -> optional, only if ptm flag is provided
        self.ptm_input_encoder, self.ptm_aa_fusion = None, None
        if self.use_prosit_ptm_features:
            self.ptm_input_encoder = nn.Sequential(
                Concatenate(dim=-1),
                nn.LazyLinear(out_features=self.regressor_layer_size // 2),
                nn.Dropout(p=self.dropout_rate),
                nn.LazyLinear(out_features=self.embedding_output_dim * 4),
                nn.Dropout(p=self.dropout_rate),
                nn.LazyLinear(out_features=self.embedding_output_dim),
                nn.Dropout(p=self.dropout_rate),
            )

            self.ptm_aa_fusion = Concatenate(dim=-1)

    def _build_decoder(self):
        self.decoder = GRUSequentialDecoder(
            recurrent_layers_sizes=self.recurrent_layers_sizes,
            dropout_rate=self.dropout_rate,
            max_ion=self.max_ion,
        )

    def forward(self, inputs, **kwargs):
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
                if isinstance(meta_data, list):
                    meta_data = torch.cat(meta_data, dim=-1)
                encoded_meta = self.meta_encoder(meta_data)

            elif self.meta_data_keys:
                raise ValueError(
                    f"Following metadata keys were specified when creating the model: {self.meta_data_keys}, but the corresponding values do not exist in the input. The actual input passed to the model contains the following keys: {list(inputs.keys())}"
                )
            else:
                pass

            # ToDo: ensure PTM features work as expected
            # read PTM features from the input dict # --> Still needs to be implemented
            ptm_ac_features = self._collect_values_from_inputs_if_exists(
                inputs, PrositIntensityPredictor.PTM_INPUT_KEYS
            )

            if self.ptm_input_encoder and len(ptm_ac_features) > 0:
                logger.debug("PTM features: ")
                for f in ptm_ac_features:
                    logger.debug(f.shape)
                encoded_ptm = self.ptm_input_encoder(ptm_ac_features)
            elif self.use_prosit_ptm_features:
                warnings.warn(
                    f"PTM features enabled and following PTM features are expected in the model for Prosit Intesity: {PrositIntensityPredictor.PTM_INPUT_KEYS}. The actual input passed to the model contains the following keys: {list(inputs.keys())}. Falling back to no PTM features."
                )

        x = self.embedding(peptides_in)

        # fusion of PTMs (before going into the GRU sequence encoder)
        if self.ptm_aa_fusion and encoded_ptm is not None:
            logger.debug("before ptm fusion: %s", x.shape)
            logger.debug("before ptm fusion enc ptm: %s", encoded_ptm.shape)
            x = self.ptm_aa_fusion([x, encoded_ptm])
            logger.debug("concatednated after fusion: %s", x.shape)

        x = self.sequence_encoder(x)

        x = self.attention(x)

        if self.meta_data_fusion_layer and encoded_meta is not None:
            x = self.meta_data_fusion_layer([x, encoded_meta])
        else:
            # no metadata -> add a dimension to comply with the shape
            x = torch.unsqueeze(x, axis=1)

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
                    single_input = torch.unsqueeze(single_input, axis=-1)
                collected_values.append(single_input)
        return collected_values


class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        for x in inputs:
            logger.debug("concatenating: %s ", x.shape)
        return torch.cat(inputs, dim=self.dim)


class MetaDataFusionBlock(torch.nn.Module):
    def __init__(self, max_ion):
        super(MetaDataFusionBlock, self).__init__()
        self.max_ion = max_ion

    def forward(self, x):
        #  x is a tuple of (features, metadata)
        features, metadata = x
        # Multiply operation
        multiplied = features * metadata
        # RepeatVector equivalent - expand along dimension to repeat max_ion times
        repeated = multiplied.unsqueeze(1).expand(-1, self.max_ion, -1)
        return repeated
