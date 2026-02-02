import logging
from collections import OrderedDict
from collections.abc import Sequence

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
    Prosit model for intensity prediction with configurable branches for PTM features and metadata.

    Parameters
    ----------
    input_keys : dict, optional
        Dictionary mapping for the input keys to look for in the input dict. Defaults to None, which uses default required keys only "seqeuence".
    meta_data_keys : list or dict, optional
        List or dict of keys for the meta data inputs to use. Defaults to None (no meta data).
    alphabet : dict, optional
        Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
    with_termini : bool, optional
        Whether to include terminal tokens in the sequence embedding. Defaults to False.
    embedding_output_dim : int, optional
        Size of the embeddings to use. Defaults to 16.
    seq_length : int, optional
        Sequence length of the peptide sequences. Defaults to 30.
    len_fragment_ion : int, optional
        Number of fragment ions to predict. Defaults to 6.
    dropout_rate : float, optional
        Probability to use for dropout layers in the encoder. Defaults to 0.2.
    latent_dropout_rate : float, optional
        Probability to use for dropout layers in the regressor layers after encoding. Defaults to 0.1.
    recurrent_layers_sizes : tuple, optional
        A tuple of 2 values for the sizes of the two GRU layers in the encoder. Defaults to (256, 512).
    regressor_layer_size : int, optional
        Size of the dense layer in the regressor after the encoder. Defaults to 512.
    use_meta_data : bool, optional
        Whether to use meta data inputs. Defaults to False.
    use_prosit_ptm_features : bool, optional
        Whether to use Prosit PTM features as input. Defaults to False.
    use_instrument_embedding : bool, optional
        Whether to use instrument type embedding as part of the meta data. Defaults to False.
    instrument_input_dim : int, optional
        Number of unique instrument types for embedding. Defaults to 3.
    instrument_output_dim : int, optional
        Size of the instrument type embedding. Defaults to 2.



    Attributes
    ----------

    REQUIRED_INPUT_SEQUENCE_KEY : str
        Key for the required sequence input in the input dictionary.
    DEFAULT_INPUT_KEYS : dict
        Default mapping of input keys for various inputs.
    META_DATA_KEYS : list
        List of possible meta data keys that can be used.
    PTM_INPUT_KEYS : list
        List of keys for the PTM feature inputs. See `dlomix.data.processing.feature_extractors.FEATURE_EXTRACTORS_PARAMETERS` for details.


    Notes
    -----
    This model is a flexible implementation of the Prosit intensity predictor, allowing for optional
    inclusion of PTM features and meta data inputs. The model architecture consists of embedding layers,
    bidirectional GRU encoders, attention mechanisms, and dense regressor layers.






    """

    REQUIRED_INPUT_SEQUENCE_KEY = "SEQUENCE_KEY"

    DEFAULT_INPUT_KEYS = {
        REQUIRED_INPUT_SEQUENCE_KEY: "sequence",
        "COLLISION_ENERGY_KEY": "collision_energy",
        "PRECURSOR_CHARGE_KEY": "precursor_charge",
        "FRAGMENTATION_TYPE_KEY": "fragmentation_type",
        "INSTRUMENT_TYPE_KEY": "instrument_type",
    }

    # can be extended to include all possible meta data
    META_DATA_KEYS = [
        "COLLISION_ENERGY_KEY",
        "PRECURSOR_CHARGE_KEY",
        "FRAGMENTATION_TYPE_KEY",
        "INSTRUMENT_TYPE_KEY",
    ]

    # retrieve the Lookup PTM feature keys
    PTM_INPUT_KEYS = [*FEATURE_EXTRACTORS_PARAMETERS.keys()]

    def __init__(
        self,
        input_keys=None,
        meta_data_keys=None,
        alphabet=None,
        with_termini=False,
        embedding_output_dim=16,
        seq_length=30,
        len_fragment_ion=6,
        dropout_rate=0.2,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        use_meta_data=False,
        use_prosit_ptm_features=False,
        use_instrument_embedding=False,
        instrument_input_dim=3,
        instrument_output_dim=2,
        **kwargs,
    ):
        super(PrositIntensityPredictor, self).__init__(**kwargs)

        # Store all configuration parameters
        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = tuple(recurrent_layers_sizes)
        self.embedding_output_dim = embedding_output_dim
        self.raw_seq_length = seq_length
        self.len_fragment_ion = len_fragment_ion
        self.use_prosit_ptm_features = use_prosit_ptm_features
        self.with_termini = with_termini
        self.use_meta_data = use_meta_data
        self.use_instrument_embedding = use_instrument_embedding
        self.instrument_input_dim = instrument_input_dim
        self.instrument_output_dim = instrument_output_dim

        # handle default and fallback attributes
        self._handle_alphabet_and_keys(alphabet, input_keys, meta_data_keys)

        self._validate_config()

        self._compute_attributes()

        # Build layers
        self._build_embedding_layers()
        self._build_encoders()
        self._build_decoder()
        self.attention = AttentionLayer(
            feature_dim=regressor_layer_size, seq_len=seq_length
        )
        self._build_meta_data_fusion_layer()
        self._build_regressor()

    def _handle_alphabet_and_keys(self, alphabet, input_keys, meta_data_keys):
        # Handle alphabet
        if alphabet is not None:
            self.alphabet = dict(alphabet)
        else:
            self.alphabet = dict(ALPHABET_UNMOD)  # fallback to unmodified amino acids

        # Handle input_keys
        if input_keys:
            self.input_keys = dict(input_keys)
        else:
            self.input_keys = {
                self.REQUIRED_INPUT_SEQUENCE_KEY: "sequence"
            }  # minimal required key for the sequence input

        # Handle meta_data_keys
        if not meta_data_keys:
            self.meta_data_keys = []
        else:
            # handle dict in case user passed a mapping, take the values in a list to use for lookup
            if isinstance(meta_data_keys, dict):
                self.meta_data_keys = list(meta_data_keys.values())
            elif isinstance(meta_data_keys, list):
                self.meta_data_keys = list(meta_data_keys)
            else:
                raise ValueError(
                    "meta_data_keys should be either a list of strings or a dict mapping. Provided type: "
                    f"{type(meta_data_keys)}, and value: {meta_data_keys}"
                )

    def _validate_config(self):
        if self.use_meta_data and not self.meta_data_keys:
            raise ValueError(
                "use_meta_data=True requires meta_data_keys to be provided as a list of keys."
            )

        if (
            self.use_instrument_embedding
            and "INSTRUMENT_TYPE_KEY" not in self.input_keys
        ):
            raise ValueError(
                "use_instrument_embedding=True requires 'INSTRUMENT_TYPE_KEY' in input_keys"
            )

    def _compute_attributes(self):
        # Compute derived attributes (will be recomputed during deserialization)
        self.max_ion = self.raw_seq_length - 1

        self.seq_length = (
            self.raw_seq_length + 2 if self.with_termini else self.raw_seq_length
        )

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(self.alphabet)

    def _build_embedding_layers(self):
        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=self.embedding_output_dim,
        )

        self.instrument_embedding = None
        if self.use_instrument_embedding:
            self.instrument_embedding = nn.Embedding(
                num_embeddings=self.instrument_input_dim,
                embedding_dim=self.instrument_output_dim,
            )

    def _build_meta_data_fusion_layer(self):
        if self.meta_data_keys:
            self.meta_data_fusion_layer = MetaDataFusionBlock(max_ion=self.max_ion)

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
        if self.use_meta_data:
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

    def _build_regressor(self):
        self.regressor = nn.Sequential(
            OrderedDict(
                [
                    ("time_dense", nn.LazyLinear(out_features=self.len_fragment_ion)),
                    ("activation", nn.LeakyReLU()),
                    ("output", nn.Flatten()),
                ]
            )
        )

    def forward(self, inputs, **kwargs):
        # Handle dict input, complex case: multiple inputs
        if isinstance(inputs, dict):
            return self._forward_dict(inputs, **kwargs)

        # Handle single input, simple case: sequence only
        peptides_in = inputs
        return self._forward_sequence_only(peptides_in, **kwargs)

    def _forward_sequence_only(self, peptides_in, **kwargs):
        x = self.embedding(peptides_in)
        x = self.sequence_encoder(x)
        x = self.attention(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.decoder(x)
        x = self.regressor(x)
        return x

    def _forward_dict(self, inputs, **kwargs):
        missing_input_keys = [k for k in self.input_keys.values() if k not in inputs]
        if missing_input_keys:
            raise ValueError(f"Missing required input keys: {missing_input_keys}")

        meta_data = []
        encoded_meta = None
        encoded_ptm = None

        # collect instrument embedding if enabled and add to meta data
        if self.use_instrument_embedding:
            instrument_type = inputs.get(self.input_keys["INSTRUMENT_TYPE_KEY"])
            instrument_embedded = self.instrument_embedding(instrument_type)
            meta_data.append(instrument_embedded)

        # collect meta data from the input dict
        if self.use_meta_data:
            missing_meta_keys = [k for k in self.meta_data_keys if k not in inputs]
            if missing_meta_keys:
                raise ValueError(
                    f"Missing required metadata inputs: {missing_meta_keys}"
                )

            meta_data.extend(
                self._collect_values_from_inputs_if_exists(inputs, self.meta_data_keys)
            )

        # collect PTM features from the input dict
        if self.use_prosit_ptm_features:
            ptm_keys_exist = [k for k in self.PTM_INPUT_KEYS if k in inputs]
            if not ptm_keys_exist:
                raise ValueError(
                    f"At least one PTM input feature is required when use_prosit_ptm_features=True. Missing all of: {self.PTM_INPUT_KEYS}"
                )

            ptm_ac_features = self._collect_values_from_inputs_if_exists(
                inputs, PrositIntensityPredictor.PTM_INPUT_KEYS
            )

        peptides_in = inputs[self.input_keys[self.REQUIRED_INPUT_SEQUENCE_KEY]]

        x = self.embedding(peptides_in)

        # encode and fuse PTM features (before going into the GRU sequence encoder)
        if self.use_prosit_ptm_features:
            encoded_ptm = self.ptm_input_encoder(ptm_ac_features)
            x = self.ptm_aa_fusion([x, encoded_ptm])

        x = self.sequence_encoder(x)
        x = self.attention(x)

        if self.use_meta_data:
            if isinstance(meta_data, list):
                meta_data = torch.cat(meta_data, dim=-1)
            encoded_meta = self.meta_encoder(meta_data)
            x = self.meta_data_fusion_layer([x, encoded_meta])
        else:
            # no metadata -> add a dimension to comply with the shape
            x = torch.unsqueeze(x, dim=1)

        x = self.decoder(x)
        x = self.regressor(x)

        return x

    def _collect_values_from_inputs_if_exists(self, inputs, keys_mapping):
        collected_values = []
        keys = []

        if isinstance(keys_mapping, dict):
            keys = list(keys_mapping.values())

        elif isinstance(keys_mapping, Sequence):
            keys = list(keys_mapping)

        for key_in_inputs in keys:
            # get the input under the specified key if exists
            single_input = inputs.get(key_in_inputs, None)
            if single_input is not None:
                if single_input.ndim == 1:
                    single_input = torch.unsqueeze(single_input, dim=-1)
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
