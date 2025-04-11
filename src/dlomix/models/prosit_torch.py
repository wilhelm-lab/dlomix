import warnings
import collections as col
import torch
import torch.nn as nn
from ..constants import ALPHABET_UNMOD
# from ..data.processing.feature_extractors import FEATURE_EXTRACTORS_PARAMETERS
from ..layers.attention_torch import AttentionLayerTorch, DecoderAttentionLayerTorch
from dlomix.layers.bi_gru_seq_encoder_torch import BiGRUSequentialEncoder
from dlomix.layers.gru_seq_decoder import GRUSequentialDecoder

class PrositIntensityPredictorTorch(nn.Module):
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
        super(PrositIntensityPredictorTorch, self).__init__()

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
        self.embeddings_count = len(self.alphabet)

        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=self.embedding_output_dim,
        )

        self._build_encoders()
        self._build_decoder()

        self.attention =  AttentionLayerTorch(feature_dim=regressor_layer_size, seq_len=seq_length)

        self.meta_data_fusion_layer = None
        # if self.meta_data_keys: ---> Still needs to be fixed
        #     self.meta_data_fusion_layer = nn.Sequential(
        #         col.OrderedDict([
        #             torch.mul(name="add_meta"),
        #             torch.Tensor.repeat(self.max_ion, name="repeat"),
        #         ])
        #     )

        
        
        self.regressor = nn.Sequential(
            col.OrderedDict([
                ("time_dense", nn.LazyLinear(out_features=len_fion)),
                ("activation", nn.LeakyReLU()),
                ("output", nn.Flatten())
            ])
        )
      

    def _build_encoders(self):
        # sequence encoder -> always present
        self.sequence_encoder = BiGRUSequentialEncoder(embedding_output_dim=self.embedding_output_dim, 
                                                       recurrent_layers_sizes=self.recurrent_layers_sizes, 
                                                       dropout_rate=self.dropout_rate)

        # # meta data encoder -> optional, only if meta data keys are provided -- not yet adapted
        self.meta_encoder = None
        if self.meta_data_keys:
            self.meta_encoder = nn.Sequential(col.OrderedDict([ # use cat? or is nn.seq fine?
                ("meta_dense", nn.LazyLinear(out_features=self.recurrent_layers_sizes[1])),
                ("dropout", nn.Dropout(p=self.dropout_rate))
            ]))

        # # # ptm encoder -> optional, only if ptm flag is provided -- not yet properly adapted
        self.ptm_input_encoder, self.ptm_aa_fusion = None, None
        if self.use_prosit_ptm_features:
              self.ptm_input_encoder = nn.Sequential(col.OrderedDict([
                ("linear1", nn.LazyLinear(out_features=self.regressor_layer_size // 2)),
                ("dropout1", nn.Dropout(p=self.dropout_rate)),
                ("linear2", nn.LazyLinear(out_features=self.embedding_output_dim * 4)),
                ("dropout2", nn.Dropout(p=self.dropout_rate)),
                ("linear3", nn.LazyLinear(out_features=self.embedding_output_dim)),
                ("dropout3", nn.Dropout(p=self.dropout_rate))
            ]))

        # self.ptm_aa_fusion = torch.cat((self.sequence_encoder, self.ptm_input_encoder))

    def _build_decoder(self):
        self.decoder = GRUSequentialDecoder(recurrent_layers_sizes=self.recurrent_layers_sizes, 
                                            dropout_rate=self.dropout_rate,
                                            max_ion=self.max_ion)

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

            #print(len(meta_data))
            #print(self.meta_encoder)

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
                
            
            
            
            # read PTM features from the input dict # --> Still needs to be implemented
            # ptm_ac_features = self._collect_values_from_inputs_if_exists(
            #     inputs, PrositIntensityPredictorTorch.PTM_INPUT_KEYS
            # )

            # if self.ptm_input_encoder and len(ptm_ac_features) > 0:
            #     encoded_ptm = self.ptm_input_encoder(ptm_ac_features)
            # elif self.use_prosit_ptm_features:
            #     warnings.warn(
            #         f"PTM features enabled and following PTM features are expected in the model for Prosit Intesity: {PrositIntensityPredictorTorch.PTM_INPUT_KEYS}. The actual input passed to the model contains the following keys: {list(inputs.keys())}. Falling back to no PTM features."
            #     )

        #print(peptides_in.shape)
        x = self.embedding(peptides_in)
        #print(x.shape)

        # fusion of PTMs (before going into the GRU sequence encoder)
        if self.ptm_aa_fusion and encoded_ptm is not None:
            x = self.ptm_aa_fusion([x, encoded_ptm])

        x = self.sequence_encoder(x)
        #print(f"encoder {x.shape}")
        x = self.attention(x)
        #print(f"attention {x.shape}")

        if self.meta_data_fusion_layer and encoded_meta is not None:
            x = self.meta_data_fusion_layer([x, encoded_meta])
        else:
            # no metadata -> add a dimension to comply with the shape
            x = torch.unsqueeze(x, axis=1)

        x = self.decoder(x)
        #print(f"decoder: {x.shape}")
        x = self.regressor(x)
        #print(f"regressor {x.shape}")
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
