from .feature_tables import PTM_LOSS_LOOKUP, PTM_MOD_ATOM_LOOKUP
from .processors import FunctionProcessor, PeptideDatasetBaseProcessor

FEATURE_EXTRACTORS_PARAMETERS = {
    "mod_loss": {
        "feature_default_value": [0, 0, 0, 0, 0, 0],
        "lookup_table": PTM_LOSS_LOOKUP,
    },
    "mod_atom": {
        "feature_default_value": 0,
        "lookup_table": PTM_MOD_ATOM_LOOKUP,
    },
}

AVAILABLE_FEATURE_EXTRACTORS = list(FEATURE_EXTRACTORS_PARAMETERS.keys())


class FeatureExtractor(PeptideDatasetBaseProcessor):
    def __init__(
        self,
        sequence_column_name: str,
        feature_column_name: str,
        feature_default_value,
        max_length: int = 30,
        batched: bool = False,
    ):
        super().__init__(sequence_column_name, batched)
        self.feature_column_name = feature_column_name
        self.feature_default_value = feature_default_value
        self.max_length = max_length

    def pad_feature_to_seq_length(self, single_feature):
        feature_length = len(single_feature)

        if feature_length > self.max_length:
            raise ValueError(
                f"Feature length ({len(single_feature)}) is longer than sequence length provided ({self.max_length})."
            )

        padding_length = self.max_length - feature_length
        single_feature += [self.feature_default_value] * padding_length

        return single_feature


class LookupFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        sequence_column_name: str,
        feature_column_name: str,
        feature_default_value,
        lookup_table: dict,
        max_length: int = 30,
        batched: bool = False,
    ):
        super().__init__(
            sequence_column_name,
            feature_column_name,
            feature_default_value,
            max_length,
            batched,
        )

        self.lookup_table = lookup_table

    def batch_process(self, input_data, **kwargs):
        input_data[self.feature_column_name] = []

        for sequence in input_data[self.sequence_column_name]:
            feature = [
                self.lookup_table.get(aa, self.feature_default_value) for aa in sequence
            ]
            input_data[self.feature_column_name].append(
                self.pad_feature_to_seq_length(feature)
            )

        return input_data

    def single_process(self, input_data, **kwargs):
        input_data[self.feature_column_name] = [
            self.lookup_table.get(aa, self.feature_default_value)
            for aa in input_data[self.sequence_column_name]
        ]
        return input_data
