from .feature_tables import (
    PTM_ATOM_COUNT_LOOKUP,
    PTM_GAIN_LOOKUP,
    PTM_LOSS_LOOKUP,
    PTM_MOD_DELTA_MASS_LOOKUP,
)
from .processors import PeptideDatasetBaseProcessor

FEATURE_EXTRACTORS_PARAMETERS = {
    "mod_loss": {
        "feature_default_value": [1, 1, 1, 1, 1, 1],
        "lookup_table": PTM_LOSS_LOOKUP,
        "description": "Loss of atoms due to PTM.",
    },
    "delta_mass": {
        "feature_default_value": 0,
        "lookup_table": PTM_MOD_DELTA_MASS_LOOKUP,
        "description": "Delta mass of PTM.",
    },
    "mod_gain": {
        "feature_default_value": [1, 1, 1, 1, 1, 1],
        "lookup_table": PTM_GAIN_LOOKUP,
        "description": "Gain of atoms due to PTM.",
    },
    "atom_count": {
        "feature_default_value": [1, 1, 1, 1, 1, 1],
        "lookup_table": PTM_ATOM_COUNT_LOOKUP,
        "description": "Atom count of PTM.",
    },
}

AVAILABLE_FEATURE_EXTRACTORS = list(FEATURE_EXTRACTORS_PARAMETERS.keys())


class FeatureExtractor(PeptideDatasetBaseProcessor):
    def __init__(
        self,
        sequence_column_name: str,
        feature_column_name: str,
        feature_default_value,
        description: str,
        max_length: int = 30,
        batched: bool = False,
    ):
        super().__init__(sequence_column_name, batched)
        self.feature_column_name = feature_column_name
        self.feature_default_value = feature_default_value
        self.description = description
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
        description: str = "",
        max_length: int = 30,
        batched: bool = False,
    ):
        super().__init__(
            sequence_column_name,
            feature_column_name,
            feature_default_value,
            description,
            max_length,
            batched,
        )

        self.lookup_table = lookup_table
        self.description = description

    def batch_process(self, input_data, **kwargs):
        feature_column = []

        for sequence in input_data[self.sequence_column_name]:
            feature = self._extract_feature(sequence)
            feature_column.append(feature)

        return {self.feature_column_name: feature_column}

    def single_process(self, input_data, **kwargs):
        feature = self._extract_feature(input_data[self.sequence_column_name])
        return {self.feature_column_name: feature}

    def _extract_feature(self, sequence):
        feature = [
            self.lookup_table.get(aa, self.feature_default_value) for aa in sequence
        ]
        feature = self.pad_feature_to_seq_length(feature)
        return feature
