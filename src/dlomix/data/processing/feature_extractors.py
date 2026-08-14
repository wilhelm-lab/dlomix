"""
Feature extractors that compute per-residue/per-sequence features from parsed peptides.

Pass built-in feature names to a dataset via ``features_to_extract``; call
:func:`available_feature_extractors` to discover them. To add your own feature, either:

1. subclass :class:`FeatureExtractor`, or
2. write a function and pass it in ``features_to_extract`` (it is wrapped in a
   ``FunctionProcessor`` and mapped over the HuggingFace dataset).

In both cases you can read the parsed sequence information from each row via the keys
exposed in ``SequenceParsingProcessor.PARSED_COL_NAMES``:

- ``_parsed_sequence``: the parsed sequence (list of amino-acid + PTM tokens)
- ``_n_term_mods``: N-terminal modifications
- ``_c_term_mods``: C-terminal modifications

A custom function must return the row with the new feature column added, and must pad the
feature to the sequence length so all tensors share the sequence-length dimension.
"""

import re
import warnings
from operator import itemgetter

import numpy as np

from .feature_tables import (
    PTM_ATOM_COUNT_LOOKUP,
    PTM_GAIN_LOOKUP,
    PTM_LOSS_LOOKUP,
    PTM_MOD_DELTA_MASS_LOOKUP,
    PTM_RED_SMILES_LOOKUP,
)
from .processors import PeptideDatasetBaseProcessor, SequenceParsingProcessor


class MissingModifiedAminoAcidError(ValueError):
    """Raised when a modified amino acid has no entry in a feature lookup table."""


class MissingModifiedAminoAcidWarning(UserWarning):
    """Warned when a modified amino acid has no entry in a feature lookup table."""


def _is_modification_token(key) -> bool:
    """Whether a parsed-sequence token represents a modification.

    Every modified token produced by ``SequenceParsingProcessor`` — in-sequence
    residues ("M[UNIMOD:35]") and modified termini ("[UNIMOD:737]-") alike — contains
    the substring "UNIMOD". Plain amino acids ("A") and unmodified terminal sentinels
    ("[]-"/"-[]") don't, so they are never required to be in a lookup table.
    """
    return isinstance(key, str) and "UNIMOD" in key


_UNIMOD_TAG_PATTERN = re.compile(r"\[UNIMOD:\d+\]")


def _swapped_modification_order_key(key):
    """Return `key` with its two UNIMOD tags swapped, or None if not applicable.

    A doubly-modified token like "R[UNIMOD:643][UNIMOD:35]" is the same residue as
    "R[UNIMOD:35][UNIMOD:643]" — only the order the two tags happen to be written in
    differs — but a flat dict lookup only matches the literal string. Returns None for
    tokens with zero, one, three or more, or two identical tags, where a swap isn't
    applicable or wouldn't change anything.
    """
    if not isinstance(key, str):
        return None
    tags = _UNIMOD_TAG_PATTERN.findall(key)
    if len(tags) != 2 or tags[0] == tags[1]:
        return None
    before, middle, after = _UNIMOD_TAG_PATTERN.split(key)
    return f"{before}{tags[1]}{middle}{tags[0]}{after}"


class _LookupTable(dict):
    """
    Dict of lookup feature values that either raises or warns on a missing modified
    amino acid, depending on ``fail_on_missing_modified_amino_acid``.

    On a miss, first retries the token with its two UNIMOD tags swapped (see
    :func:`_swapped_modification_order_key`) — if that swapped form is a real entry,
    it's returned directly, no warning/error involved at all. Only a token still
    missing under both orderings raises or warns, per the flag; plain amino acids and
    unmodified termini (see :func:`_is_modification_token`) always fall back to
    ``default_value`` regardless of the flag. Both checks only run on a dict miss, so
    present keys are looked up at plain ``dict`` speed.
    """

    def __init__(
        self,
        lookup_table: dict,
        default_value,
        feature_column_name: str,
        fail_on_missing_modified_amino_acid: bool = False,
    ):
        super().__init__(lookup_table)
        self.default_value = default_value
        self.feature_column_name = feature_column_name
        self.fail_on_missing_modified_amino_acid = fail_on_missing_modified_amino_acid

    def __missing__(self, key):
        swapped_key = _swapped_modification_order_key(key)
        if swapped_key is not None and swapped_key in self:
            return self[swapped_key]

        if _is_modification_token(key):
            if self.fail_on_missing_modified_amino_acid:
                raise MissingModifiedAminoAcidError(
                    f"Modified amino acid '{key}' has no entry in the lookup table for "
                    f"feature '{self.feature_column_name}'. Please add it to the "
                    "lookup table. This is only enforced on the train/val split — the "
                    "eval/test split and inference already fall back to the default "
                    "value with a warning instead of raising an exception."
                )
            warnings.warn(
                f"Modified amino acid '{key}' has no entry in the lookup table "
                f"for feature '{self.feature_column_name}'; falling back to the "
                "default value. Consider enriching the lookup table with this "
                "modification.",
                MissingModifiedAminoAcidWarning,
            )
        return self.default_value


FEATURE_EXTRACTORS_PARAMETERS = {
    "mod_loss": {
        # True default is 0 (no atoms lost) for all SIX atoms; the lookup table stores
        # true 0-based deltas, and feature_value_offset shifts every value (looked up,
        # defaulted, or padded) by +1 so the tensor reaching the model is unchanged from
        # before this table was re-baselined to 0.
        "feature_default_value": [0] * 6,
        "feature_value_offset": 1,
        "lookup_table": PTM_LOSS_LOOKUP,
        "description": "Loss of atoms due to PTM.",
    },
    "delta_mass": {
        # Default value is 0 for the whole sequence
        "feature_default_value": 0,
        "lookup_table": PTM_MOD_DELTA_MASS_LOOKUP,
        "description": "Delta mass of PTM.",
    },
    "mod_gain": {
        # True default is 0 (no atoms gained); see the mod_loss comment above.
        "feature_default_value": [0] * 6,
        "feature_value_offset": 1,
        "lookup_table": PTM_GAIN_LOOKUP,
        "description": "Gain of atoms due to PTM.",
    },
    "atom_count": {
        # True default is 0; see the mod_loss comment above.
        "feature_default_value": [0] * 6,
        "feature_value_offset": 1,
        "lookup_table": PTM_ATOM_COUNT_LOOKUP,
        "description": "Atom count of PTM.",
    },
    "red_smiles": {
        # Default value is 0 for the whole PTM smiles representation (currently 60)
        "feature_default_value": [0] * 60,
        "lookup_table": PTM_RED_SMILES_LOOKUP,
        "description": "Reduced SMILES representation of PTM.",
    },
}

AVAILABLE_FEATURE_EXTRACTORS = list(FEATURE_EXTRACTORS_PARAMETERS.keys())


def available_feature_extractors() -> dict:
    """Return a ``{name: description}`` mapping of the built-in feature extractors.

    Use any of the returned names in a dataset's ``features_to_extract``. See this
    module's docstring for how to write a custom feature extractor.
    """
    return {
        name: params.get("description")
        for name, params in FEATURE_EXTRACTORS_PARAMETERS.items()
    }


class FeatureExtractor(PeptideDatasetBaseProcessor):
    """
    Base class for feature extractors.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    feature_column_name : str
        Name of the column to store the extracted feature.
    feature_default_value : int or list
        Default value to use for padding the feature.
    description : str
        Description of the feature.
    max_length : int (default=30)
        Maximum length of the feature.
    batched : bool (default=False)
        Whether to process data in batches.
    """

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
        self._feature_shape = np.array(self.feature_default_value).shape
        self.description = description
        self.max_length = max_length

    def pad_feature_to_seq_length(self, single_feature, unpadded_seq_len):
        """
        Pad the feature to the maximum sequence length.

        Parameters
        ----------
        single_feature : list
            List of feature values.
        unpadded_seq_len : int
            Length of the unpadded original sequence.

        Returns
        -------
        list
            Padded feature list.
        """

        if unpadded_seq_len > self.max_length:
            raise ValueError(
                f"Feature length ({unpadded_seq_len}) is longer than sequence length provided ({self.max_length})."
            )

        single_feature[unpadded_seq_len:] = self.feature_default_value

        # expand dims if needed
        single_feature = self._expand_dims(single_feature)
        return single_feature

    def _expand_dims(self, single_feature):
        if single_feature.ndim == 1:
            single_feature = np.expand_dims(single_feature, axis=-1)
        return single_feature


class LookupFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that uses a lookup table to extract features.

    Parameters
    ----------
    sequence_column_name : str
        Name of the column containing the peptide sequence.
    feature_column_name : str
        Name of the column to store the extracted feature.
    feature_default_value : int or list
        Default value to use for padding the feature.
    lookup_table : dict
        Lookup table for extracting features.
    description : str
        Description of the feature.
    max_length : int (default=30)
        Maximum length of the sequences.
    batched : bool (default=False)
        Whether to process data in batches.
    fail_on_missing_modified_amino_acid : bool (default=False)
        Whether to raise ``MissingModifiedAminoAcidError`` (True) or emit
        ``MissingModifiedAminoAcidWarning`` and fall back to ``feature_default_value``
        (False) when a modified amino acid (any token containing "UNIMOD", e.g.
        "M[UNIMOD:35]" or a modified terminus) is missing from the lookup table. Plain
        amino acids and unmodified termini ("[]-"/"-[]") always fall back to
        ``feature_default_value`` silently, regardless of this flag. When applied
        through :func:`~dlomix.data.processing.chain.build_processing_chain` (i.e. via a
        dataset's processing pipeline), a strict instance is automatically relaxed to a
        warning on the eval/test split — see ``apply_to_split``.
    feature_value_offset : float (default=0)
        Constant added to every value in the extracted feature array (looked up,
        defaulted, or padded alike) right before it is returned. Lets the lookup table
        and ``feature_default_value`` store true, human-readable values (e.g. 0 for "no
        atoms lost") while still producing whatever shifted values a downstream model
        was trained on.
    """

    def __init__(
        self,
        sequence_column_name: str,
        feature_column_name: str,
        feature_default_value,
        lookup_table: dict,
        description: str = "",
        max_length: int = 30,
        batched: bool = False,
        fail_on_missing_modified_amino_acid: bool = False,
        feature_value_offset: float = 0,
    ):
        super().__init__(
            sequence_column_name,
            feature_column_name,
            feature_default_value,
            description,
            max_length,
            batched,
        )

        self.fail_on_missing_modified_amino_acid = fail_on_missing_modified_amino_acid
        self.feature_value_offset = feature_value_offset

        self.lookup_table = _LookupTable(
            lookup_table,
            self.feature_default_value,
            feature_column_name,
            fail_on_missing_modified_amino_acid,
        )

        self.description = description

    def apply_to_split(self, dataset, split, ctx):
        """Relax a strict instance to a warning on non-fit (eval/test) splits.

        A modified amino acid missing from the lookup table on a fit split (train/val)
        still raises — that data is expected to be complete. On the eval/test split, a
        single incomplete entry shouldn't block scoring, so this rebuilds a lenient,
        warning instance for that split only; the pipeline's ``fit_splits`` come from
        ``ctx`` exactly like :class:`~dlomix.data.processing.processors.SequenceEncodingProcessor`.
        Instances that were already lenient (``fail_on_missing_modified_amino_acid=False``)
        behave identically on every split.

        A fresh instance is built through the constructor rather than mutating ``self``
        in place: ``PeptideDatasetBaseProcessor.__init__`` binds
        ``self._process_fn = self.batch_process``, a bound method tied to this specific
        object, so a shallow copy or an in-place attribute swap would leave that bound
        method pointing at stale state. Going through ``__init__`` sidesteps that
        entirely, at the cost of only one changed argument here.
        """
        if not self.fail_on_missing_modified_amino_acid or split in ctx.fit_splits:
            return super().apply_to_split(dataset, split, ctx)

        eval_processor = LookupFeatureExtractor(
            sequence_column_name=self.sequence_column_name,
            feature_column_name=self.feature_column_name,
            feature_default_value=self.feature_default_value,
            lookup_table=self.lookup_table,
            description=self.description,
            max_length=self.max_length,
            batched=self.batched,
            fail_on_missing_modified_amino_acid=False,
            feature_value_offset=self.feature_value_offset,
        )
        return dataset.map(
            eval_processor,
            desc=f"Mapping {eval_processor._map_description()} on split {split}",
            batched=self.batched,
            batch_size=ctx.batch_size,
            num_proc=ctx.num_proc,
        )

    def batch_process(self, input_data, **kwargs):
        feature_column = []

        for n_term, sequence, c_term in zip(
            input_data[SequenceParsingProcessor.PARSED_COL_NAMES["n_term"]],
            input_data[self.sequence_column_name],
            input_data[SequenceParsingProcessor.PARSED_COL_NAMES["c_term"]],
        ):
            feature = self._extract_feature([n_term] + sequence + [c_term])
            feature_column.append(feature)

        return {self.feature_column_name: feature_column}

    def single_process(self, input_data, **kwargs):
        seq_with_terms = (
            [input_data[SequenceParsingProcessor.PARSED_COL_NAMES["n_term"]]]
            + input_data[self.sequence_column_name]
            + [input_data[SequenceParsingProcessor.PARSED_COL_NAMES["c_term"]]]
        )
        feature = self._extract_feature(seq_with_terms)
        return {self.feature_column_name: feature}

    def _extract_feature(self, sequence):
        # we lookup unttil the max length only because some sequences in train/val can be longer
        lookup_length = min(self.max_length, len(sequence))
        sequence_for_lookup = sequence[:lookup_length]

        feature = np.empty((self.max_length, *self._feature_shape), dtype=np.float32)

        feature[:lookup_length] = itemgetter(*sequence_for_lookup)(self.lookup_table)

        # pad from lookup_length to max_length if needed and expand dims if one-dimensional
        # technically, complementing the previous step with a call feature[lookup_length:] = self.feature_default_value
        feature = self.pad_feature_to_seq_length(feature, lookup_length)

        # shift every value (looked up, defaulted, or padded) by the configured offset;
        # unconditional so there is no extra branch on the hot path (adding 0 is a no-op)
        feature += self.feature_value_offset

        return feature
