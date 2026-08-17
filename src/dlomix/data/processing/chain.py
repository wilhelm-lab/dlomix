"""
Single source of truth for the ordered sequence-processing chain.

Both the training-time :class:`~dlomix.data.processing.pipeline.ProcessingPipeline` and
the inference-time :class:`~dlomix.data.inference.PeptidePreprocessor` build the same
chain — parse → (PTM removal) → encode → (pad) → feature extractors — and differ only in
how the encoding step is configured (learn-the-vocab vs frozen-vocab-with-fallback). This
builder centralizes that construction so the two cannot drift apart.
"""

import warnings
from typing import Callable, List, Optional, Tuple, Union

from ..dataset_utils import EncodingScheme
from .feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    FEATURE_EXTRACTORS_PARAMETERS,
    LookupFeatureExtractor,
)
from .processors import (
    FunctionProcessor,
    PeptideDatasetBaseProcessor,
    SequenceEncodingProcessor,
    SequencePaddingProcessor,
    SequenceParsingProcessor,
    SequencePTMRemovalProcessor,
)


def build_processing_chain(
    *,
    sequence_column: str,
    encoding_scheme: EncodingScheme,
    with_termini: bool,
    max_seq_len: int,
    padding_value: str,
    alphabet: dict,
    pad: bool = True,
    features_to_extract: Optional[List[Union[str, Callable]]] = None,
    encoding_extend_alphabet: bool = False,
    encoding_fallback_unmodified: bool = False,
    warn_unmod: bool = True,
    fail_on_missing_modified_amino_acid: bool = True,
) -> Tuple[List[PeptideDatasetBaseProcessor], List[str]]:
    """Build the ordered processor chain and the names of extracted feature columns.

    The encoding behavior is selected by the caller:
    - training: ``encoding_extend_alphabet=learning_mode`` (per-split learn/fallback is
      then handled by the encoder's ``apply_to_split``);
    - inference: ``encoding_extend_alphabet=False, encoding_fallback_unmodified=True``
      (frozen vocabulary, unseen tokens fall back to the unmodified amino acid).

    ``fail_on_missing_modified_amino_acid`` mirrors that same training/inference split:
    strict (default, ``True``, raises) while curating a lookup table against known
    training data; ``False`` warns and falls back to the default value instead. For the
    training pipeline this is automatically relaxed to ``False`` on the test split by
    ``LookupFeatureExtractor.apply_to_split`` — no extra wiring needed here.
    ``PeptidePreprocessor`` (real inference) instead has no split concept and applies
    these processors directly, so it passes ``fail_on_missing_modified_amino_acid=False``
    itself to get the same "warn, don't crash scoring" behavior.

    Returns ``(processors, extracted_feature_names)``.
    """
    max_length = max_seq_len + 2 if with_termini else max_seq_len

    processors: List[PeptideDatasetBaseProcessor] = [
        SequenceParsingProcessor(
            sequence_column, batched=True, with_termini=with_termini
        )
    ]

    if encoding_scheme == EncodingScheme.UNMOD:
        if warn_unmod:
            warnings.warn(
                f"Encoding scheme is {encoding_scheme}, this enforces removing all "
                "occurences of PTMs in the sequences. If you prefer to encode the "
                "(amino-acids)+PTM combinations as tokens in the vocabulary, please use "
                "the encoding scheme 'naive-mods'."
            )
        processors.append(
            SequencePTMRemovalProcessor(
                sequence_column_name=sequence_column, batched=True
            )
        )
    elif encoding_scheme != EncodingScheme.NAIVE_MODS:
        raise NotImplementedError(
            f"Encoding scheme {encoding_scheme} is not implemented. Available "
            f"encoding schemes are: {list(EncodingScheme.__members__)}."
        )

    processors.append(
        SequenceEncodingProcessor(
            sequence_column_name=sequence_column,
            alphabet=alphabet,
            batched=True,
            extend_alphabet=encoding_extend_alphabet,
            fallback_unmodified=encoding_fallback_unmodified,
        )
    )

    if pad:
        processors.append(
            SequencePaddingProcessor(
                sequence_column_name=sequence_column,
                batched=True,
                padding_index=alphabet[padding_value],
                max_length=max_length,
            )
        )
    else:
        warnings.warn(
            "Padding is turned off, sequences will have variable lengths. Converting "
            "this dataset to tensors will cause errors unless proper stacking of "
            "examples is done."
        )

    feature_processors, extracted_feature_names = _build_feature_extractors(
        features_to_extract,
        max_length,
        fail_on_missing_modified_amino_acid,
    )
    processors.extend(feature_processors)

    return processors, extracted_feature_names


def _build_feature_extractors(
    features_to_extract,
    max_length,
    fail_on_missing_modified_amino_acid: bool = True,
) -> Tuple[List[PeptideDatasetBaseProcessor], List[str]]:
    processors: List[PeptideDatasetBaseProcessor] = []
    names: List[str] = []
    if not features_to_extract:
        return processors, names

    for feature in features_to_extract:
        if isinstance(feature, str):
            feature_name = feature.lower()
            if feature_name not in AVAILABLE_FEATURE_EXTRACTORS:
                warnings.warn(
                    f"Skipping feature extractor {feature} since it is not available. "
                    f"Please choose from: {AVAILABLE_FEATURE_EXTRACTORS}."
                )
                continue
            processors.append(
                LookupFeatureExtractor(
                    sequence_column_name=SequenceParsingProcessor.PARSED_COL_NAMES[
                        "seq"
                    ],
                    feature_column_name=feature_name,
                    **FEATURE_EXTRACTORS_PARAMETERS[feature_name],
                    max_length=max_length,
                    batched=True,
                    # provided features are expected to be complete, so we fail if a modified amino acid is missing from the lookup table
                    # this is critical to give the user a chance to fix the lookup table, otherwise the model will be trained with missing features and will not learn the correct representation
                    # (relaxed to a warning on the eval/test split by LookupFeatureExtractor.apply_to_split;
                    # disabled by the caller entirely for real inference, since unseen data should warn and fall back rather than crash)
                    fail_on_missing_modified_amino_acid=fail_on_missing_modified_amino_acid,
                )
            )
        elif callable(feature):
            warnings.warn(
                f"Using custom feature extractor from the user function "
                f"{feature.__name__}; please ensure it pads the feature to the "
                "sequence length so all tensors share the sequence-length dimension."
            )
            feature_name = feature.__name__
            processors.append(FunctionProcessor(feature))
        else:
            raise ValueError(
                f"Feature extractor {feature} is not a valid type. Provide a function "
                "or a string naming a valid feature extractor."
            )
        names.append(feature_name)

    return processors, names
