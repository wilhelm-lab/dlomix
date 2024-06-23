import json

from .feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    FEATURE_EXTRACTORS_PARAMETERS,
    FeatureExtractor,
    LookupFeatureExtractor,
)
from .processors import FunctionProcessor, SequenceParsingProcessor

__all__ = [
    "AVAILABLE_FEATURE_EXTRACTORS",
    "LookupFeatureExtractor",
    "FeatureExtractor",
    "FunctionProcessor",
]

d = dict(
    zip(
        AVAILABLE_FEATURE_EXTRACTORS,
        [
            FEATURE_EXTRACTORS_PARAMETERS.get(f, {}).get("description")
            for f in AVAILABLE_FEATURE_EXTRACTORS
        ],
    )
)

print(
    f"""
Avaliable feature extractors are (use the key of the following dict and pass it to features_to_extract in the Dataset Class):
{json.dumps(d, indent=3, sort_keys=True)}.
When writing your own feature extractor, you can either
    (1) use the FeatureExtractor class or
    (2) write a function that can be mapped to the Hugging Face dataset.
In both cases, you can access the parsed sequence information from the dataset using the following keys, which all provide python lists:
    - {SequenceParsingProcessor.PARSED_COL_NAMES["seq"]}: parsed sequence
    - {SequenceParsingProcessor.PARSED_COL_NAMES["n_term"]}: N-terminal modifications
    - {SequenceParsingProcessor.PARSED_COL_NAMES["c_term"]}: C-terminal modifications
"""
)
