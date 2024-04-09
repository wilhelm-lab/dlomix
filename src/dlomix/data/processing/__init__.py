import json

from .feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    FEATURE_EXTRACTORS_PARAMETERS,
    FeatureExtractor,
    LookupFeatureExtractor,
)
from .processors import FunctionProcessor

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
{json.dumps(d, indent=3, sort_keys=True)}
"""
)
