"""
Reusable, serializable preprocessing for inference on new peptide data.

``PeptidePreprocessor`` captures the exact preprocessing recipe used to build a
training dataset (learned alphabet, encoding scheme, padding, feature extractors) and
applies it to raw inputs (strings, lists, numpy arrays, dicts, HF datasets, DataFrames),
producing the same model-ready tensors a :class:`~dlomix.data.dataset.PeptideDataset`
would. It can be created from a live dataset, from a saved dataset directory, or loaded
from its own lightweight artifact for shipping alongside a model.
"""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset

from ..config import _BACKEND, PYTORCH_BACKEND
from .dataset_utils import EncodingScheme
from .processing.chain import build_processing_chain
from .processing.processors import SequencePaddingProcessor
from .tensor_conversion import (
    cast_feature_columns_to_float,
    to_tf_tensor_dataset,
    to_torch_dataloader,
)

ARTIFACT_NAME = "dlomix_preprocessor.json"


class PeptidePreprocessor:
    """Turn raw peptide inputs into model-ready tensors using a fixed recipe.

    Parameters
    ----------
    alphabet : dict
        Learned alphabet (token -> integer index), including padding and unknown tokens.
    sequence_column : str
        Name of the column holding peptide sequences.
    max_seq_len : int
        Maximum (unmodified) sequence length used during training.
    padding_value : str
        Padding token; must exist in ``alphabet``. Default '-'.
    encoding_scheme : str or EncodingScheme
        'unmod' or 'naive-mods'. Default 'unmod'.
    with_termini : bool
        Whether N/C termini were added to sequences. Default True.
    model_features : list of str, optional
        Feature columns carried into the model as tensors (must be present in inputs).
    features_to_extract : list, optional
        Built-in feature names (str) or custom callables, reproducing training features.
    dataset_type : str
        'tf' or 'pt'; selects the output tensor format. Defaults to None, which resolves
        to 'pt' or 'tf' based on the active DLOMIX_BACKEND.
    batch_size : int
        Batch size for the produced tensor dataset. Default 64.
    batch_processing_size : int
        Batch size for the internal ``.map()`` processing calls. Default 1000.
    """

    def __init__(
        self,
        alphabet: Dict[str, int],
        sequence_column: str,
        max_seq_len: int,
        padding_value: str = "-",
        encoding_scheme: Union[str, EncodingScheme] = EncodingScheme.UNMOD,
        with_termini: bool = True,
        model_features: Optional[List[str]] = None,
        features_to_extract: Optional[List[Union[str, Callable]]] = None,
        dataset_type: Optional[str] = None,
        batch_size: int = 64,
        batch_processing_size: int = 1000,
    ):
        if dataset_type is None:
            dataset_type = "pt" if _BACKEND in PYTORCH_BACKEND else "tf"

        self.alphabet = dict(alphabet)
        self.sequence_column = sequence_column
        self.max_seq_len = max_seq_len
        self.padding_value = padding_value
        self.encoding_scheme = EncodingScheme(encoding_scheme)
        self.with_termini = with_termini
        self.model_features = list(model_features) if model_features else []
        self.features_to_extract = (
            list(features_to_extract) if features_to_extract else []
        )
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.batch_processing_size = batch_processing_size

        if self.padding_value not in self.alphabet:
            raise ValueError(
                f"padding_value '{self.padding_value}' is not present in the alphabet."
            )

        self.vocab_size = len(self.alphabet)
        self._build_pipeline()

    # ------------------------------------------------------------------ builders

    def _build_pipeline(self) -> None:
        """Build the inference processor chain (frozen vocabulary, fallback encoding).

        This is the same chain the training pipeline applies to its test split, built
        via the shared :func:`~dlomix.data.processing.chain.build_processing_chain`.
        """
        self._processors, self._extracted_feature_names = build_processing_chain(
            sequence_column=self.sequence_column,
            encoding_scheme=self.encoding_scheme,
            with_termini=self.with_termini,
            max_seq_len=self.max_seq_len,
            padding_value=self.padding_value,
            alphabet=self.alphabet,
            pad=True,
            features_to_extract=self.features_to_extract,
            encoding_extend_alphabet=False,
            encoding_fallback_unmodified=True,
            warn_unmod=False,
        )

    @property
    def input_columns(self) -> List[str]:
        """Columns fed to the model as tensors (sequence + features), order stable."""
        return [
            self.sequence_column,
            *self.model_features,
            *self._extracted_feature_names,
        ]

    @property
    def extracted_feature_names(self) -> List[str]:
        """Names of features computed from the sequence (e.g. PTM features)."""
        return list(self._extracted_feature_names)

    @property
    def fingerprint(self) -> str:
        """Short stable hash of the recipe, used for model/preprocessor consistency."""
        payload = json.dumps(
            {
                "alphabet": dict(sorted(self.alphabet.items())),
                "sequence_column": self.sequence_column,
                "max_seq_len": self.max_seq_len,
                "padding_value": self.padding_value,
                "encoding_scheme": self.encoding_scheme.value,
                "with_termini": self.with_termini,
                "model_features": sorted(self.model_features),
                "features": sorted(self._extracted_feature_names),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # --------------------------------------------------------------- constructors

    @classmethod
    def from_dataset(cls, dataset) -> "PeptidePreprocessor":
        """Build a preprocessor from a live, processed ``PeptideDataset``."""
        if not getattr(dataset, "processed", False):
            raise ValueError(
                "Dataset must be processed before a preprocessor can be derived from it."
            )
        return cls(
            alphabet=dataset.extended_alphabet,
            sequence_column=dataset.sequence_column,
            max_seq_len=dataset.max_seq_len,
            padding_value=dataset.padding_value,
            encoding_scheme=dataset.encoding_scheme,
            with_termini=dataset.with_termini,
            model_features=dataset.model_features,
            features_to_extract=dataset.features_to_extract,
            dataset_type=dataset.dataset_type,
            batch_size=dataset.batch_size,
            batch_processing_size=getattr(dataset, "batch_processing_size", 1000),
        )

    @classmethod
    def from_saved(cls, path: str) -> "PeptidePreprocessor":
        """Build a preprocessor from a ``PeptideDataset.save_to_disk`` directory.

        Reads the config + metadata only; the HF data on disk is not loaded.
        """
        from .dataset import PeptideDataset  # local import to avoid a cycle

        path_obj = Path(path)
        config = json.loads(
            (path_obj / PeptideDataset.CONFIG_JSON_NAME).read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (path_obj / PeptideDataset.METADATA_JSON_NAME).read_text(encoding="utf-8")
        )
        state = metadata.get("state", {})

        alphabet = state.get("extended_alphabet")
        if not alphabet:
            raise ValueError(
                f"No learned alphabet found in saved dataset metadata at {path}."
            )

        # Prefer the saved extracted column names (they're the ground truth for what was
        # actually applied at training time). Fall back to config only if the key is absent
        # (e.g. datasets saved before this attribute existed). An explicit empty list means
        # no features were extracted — don't fall through to the config in that case.
        extracted = state.get("_extracted_features_columns")
        features_to_extract = (
            extracted
            if extracted is not None
            else _string_features(config.get("features_to_extract"))
        )

        return cls(
            alphabet=alphabet,
            sequence_column=config["sequence_column"],
            max_seq_len=config["max_seq_len"],
            padding_value=config["padding_value"],
            encoding_scheme=config["encoding_scheme"],
            with_termini=config["with_termini"],
            model_features=config.get("model_features"),
            features_to_extract=features_to_extract,
            dataset_type=config["dataset_type"],
            batch_size=config.get("batch_size", 64),
            batch_processing_size=config.get("batch_processing_size", 1000),
        )

    # --------------------------------------------------------------- persistence

    def save(self, path: str) -> str:
        """Save the recipe to a lightweight JSON artifact and return the file path."""
        callable_features = [
            f for f in self.features_to_extract if not isinstance(f, str)
        ]
        if callable_features:
            warnings.warn(
                "Custom callable feature extractors cannot be serialized and will be "
                f"dropped on save: {[f.__name__ for f in callable_features]}. "
                "Re-create the preprocessor with from_dataset() to retain them."
            )

        path_obj = Path(path)
        if path_obj.suffix != ".json":
            path_obj.mkdir(parents=True, exist_ok=True)
            path_obj = path_obj / ARTIFACT_NAME

        artifact = {
            "alphabet": self.alphabet,
            "sequence_column": self.sequence_column,
            "max_seq_len": self.max_seq_len,
            "padding_value": self.padding_value,
            "encoding_scheme": self.encoding_scheme.value,
            "with_termini": self.with_termini,
            "model_features": self.model_features,
            "features_to_extract": [
                f for f in self.features_to_extract if isinstance(f, str)
            ],
            "dataset_type": self.dataset_type,
            "batch_size": self.batch_size,
            "batch_processing_size": self.batch_processing_size,
            "fingerprint": self.fingerprint,
        }
        path_obj.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return str(path_obj)

    @classmethod
    def load(cls, path: str) -> "PeptidePreprocessor":
        """Load a preprocessor from a JSON artifact (file or containing directory)."""
        path_obj = Path(path)
        if path_obj.is_dir():
            path_obj = path_obj / ARTIFACT_NAME
        artifact = json.loads(path_obj.read_text(encoding="utf-8"))
        artifact.pop("fingerprint", None)
        return cls(**artifact)

    # ------------------------------------------------------------------ transform

    def transform(self, inputs):
        """Convert raw inputs into a backend tensor dataset ready for the model."""
        hf_dataset = self._to_hf_dataset(inputs)
        self._validate_columns(hf_dataset)

        for processor in self._processors:
            hf_dataset = hf_dataset.map(
                processor,
                batched=processor.batched,
                batch_size=self.batch_processing_size,
                num_proc=None,
            )

        # Padding adds a bookkeeping column; keep all rows (mirrors test split).
        if SequencePaddingProcessor.KEEP_COLUMN_NAME in hf_dataset.column_names:
            hf_dataset = hf_dataset.remove_columns(
                SequencePaddingProcessor.KEEP_COLUMN_NAME
            )

        hf_dataset = hf_dataset.select_columns(self.input_columns)

        feature_columns = [*self.model_features, *self._extracted_feature_names]
        hf_dataset = cast_feature_columns_to_float(
            hf_dataset, feature_columns, batch_size=self.batch_processing_size
        )

        if self.dataset_type == "pt":
            return to_torch_dataloader(
                hf_dataset, self.input_columns, batch_size=self.batch_size
            )
        return to_tf_tensor_dataset(
            hf_dataset, self.input_columns, batch_size=self.batch_size
        )

    def __call__(self, inputs):
        return self.transform(inputs)

    # -------------------------------------------------------------------- helpers

    def _to_hf_dataset(self, inputs) -> Dataset:
        if isinstance(inputs, Dataset):
            return inputs
        if isinstance(inputs, pd.DataFrame):
            return Dataset.from_pandas(inputs, preserve_index=False)
        if isinstance(inputs, str):
            return Dataset.from_dict({self.sequence_column: [inputs]})
        if isinstance(inputs, np.ndarray):
            return Dataset.from_dict({self.sequence_column: inputs.tolist()})
        if isinstance(inputs, dict):
            return Dataset.from_dict(inputs)
        if isinstance(inputs, (list, tuple)):
            return Dataset.from_dict({self.sequence_column: list(inputs)})
        raise ValueError(
            "Unsupported input type for preprocessing: "
            f"{type(inputs)}. Provide a string, list/array of strings, dict, "
            "pandas DataFrame, or a HuggingFace Dataset."
        )

    def _validate_columns(self, hf_dataset: Dataset) -> None:
        if self.sequence_column not in hf_dataset.column_names:
            raise ValueError(
                f"Sequence column '{self.sequence_column}' not found in inputs. "
                f"Available columns: {hf_dataset.column_names}"
            )
        missing = [f for f in self.model_features if f not in hf_dataset.column_names]
        if missing:
            raise ValueError(
                f"Model feature column(s) {missing} required by this preprocessor are "
                f"missing from the inputs. Available columns: {hf_dataset.column_names}"
            )


def _string_features(features) -> List[Union[str, Callable]]:
    """Keep only string feature names from a possibly-mixed list (drop reprs)."""
    if not features:
        return []
    return [f for f in features if isinstance(f, str)]
