"""
Bundle a trained model with its preprocessor for one-call inference (HF-style).

``InferencePipeline`` ties a model together with the
:class:`~dlomix.data.inference.PeptidePreprocessor` that reproduces its training-time
preprocessing, so users can ``predict`` directly on raw sequences and ``save``/``load``
both as a single artifact. Backend (TensorFlow/PyTorch) is selected at import time via
``DLOMIX_BACKEND`` (see :mod:`dlomix.config`).
"""

import json
import warnings
from pathlib import Path

import numpy as np

from ..config import _BACKEND, PYTORCH_BACKEND
from ..data.inference import PeptidePreprocessor

_IS_TORCH = _BACKEND in PYTORCH_BACKEND

PIPELINE_META_NAME = "dlomix_inference_pipeline.json"
TF_MODEL_FILE = "model.keras"
TORCH_MODEL_FILE = "model.pt"


def _model_vocab_size(model):
    """Best-effort read of a model's embedding vocab size (None if unavailable)."""
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return None
    if hasattr(embedding, "input_dim"):  # tf.keras.layers.Embedding
        return embedding.input_dim
    if hasattr(embedding, "num_embeddings"):  # torch.nn.Embedding
        return embedding.num_embeddings
    return None


def _model_expected_seq_len(model):
    """Best-effort read of a model's expected input sequence length (None if unavailable)."""
    raw_seq_length = getattr(model, "raw_seq_length", None)
    if raw_seq_length is None:
        return None
    with_termini = getattr(model, "with_termini", False)
    return raw_seq_length + 2 if with_termini else raw_seq_length


class InferencePipeline:
    """A model + its preprocessor, with a single ``predict`` and combined save/load.

    Parameters
    ----------
    model :
        A trained dlomix model (TensorFlow or PyTorch, matching the active backend).
    preprocessor : PeptidePreprocessor
        The preprocessor reproducing the model's training-time preprocessing.
    """

    def __init__(self, model, preprocessor: PeptidePreprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self._check_model_consistency()

    @classmethod
    def from_model_and_dataset(cls, model, dataset) -> "InferencePipeline":
        """Bundle a model with the preprocessor derived from a processed dataset."""
        return cls(model, dataset.get_preprocessor())

    def _check_model_consistency(self) -> None:
        model_vocab = _model_vocab_size(self.model)
        if model_vocab is not None and model_vocab != self.preprocessor.vocab_size:
            raise ValueError(
                f"Model/preprocessor mismatch: the model's embedding vocabulary size is "
                f"{model_vocab}, but the preprocessor alphabet has "
                f"{self.preprocessor.vocab_size} tokens. They were likely not trained "
                f"together."
            )

        expected_seq_len = _model_expected_seq_len(self.model)
        if (
            expected_seq_len is not None
            and expected_seq_len != self.preprocessor.max_seq_len
        ):
            raise ValueError(
                f"Model/preprocessor mismatch: the model expects sequences of length "
                f"{expected_seq_len}, but the preprocessor pads sequences to "
                f"max_seq_len={self.preprocessor.max_seq_len}. They were likely not "
                f"trained together."
            )

    def predict(self, inputs, **predict_kwargs) -> np.ndarray:
        """Preprocess raw inputs and run the model, returning a numpy array."""
        tensors = self.preprocessor(inputs)

        if _IS_TORCH:
            return self._predict_torch(tensors)
        return np.asarray(self.model.predict(tensors, **predict_kwargs))

    def _predict_torch(self, loader) -> np.ndarray:
        import torch

        single_input = len(self.preprocessor.input_columns) == 1
        seq_column = self.preprocessor.input_columns[0]

        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch in loader:
                model_input = batch[seq_column] if single_input else batch
                out = self.model(model_input)
                outputs.append(out.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def save(self, path: str, overwrite: bool = False) -> str:
        """Save the preprocessor + model + pipeline metadata to a directory."""
        path_obj = Path(path)
        if path_obj.exists() and any(path_obj.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Directory {path} already exists and is not empty. "
                f"Set overwrite=True to replace it."
            )
        path_obj.mkdir(parents=True, exist_ok=True)

        self.preprocessor.save(str(path_obj))

        if _IS_TORCH:
            import torch

            # full-object pickle; loading executes code. TODO: state_dict + safetensors
            # (with stored init-config) for safer sharing of public Hub repos.
            torch.save(self.model, str(path_obj / TORCH_MODEL_FILE))
        else:
            # .keras format reloads registered models as their original class
            self.model.save(str(path_obj / TF_MODEL_FILE))

        meta = {
            "backend": "pytorch" if _IS_TORCH else "tensorflow",
            "model_class": type(self.model).__name__,
            "fingerprint": self.preprocessor.fingerprint,
            "vocab_size": self.preprocessor.vocab_size,
        }
        (path_obj / PIPELINE_META_NAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return str(path_obj)

    @classmethod
    def load(cls, path: str) -> "InferencePipeline":
        """Load a bundled model + preprocessor saved with :meth:`save`."""
        path_obj = Path(path)
        meta = json.loads((path_obj / PIPELINE_META_NAME).read_text(encoding="utf-8"))

        saved_backend = meta.get("backend")
        current_backend = "pytorch" if _IS_TORCH else "tensorflow"
        if saved_backend and saved_backend != current_backend:
            raise ValueError(
                f"Pipeline was saved with the '{saved_backend}' backend but the active "
                f"backend is '{current_backend}'. Set DLOMIX_BACKEND={saved_backend} "
                f"before importing dlomix."
            )

        preprocessor = PeptidePreprocessor.load(str(path_obj))

        if meta.get("fingerprint") and meta["fingerprint"] != preprocessor.fingerprint:
            warnings.warn(
                "Loaded preprocessor fingerprint does not match the one recorded when "
                "the pipeline was saved; preprocessing may differ from training."
            )

        if _IS_TORCH:
            import torch

            model = torch.load(str(path_obj / TORCH_MODEL_FILE), weights_only=False)
        else:
            import tensorflow as tf

            model = tf.keras.models.load_model(str(path_obj / TF_MODEL_FILE))

        return cls(model, preprocessor)

    # ----------------------------------------------------------- HuggingFace Hub

    def push_to_hub(
        self,
        repo_id: str,
        token: str = None,
        private: bool = False,
        commit_message: str = "Upload dlomix inference pipeline",
    ) -> str:
        """Push the bundled model + preprocessor to a HuggingFace Hub repo.

        The repo holds the same files as :meth:`save` (preprocessor, model, metadata)
        plus an auto-generated model card. Returns the repo id.
        """
        import tempfile

        from huggingface_hub import HfApi, create_repo

        create_repo(repo_id, token=token, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save(tmp, overwrite=True)
            self._write_model_card(tmp, repo_id)
            HfApi(token=token).upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                commit_message=commit_message,
                # mirror the repo to the bundle: drop any stale files from prior pushes
                delete_patterns="*",
            )
        return repo_id

    @classmethod
    def from_pretrained(
        cls, repo_id: str, token: str = None, revision: str = None, **snapshot_kwargs
    ) -> "InferencePipeline":
        """Load a pipeline previously pushed with :meth:`push_to_hub`."""
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=repo_id, token=token, revision=revision, **snapshot_kwargs
        )
        return cls.load(local_dir)

    def _write_model_card(self, directory: str, repo_id: str) -> None:
        backend = "pytorch" if _IS_TORCH else "tensorflow"
        prep = self.preprocessor
        predict_block = self._usage_predict_block(prep)

        notes = []
        if prep.model_features:
            cols = ", ".join(f"`{f}`" for f in prep.model_features)
            notes.append(
                f"This model requires the feature column(s) {cols} to be provided "
                f"alongside the sequences at inference time."
            )
        if prep.extracted_feature_names:
            cols = ", ".join(f"`{f}`" for f in prep.extracted_feature_names)
            notes.append(
                f"The feature(s) {cols} are computed from the sequence automatically — "
                f"you do not supply them."
            )
        features_note = ("\n\n" + "\n\n".join(notes)) if notes else ""

        card = f"""---
tags:
- dlomix
- proteomics
library_name: dlomix
---

# {repo_id}

A [DLOmix](https://github.com/wilhelm-lab/dlomix) inference pipeline bundling a
`{type(self.model).__name__}` model with its `PeptidePreprocessor`.

- Backend: **{backend}**
- Vocabulary size: {prep.vocab_size}
- Preprocessor fingerprint: `{prep.fingerprint}`{features_note}

## Usage

```python
from dlomix.pipelines import InferencePipeline

pipeline = InferencePipeline.from_pretrained("{repo_id}")
predictions = {predict_block}
```

> Load under the same `DLOMIX_BACKEND` ({backend}) used when the pipeline was saved.
"""
        (Path(directory) / "README.md").write_text(card, encoding="utf-8")

    @staticmethod
    def _usage_predict_block(prep) -> str:
        """Build a predict() example matching the model's required inputs."""
        example_seqs = '["PEPTIDEK", "ACDEM[UNIMOD:35]K"]'
        if not prep.model_features:
            # sequence-only model: a plain list of sequences is enough
            return f"pipeline.predict({example_seqs})"

        # metadata model: inputs must be a dict of sequence + feature columns
        lines = [f'    "{prep.sequence_column}": {example_seqs},']
        for feat in prep.model_features:
            lines.append(f'    "{feat}": [...],  # one entry per sequence')
        return "pipeline.predict({\n" + "\n".join(lines) + "\n})"
