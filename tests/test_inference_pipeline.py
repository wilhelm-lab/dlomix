"""Tests for InferencePipeline (model + preprocessor bundling)."""

import warnings
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from dlomix.config import _BACKEND, PYTORCH_BACKEND
from dlomix.data import PeptidePreprocessor, RetentionTimeDataset
from dlomix.models import PrositRetentionTimePredictor
from dlomix.pipelines import InferencePipeline

DATASET_TYPE = "pt" if _BACKEND in PYTORCH_BACKEND else "tf"

_CARD_ALPHABET = {
    "-": 0,
    "X": 1,
    "A": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "K": 6,
    "P": 7,
    "M": 8,
}


class _FakeModel:
    """Stand-in model with no embedding (consistency check is skipped)."""


class _FakeSeqLenModel:
    """Stand-in model exposing only raw_seq_length/with_termini."""

    def __init__(self, raw_seq_length, with_termini):
        self.raw_seq_length = raw_seq_length
        self.with_termini = with_termini


def _make_card(tmp_path, **prep_kwargs) -> str:
    prep = PeptidePreprocessor(
        alphabet=_CARD_ALPHABET,
        sequence_column="modified_sequence",
        max_seq_len=30,
        dataset_type=DATASET_TYPE,
        **prep_kwargs,
    )
    pipe = InferencePipeline(_FakeModel(), prep)
    pipe._write_model_card(str(tmp_path), "omsh/test-model")
    return (tmp_path / "README.md").read_text()


RAW_SEQUENCES = [
    "ACDEFGHIK",
    "PEPTIDEK",
    "MKLVAAR",
    "GGGGSSSK",
    "ACDEFGHIKLMN",
    "PEPK",
    "MMMKL",
    "AAACCCDDD",
    "KKLLMMNN",
    "PPQQRRSS",
]


@pytest.fixture
def rt_dataset():
    seqs = RAW_SEQUENCES * 4
    data = {
        "modified_sequence": seqs,
        "indexed_retention_time": [float(len(s)) for s in seqs],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RetentionTimeDataset(
            data_source=Dataset.from_dict(data),
            data_format="hf",
            sequence_column="modified_sequence",
            label_column="indexed_retention_time",
            val_ratio=0.2,
            max_seq_len=20,
            batch_size=8,
            dataset_type=DATASET_TYPE,
        )


@pytest.fixture
def rt_model(rt_dataset):
    model = PrositRetentionTimePredictor(
        seq_length=22, alphabet=rt_dataset.extended_alphabet
    )
    # build/initialize parameters on real tensors
    if DATASET_TYPE == "pt":
        import torch

        batch = next(iter(rt_dataset.tensor_train_data))
        with torch.no_grad():
            model(batch["modified_sequence"])
    else:
        model.predict(rt_dataset.tensor_train_data, verbose=0)
    return model


def test_predict_on_raw_sequences(rt_dataset, rt_model):
    pipe = InferencePipeline.from_model_and_dataset(rt_model, rt_dataset)
    preds = pipe.predict(["ACDEFGHIK", "PEPTIDEK"])
    assert preds.shape[0] == 2
    assert np.isfinite(preds).all()


def test_consistency_check_rejects_mismatched_model(rt_dataset):
    # model built with a deliberately different (smaller) vocabulary
    small_alphabet = {"-": 0, "X": 1, "A": 2, "C": 3}
    bad_model = PrositRetentionTimePredictor(seq_length=22, alphabet=small_alphabet)
    with pytest.raises(ValueError, match="Model/preprocessor mismatch"):
        InferencePipeline.from_model_and_dataset(bad_model, rt_dataset)


def test_consistency_check_rejects_mismatched_seq_len(rt_dataset):
    # rt_dataset.max_seq_len is 20; a model expecting 22 (20 + termini) should be rejected.
    bad_model = _FakeSeqLenModel(raw_seq_length=20, with_termini=True)
    with pytest.raises(ValueError, match="Model/preprocessor mismatch"):
        InferencePipeline.from_model_and_dataset(bad_model, rt_dataset)


def test_save_load_reproduces_predictions(rt_dataset, rt_model, tmp_path):
    pipe = InferencePipeline.from_model_and_dataset(rt_model, rt_dataset)
    seqs = ["ACDEFGHIK", "PEPTIDEK", "MKLVAAR"]
    before = pipe.predict(seqs)

    save_dir = str(tmp_path / "bundle")
    pipe.save(save_dir)

    loaded = InferencePipeline.load(save_dir)
    after = loaded.predict(seqs)

    assert before.shape == after.shape
    np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-4)


def test_save_refuses_existing_nonempty_dir(rt_dataset, rt_model, tmp_path):
    pipe = InferencePipeline.from_model_and_dataset(rt_model, rt_dataset)
    save_dir = tmp_path / "bundle"
    save_dir.mkdir()
    (save_dir / "sentinel.txt").write_text("x")
    with pytest.raises(FileExistsError):
        pipe.save(str(save_dir))
    # overwrite=True succeeds
    assert pipe.save(str(save_dir), overwrite=True)


def test_push_to_hub_uploads_bundle(rt_dataset, rt_model, tmp_path, monkeypatch):
    """push_to_hub writes the full bundle + model card and uploads the folder."""
    import huggingface_hub

    pipe = InferencePipeline.from_model_and_dataset(rt_model, rt_dataset)
    captured = {}

    def fake_create_repo(repo_id, **kwargs):
        captured["repo_id"] = repo_id

    class FakeApi:
        def __init__(self, token=None):
            captured["token"] = token

        def upload_folder(self, repo_id, folder_path, commit_message=None, **kwargs):
            captured["folder"] = folder_path
            captured["delete_patterns"] = kwargs.get("delete_patterns")
            captured["files"] = sorted(p.name for p in Path(folder_path).iterdir())

    monkeypatch.setattr(huggingface_hub, "create_repo", fake_create_repo)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    pipe.push_to_hub("user/rt-model", token="hf_dummy")

    assert captured["repo_id"] == "user/rt-model"
    assert "README.md" in captured["files"]
    assert "dlomix_inference_pipeline.json" in captured["files"]
    assert "dlomix_preprocessor.json" in captured["files"]
    assert any(f.startswith("model.") for f in captured["files"])
    # mirrors the repo to the bundle (drops stale files from prior pushes)
    assert captured["delete_patterns"] == "*"


def test_from_pretrained_delegates_to_load(rt_dataset, rt_model, tmp_path, monkeypatch):
    """from_pretrained downloads the snapshot then reuses load()."""
    import huggingface_hub

    pipe = InferencePipeline.from_model_and_dataset(rt_model, rt_dataset)
    save_dir = str(tmp_path / "snapshot")
    pipe.save(save_dir)

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kwargs: save_dir
    )

    loaded = InferencePipeline.from_pretrained("user/rt-model")
    preds = loaded.predict(["ACDEFGHIK", "PEPTIDEK"])
    assert preds.shape[0] == 2


def test_model_card_seq_only_uses_plain_list(tmp_path):
    card = _make_card(tmp_path)
    assert 'pipeline.predict(["PEPTIDEK"' in card
    assert "predict({" not in card  # no dict form for a sequence-only model
    assert "requires the feature column" not in card


def test_model_card_with_model_features_uses_dict(tmp_path):
    card = _make_card(
        tmp_path,
        model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    )
    assert "predict({" in card
    assert '"modified_sequence":' in card
    assert '"collision_energy_aligned_normed": [...]' in card
    assert '"precursor_charge_onehot": [...]' in card
    assert "requires the feature column(s)" in card


def test_model_card_notes_extracted_ptm_features(tmp_path):
    card = _make_card(
        tmp_path,
        model_features=["collision_energy_aligned_normed"],
        features_to_extract=["mod_loss", "delta_mass"],
    )
    # extracted features are computed, not supplied -> not in the predict() dict
    assert '"mod_loss"' not in card
    assert '"delta_mass"' not in card
    assert "computed from the sequence automatically" in card
    assert "`mod_loss`, `delta_mass`" in card
