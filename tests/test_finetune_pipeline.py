"""
Regression tests for FineTunePipeline.
"""

import os
import subprocess
import sys
from os.path import join
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
import yaml
from datasets import Dataset

from dlomix.constants import ALPHABET_UNMOD
from dlomix.data import FragmentIonIntensityDataset
from dlomix.models import PrositIntensityPredictor, load_and_adapt_pretrained_model
from dlomix.pipelines import InferencePipeline
from dlomix.pipelines.finetune import FineTunePipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_intensity_model(tmp_path):
    # with_termini=False: this model is built/pickled at a fixed width of 30 (see the
    # dummy input below), matching the with_termini=False dataset in
    # intensity_dataset_kwargs. Leaving this at the class default (True) would make the
    # model's own with_termini metadata disagree with that width.
    model = PrositIntensityPredictor(
        embedding_output_dim=8,
        seq_length=30,
        alphabet=ALPHABET_UNMOD,
        with_termini=False,
        use_meta_data=True,
        meta_data_keys=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    )
    dummy = {
        "sequence": tf.zeros((2, 30), dtype=tf.int32),
        "collision_energy_aligned_normed": tf.ones((2, 1)),
        "precursor_charge_onehot": tf.ones((2, 6)),
    }
    model(dummy)
    path = str(tmp_path / "model.keras")
    model.save(path)
    return path


@pytest.fixture
def intensity_parquet_path(download_path_for_assets):
    return join(download_path_for_assets, "file_3.parquet")


@pytest.fixture
def intensity_dataset_kwargs():
    # with_termini=False keeps sequences at max_seq_len=30, matching the fixed
    # width `saved_intensity_model` was built with (seq_length=30, dummy input
    # shaped (2, 30)). The dataset default is with_termini=True, which would pad
    # to 32 and break the AttentionLayer bias shape.
    return {
        "sequence_column": "sequence",
        "label_column": "intensities",
        "model_features": [
            "precursor_charge_onehot",
            "collision_energy_aligned_normed",
        ],
        "with_termini": False,
        "val_ratio": 0.1,
    }


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_requires_at_least_one_model_source(self):
        with pytest.raises(
            ValueError, match="base_model_name.*base_model_weights_filepath"
        ):
            FineTunePipeline(finetune_dataset_path="data.parquet")

    def test_defaults(self):
        p = FineTunePipeline(
            finetune_dataset_path="data.parquet",
            base_model_name="some_model",
        )
        assert p.initialization_strategy == "random"
        assert p.epochs == 10
        assert p.batch_size == 64
        assert p.learning_rate == 1e-4
        assert p.seed == 42
        assert p.best_fit_kwargs is None
        assert p.model is None
        assert p.dataset is None
        assert p.best_fit_info is None


# ---------------------------------------------------------------------------
# from_config_file
# ---------------------------------------------------------------------------


class TestFromConfigFile:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FineTunePipeline.from_config_file(str(tmp_path / "nonexistent.yaml"))

    def test_bad_extension_raises(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported config format"):
            FineTunePipeline.from_config_file(str(f))

    def test_valid_yaml_creates_pipeline(self, tmp_path):
        config = {
            "finetune_dataset_path": "data.parquet",
            "base_model_name": "Prosit_2020_intensity_HCD",
            "epochs": 5,
            "learning_rate": 3e-4,
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(config))

        p = FineTunePipeline.from_config_file(str(cfg_file))

        assert p.epochs == 5
        assert p.learning_rate == pytest.approx(3e-4)
        assert p.base_model_name == "Prosit_2020_intensity_HCD"


# ---------------------------------------------------------------------------
# Pre-setup guards
# ---------------------------------------------------------------------------


class TestPreSetupGuards:
    @pytest.fixture
    def unsetup_pipeline(self, tmp_path):
        return FineTunePipeline(
            finetune_dataset_path="data.parquet",
            base_model_weights_filepath=str(tmp_path / "model.keras"),
        )

    def test_finetune_before_setup_raises(self, unsetup_pipeline):
        with pytest.raises(RuntimeError, match="setup"):
            unsetup_pipeline.finetune()

    def test_repr_does_not_crash_before_setup(self, unsetup_pipeline):
        r = repr(unsetup_pipeline)
        assert "FineTunePipeline" in r
        assert "not set up" in r


# ---------------------------------------------------------------------------
# Integration: setup() + finetune() + save()
# ---------------------------------------------------------------------------


class TestSetupAndFinetune:
    def test_setup_finetune_save(
        self,
        saved_intensity_model,
        intensity_parquet_path,
        intensity_dataset_kwargs,
        tmp_path,
    ):
        out_path = str(tmp_path / "finetuned")
        pipeline = FineTunePipeline(
            finetune_dataset_path=intensity_parquet_path,
            base_model_weights_filepath=saved_intensity_model,
            epochs=1,
            output_model_path=out_path,
            dataset_kwargs=intensity_dataset_kwargs,
        )
        pipeline.setup()
        assert pipeline.model is not None
        assert pipeline.dataset is not None

        history = pipeline.finetune()
        assert "loss" in history.history

        # Keras 3 only writes to a .keras path, so save() normalises the
        # extension and reports the file it actually wrote.
        returned_path = pipeline.save()
        assert returned_path == out_path + ".keras"
        assert Path(returned_path).exists()

    def test_base_model_name_downloads_from_the_hub(
        self,
        saved_intensity_model,
        intensity_parquet_path,
        intensity_dataset_kwargs,
    ):
        """base_model_name is a Hub repo id; its model.keras is downloaded."""
        with patch(
            "huggingface_hub.hf_hub_download", return_value=saved_intensity_model
        ) as mock_download:
            pipeline = FineTunePipeline(
                finetune_dataset_path=intensity_parquet_path,
                base_model_name="my-org/prosit-intensity",
                dataset_kwargs=intensity_dataset_kwargs,
            )
            pipeline.setup()

        mock_download.assert_called_once()
        assert mock_download.call_args.kwargs["repo_id"] == "my-org/prosit-intensity"
        assert mock_download.call_args.kwargs["filename"] == "model.keras"
        assert pipeline.base_model_weights_filepath == saved_intensity_model
        assert pipeline.model is not None

    def test_new_model_vocab_is_used_to_encode_the_dataset(
        self,
        saved_intensity_model,
        intensity_parquet_path,
        intensity_dataset_kwargs,
    ):
        """An explicit new_model_vocab must also encode the data; a dataset left
        to learn its own alphabet would give the same tokens other indices."""
        from dlomix.data import FragmentIonIntensityDataset

        learned = FragmentIonIntensityDataset(
            data_format="parquet",
            data_source=intensity_parquet_path,
            **intensity_dataset_kwargs,
        ).extended_alphabet
        # same tokens, two residues swapped: a vocab no dataset would learn
        vocab = dict(learned)
        residues = [t for t in vocab if t not in ("-", "X")]
        vocab[residues[0]], vocab[residues[1]] = vocab[residues[1]], vocab[residues[0]]

        pipeline = FineTunePipeline(
            finetune_dataset_path=intensity_parquet_path,
            base_model_weights_filepath=saved_intensity_model,
            new_model_vocab=vocab,
            dataset_kwargs=intensity_dataset_kwargs,
        )
        pipeline.setup()

        assert pipeline.dataset.extended_alphabet == vocab
        assert pipeline.model.alphabet == vocab

    def test_conflicting_dataset_alphabet_raises(
        self, intensity_parquet_path, saved_intensity_model
    ):
        pipeline = FineTunePipeline(
            finetune_dataset_path=intensity_parquet_path,
            base_model_weights_filepath=saved_intensity_model,
            new_model_vocab={"-": 0, "X": 1, "A": 2},
            dataset_kwargs={"alphabet": {"-": 0, "X": 1, "C": 2}},
        )
        with pytest.raises(ValueError, match="new_model_vocab"):
            pipeline._prepare_dataset()

    def test_best_fit_kwargs_forwarded_to_load_and_adapt(
        self,
        saved_intensity_model,
        intensity_parquet_path,
        intensity_dataset_kwargs,
    ):
        """best_fit_kwargs must reach load_and_adapt_pretrained_model unchanged."""
        sentinel = {"new_hf_data": None, "sequence_column": "seq"}

        with patch(
            "dlomix.pipelines.finetune.load_and_adapt_pretrained_model",
            return_value=tf.keras.Sequential(),
        ) as mock_load:
            pipeline = FineTunePipeline(
                finetune_dataset_path=intensity_parquet_path,
                base_model_weights_filepath=saved_intensity_model,
                initialization_strategy="best-fit",
                best_fit_kwargs=sentinel,
                dataset_kwargs=intensity_dataset_kwargs,
            )
            pipeline._prepare_dataset()
            pipeline._resolve_vocab()
            pipeline._load_and_adapt_model()

            _, call_kwargs = mock_load.call_args
            assert call_kwargs["best_fit_kwargs"] is sentinel
            assert call_kwargs["initialization_strategy"] == "best-fit"


# ---------------------------------------------------------------------------
# Shared fixture: a fully set-up pipeline (dataset + model loaded)
# ---------------------------------------------------------------------------


@pytest.fixture
def ready_pipeline(
    saved_intensity_model, intensity_parquet_path, intensity_dataset_kwargs
):
    p = FineTunePipeline(
        finetune_dataset_path=intensity_parquet_path,
        base_model_weights_filepath=saved_intensity_model,
        epochs=1,
        dataset_kwargs=intensity_dataset_kwargs,
    )
    p.setup()
    return p


# ---------------------------------------------------------------------------
# Backend guard
# ---------------------------------------------------------------------------


class TestBackendGuard:
    def test_finetune_raises_on_torch_backend(self):
        # The guard fires before any training work — no real dataset/model needed.
        p = FineTunePipeline(finetune_dataset_path="x", base_model_name="m")
        p.model = object()
        p.dataset = object()
        with patch("dlomix.pipelines.finetune._IS_TORCH", True):
            with pytest.raises(NotImplementedError, match="TensorFlow"):
                p.finetune()

    def test_setup_and_save_raise_on_torch_backend(self):
        p = FineTunePipeline(finetune_dataset_path="x", base_model_name="m")
        with patch("dlomix.pipelines.finetune._IS_TORCH", True):
            # setup() fails before touching the (non-existent) dataset path
            with pytest.raises(NotImplementedError, match=r"setup\(\).*TensorFlow"):
                p.setup()
            p.model = object()
            p.dataset = object()
            with pytest.raises(NotImplementedError, match=r"save\(\).*TensorFlow"):
                p.save()

    def test_pipelines_import_on_torch_backend(self):
        # The backend is fixed at import time, so check it in a fresh interpreter.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from dlomix.pipelines import FineTunePipeline, InferencePipeline",
            ],
            env={**os.environ, "DLOMIX_BACKEND": "pytorch"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# save() overwrite behaviour
# ---------------------------------------------------------------------------


class TestSaveOverwrite:
    def test_save_raises_on_existing_path(self, tmp_path):
        # FileExistsError fires before model.save() — no real model needed.
        p = FineTunePipeline(finetune_dataset_path="x", base_model_name="m")
        p.model = object()
        p.dataset = object()
        out = tmp_path / "model.keras"
        out.touch()
        with pytest.raises(FileExistsError):
            p.save(str(out))

    def test_save_overwrite_true_succeeds(self, ready_pipeline, tmp_path):
        out = tmp_path / "model.keras"
        out.touch()
        result = ready_pipeline.save(str(out), overwrite=True)
        assert result == str(out)


# ---------------------------------------------------------------------------
# from_dataset_and_model
# ---------------------------------------------------------------------------


class TestFromDatasetAndModel:
    def test_is_immediately_ready(self, ready_pipeline):
        p = FineTunePipeline.from_dataset_and_model(
            dataset=ready_pipeline.dataset,
            model=ready_pipeline.model,
        )
        assert p.model is ready_pipeline.model
        assert p.dataset is ready_pipeline.dataset

    def test_repr_shows_provided(self):
        # Repr only needs the attributes set by from_dataset_and_model — mock is enough.
        from unittest.mock import MagicMock

        mock_ds = MagicMock()
        mock_ds.batch_size = 64
        p = FineTunePipeline.from_dataset_and_model(dataset=mock_ds, model=object())
        r = repr(p)
        assert "provided" in r
        assert "ready" in r


# ---------------------------------------------------------------------------
# to_inference_pipeline
# ---------------------------------------------------------------------------


class TestToInferencePipeline:
    def test_raises_before_setup(self):
        p = FineTunePipeline(finetune_dataset_path="x", base_model_name="m")
        with pytest.raises(RuntimeError, match="setup"):
            p.to_inference_pipeline()

    def test_returns_inference_pipeline(self, ready_pipeline):
        pipe = ready_pipeline.to_inference_pipeline()
        assert isinstance(pipe, InferencePipeline)
        assert pipe.model is ready_pipeline.model
        assert pipe.preprocessor.vocab_size == len(
            ready_pipeline.dataset.extended_alphabet
        )
        assert (
            pipe.preprocessor.sequence_column == ready_pipeline.dataset.sequence_column
        )


# ---------------------------------------------------------------------------
# What finetune() trains
# ---------------------------------------------------------------------------

OLD_VOCAB = {"-": 0, "X": 1, "A": 2, "C": 3, "D": 4, "E": 5, "F": 6}
NEW_TOKEN = "C[UNIMOD:4]"
NEW_VOCAB = {**OLD_VOCAB, NEW_TOKEN: 7}
FEATURES = ["collision_energy", "precursor_charge"]


def _small_dataset(sequences):
    rng = np.random.default_rng(0)
    return FragmentIonIntensityDataset(
        data_format="hf",
        data_source=Dataset.from_dict(
            {
                "sequence": sequences,
                "collision_energy": [0.3] * len(sequences),
                "precursor_charge": [2.0] * len(sequences),
                "label": rng.random((len(sequences), 54)).tolist(),
            }
        ),
        sequence_column="sequence",
        label_column="label",
        model_features=FEATURES,
        alphabet=NEW_VOCAB,
        max_seq_len=10,
        with_termini=False,
        val_ratio=0.25,
        batch_size=4,
        num_proc=None,
    )


@pytest.fixture
def base_model_path(tmp_path):
    model = PrositIntensityPredictor(
        embedding_output_dim=8,
        seq_length=10,
        alphabet=OLD_VOCAB,
        with_termini=False,
        use_meta_data=True,
        meta_data_keys=FEATURES,
        recurrent_layers_sizes=(8, 16),
        regressor_layer_size=16,
    )
    model(
        {
            "sequence": tf.zeros((2, 10), dtype=tf.int32),
            "collision_energy": tf.ones((2, 1)),
            "precursor_charge": tf.ones((2, 1)),
        }
    )
    path = str(tmp_path / "base.keras")
    model.save(path)
    return path


@pytest.fixture
def new_token_pipeline(base_model_path):
    """A model whose vocabulary gained NEW_TOKEN, and data to fine-tune it on."""
    tf.keras.utils.set_random_seed(0)
    model = load_and_adapt_pretrained_model(
        base_model_path, NEW_VOCAB, initialization_strategy="mean"
    )
    dataset = _small_dataset([f"A{NEW_TOKEN}{rest}" for rest in "ACDEF"] * 4)
    return FineTunePipeline.from_dataset_and_model(
        dataset, model, epochs=2, learning_rate=1e-2, new_tokens=[NEW_TOKEN]
    )


def _weights(model):
    """(embedding rows of the known tokens, row of the new token, all other weights)"""
    embedding = model.embedding.get_weights()[0]
    body = [w.numpy() for w in model.weights if w is not model.embedding.embeddings]
    return embedding[:7], embedding[7], body


def _changed(before, after):
    return not all(np.array_equal(b, a) for b, a in zip(before, after, strict=True))


class TestTrainable:
    @pytest.mark.parametrize(
        "trainable, changes",  # known tokens' rows, new token's row, other weights
        [
            ("all", (True, True, True)),
            ("embedding", (True, True, False)),
            ("new_tokens", (False, True, False)),
        ],
    )
    def test_only_the_weights_of_the_mode_change(
        self, new_token_pipeline, trainable, changes
    ):
        model = new_token_pipeline.model
        before = _weights(model)
        n_trainable = len(model.trainable_weights)

        new_token_pipeline.finetune(trainable=trainable)

        after = _weights(model)
        assert _changed(before[0], after[0]) == changes[0]
        assert _changed(before[1], after[1]) == changes[1]
        assert _changed(before[2], after[2]) == changes[2]
        # the model is not left frozen
        assert len(model.trainable_weights) == n_trainable

    def test_new_tokens_keeps_predictions_for_known_peptides_also_after_saving(
        self, new_token_pipeline, tmp_path
    ):
        known_peptides = _small_dataset(["ACDEF", "FEDCA", "AACCD", "DDEEF"])
        model = new_token_pipeline.model
        before = model.predict(known_peptides.tensor_train_data, verbose=0)

        new_token_pipeline.finetune(trainable="new_tokens")

        reloaded = tf.keras.models.load_model(
            new_token_pipeline.save(str(tmp_path / "finetuned.keras"))
        )
        for finetuned in (model, reloaded):
            after = finetuned.predict(known_peptides.tensor_train_data, verbose=0)
            np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(
            reloaded.embedding.get_weights()[0], model.embedding.get_weights()[0]
        )

    def test_new_tokens_are_found_from_the_base_model(self, base_model_path):
        pipeline = FineTunePipeline(
            finetune_dataset_path="unused.parquet",
            base_model_weights_filepath=base_model_path,
            trainable="new_tokens",
        )
        pipeline.model = load_and_adapt_pretrained_model(base_model_path, NEW_VOCAB)

        assert pipeline._resolve_new_tokens() == [NEW_TOKEN]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="trainable"):
            FineTunePipeline(
                finetune_dataset_path="x", base_model_name="m", trainable="body"
            )


class TestKeepStartingModel:
    @pytest.mark.parametrize(
        "learning_rate, kept_model",
        [(1e-2, "trained"), (-1e-2, "starting")],  # descent, ascent
    )
    def test_starting_weights_return_when_no_epoch_beats_them(
        self, new_token_pipeline, learning_rate, kept_model
    ):
        model = new_token_pipeline.model
        before = model.get_weights()

        history = new_token_pipeline.finetune(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate)
        )

        improved = min(history.history["val_loss"]) < history.starting_val_loss
        assert improved == (kept_model == "trained")
        assert history.kept_model == kept_model
        assert _changed(before, model.get_weights()) == (kept_model == "trained")

    def test_can_be_turned_off(self, new_token_pipeline):
        before = new_token_pipeline.model.get_weights()

        history = new_token_pipeline.finetune(
            optimizer=tf.keras.optimizers.SGD(learning_rate=-1e-2),
            keep_starting_model=False,
        )

        assert not hasattr(history, "kept_model")
        assert _changed(before, new_token_pipeline.model.get_weights())
