"""
Regression tests for FineTunePipeline.
"""

from os.path import join
from pathlib import Path
from unittest.mock import patch

import pytest
import tensorflow as tf
import yaml

from dlomix.constants import ALPHABET_UNMOD
from dlomix.models import PrositIntensityPredictor
from dlomix.pipelines.finetune import FineTunePipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_intensity_model(tmp_path):
    model = PrositIntensityPredictor(
        embedding_output_dim=8,
        seq_length=30,
        alphabet=ALPHABET_UNMOD,
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
    # with_termini=False keeps sequences at max_seq_len=30, matching the model
    # default (PrositIntensityPredictor also defaults to with_termini=False).
    # The dataset default is with_termini=True which would pad to 32 and break
    # the AttentionLayer bias shape.
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

    def test_best_fit_kwargs_stored(self):
        kwargs = {"new_hf_data": "placeholder", "sequence_column": "seq"}
        p = FineTunePipeline(
            finetune_dataset_path="data.parquet",
            base_model_name="m",
            initialization_strategy="best-fit",
            best_fit_kwargs=kwargs,
        )
        assert p.best_fit_kwargs is kwargs


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
    def test_setup_populates_model_and_dataset(
        self,
        saved_intensity_model,
        intensity_parquet_path,
        intensity_dataset_kwargs,
    ):
        pipeline = FineTunePipeline(
            finetune_dataset_path=intensity_parquet_path,
            base_model_weights_filepath=saved_intensity_model,
            dataset_kwargs=intensity_dataset_kwargs,
        )
        pipeline.setup()

        assert pipeline.model is not None
        assert pipeline.dataset is not None

    def test_finetune_returns_history(
        self,
        saved_intensity_model,
        intensity_parquet_path,
        intensity_dataset_kwargs,
    ):
        pipeline = FineTunePipeline(
            finetune_dataset_path=intensity_parquet_path,
            base_model_weights_filepath=saved_intensity_model,
            epochs=1,
            dataset_kwargs=intensity_dataset_kwargs,
        )
        pipeline.setup()
        history = pipeline.finetune()

        assert history is not None
        assert "loss" in history.history

    def test_save_writes_model_to_disk(
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
        pipeline.finetune()
        returned_path = pipeline.save()

        assert returned_path == out_path
        assert Path(out_path).exists()

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
