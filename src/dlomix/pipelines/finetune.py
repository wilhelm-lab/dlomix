"""
FineTunePipeline
================

Typical usage
-------------
    # Option A — constructor directly
    pipeline = FineTunePipeline(
        finetune_dataset_path="data/finetuning.parquet",
        base_model_name="Prosit_2020_intensity_HCD",
        epochs=20,
        learning_rate=3e-4,
    )
    pipeline.setup()
    history = pipeline.finetune()

    # Option B — from a YAML config file
    pipeline = FineTunePipeline.from_config_file("configs/finetune.yaml")
    pipeline.setup()
    history = pipeline.finetune()

    # Option C — chained one-liner
    history = FineTunePipeline(...).setup().finetune()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..data import FragmentIonIntensityDataset, PeptideDataset
from ..losses import masked_spectral_distance
from ..models import download_remote_model_weights, load_and_adapt_pretrained_model

logger = logging.getLogger(__name__)


class FineTunePipeline:
    """Fine-tuning pipeline for DLomix fragment ion intensity models.

    Parameters
    ----------
    finetune_dataset_path:
        Path to the Parquet file used for fine-tuning.
    base_model_name:
        Name of a registered remote model whose weights will be downloaded
        automatically.  Either this *or* ``base_model_weights_filepath`` must
        be supplied.
    base_model_weights_filepath:
        Path to a local model weights file.  Takes precedence over
        ``base_model_name`` when both are given.
    old_model_vocab:
        Amino-acid vocabulary of the *pretrained* model.  ``None`` means the
        pipeline will infer it from the loaded model object provided via the weights file.
    new_model_vocab:
        Target vocabulary for the fine-tuned model.  ``None`` means the
        pipeline will derive it from the dataset's ``extended_alphabet``.
    initialization_strategy:
        Strategy for initialising weights that are new after a vocabulary
        expansion (e.g. ``"random"``, ``"zeros"``, ``"mean"``, ``"best-fit"``).
    seed:
        Random seed for reproducibility.
    output_model_path:
        Directory where the fine-tuned model will be saved after training.
    epochs:
        Number of training epochs.
    batch_size:
        Batch size used by the dataset loader.
    learning_rate:
        Initial learning rate for the Adam optimiser.
    dataset_kwargs:
        Extra keyword arguments forwarded verbatim to
        :class:`PeptideDataset` or one of its dataset subclasses.
    """

    def __init__(
        self,
        finetune_dataset_path: str,
        base_model_name: str | None = None,
        base_model_weights_filepath: str | None = None,
        old_model_vocab: dict | None = None,
        new_model_vocab: dict | None = None,
        initialization_strategy: str = "random",
        seed: int = 42,
        output_model_path: str = "./finetuned_model",
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        dataset_kwargs: dict | None = None,
    ) -> None:
        if not base_model_name and not base_model_weights_filepath:
            raise ValueError(
                "Provide either 'base_model_name' (remote download) or "
                "'base_model_weights_filepath' (local file)."
            )

        self.finetune_dataset_path = finetune_dataset_path
        self.base_model_name = base_model_name
        self.base_model_weights_filepath = base_model_weights_filepath

        self.old_model_vocab = old_model_vocab
        self.new_model_vocab = new_model_vocab
        self.initialization_strategy = initialization_strategy
        self.seed = seed

        self.output_model_path = output_model_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset_kwargs = dataset_kwargs or {}

        # populated by setup()
        self.dataset: PeptideDataset | None = None
        self.model: Any | None = None

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config_file(cls, config_path: str) -> FineTunePipeline:
        """Instantiate a pipeline from a YAML config file.

        The config file must contain a mapping whose keys match the
        constructor's parameter names.  YAML requires PyYAML to be installed.

        Parameters
        ----------
        config_path:
            Path to a ``.yaml``, ``.yml`` config file.

        Returns
        -------
        FineTunePipeline
            An uninitialised pipeline (call :meth:`setup` before
            :meth:`finetune`).

        Example YAML
        ------------
        .. code-block:: yaml

            finetune_dataset_path: data/finetuning.parquet
            base_model_name: Prosit_2020_intensity_HCD
            epochs: 20
            learning_rate: 0.0003
            output_model_path: models/my_finetuned
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load YAML configs: pip install pyyaml"
                ) from exc
            with path.open() as fh:
                config = yaml.safe_load(fh)
        else:
            raise ValueError(
                f"Unsupported config format '{path.suffix}'. Use .yaml or .yml."
            )

        return cls(**config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> FineTunePipeline:
        """Prepare the dataset and load / adapt the model.

        Must be called before :meth:`finetune`.  Returns *self* so calls can
        be chained::

            history = FineTunePipeline(...).setup().finetune()
        """
        logger.info("Setting up FineTunePipeline …")
        self._prepare_dataset()
        self._resolve_vocab()
        self._resolve_weights()
        self._load_and_adapt_model()
        logger.info("Setup complete.")
        return self

    def _prepare_dataset(self) -> None:
        logger.info("Loading dataset from '%s' …", self.finetune_dataset_path)
        self.dataset = FragmentIonIntensityDataset(
            data_format="parquet",
            data_source=self.finetune_dataset_path,
            batch_size=self.batch_size,
            **self.dataset_kwargs,
        )

    def _resolve_vocab(self) -> None:
        """Fall back to the dataset's extended alphabet when no target vocab is given."""
        if self.new_model_vocab is None:
            logger.info(
                "No new_model_vocab supplied — using dataset.extended_alphabet."
            )
            self.new_model_vocab = self.dataset.extended_alphabet

    def _resolve_weights(self) -> None:
        """Resolve local weights path, downloading from remote when necessary."""
        if not self.base_model_weights_filepath:
            logger.info(
                "No local weights path given — downloading '%s' …", self.base_model_name
            )
            self.base_model_weights_filepath = download_remote_model_weights(
                self.base_model_name
            )

    def _load_and_adapt_model(self) -> None:
        logger.info(
            "Loading and adapting model from '%s' …", self.base_model_weights_filepath
        )
        self.model = load_and_adapt_pretrained_model(
            model_path=self.base_model_weights_filepath,
            new_alphabet=self.new_model_vocab,
            old_alphabet=self.old_model_vocab,
            initialization_strategy=self.initialization_strategy,
            random_seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def finetune(
        self,
        loss_fn=None,
        metrics: list | None = None,
        callbacks: list | None = None,
        optimizer=None,
    ):
        """Compile and fine-tune the model.

        Parameters
        ----------
        loss_fn:
            A compiled Keras loss function or callable.  Defaults to
            :func:`masked_spectral_distance`.
        metrics:
            List of Keras metrics to track during training.
        callbacks:
            List of Keras callbacks (e.g. ``EarlyStopping``, ``ModelCheckpoint``).
        optimizer:
            A Keras optimiser instance.  Defaults to ``Adam`` with the
            pipeline's ``learning_rate``.

        Returns
        -------
        keras.callbacks.History
            The Keras training history object returned by ``model.fit``.

        Raises
        ------
        RuntimeError
            If :meth:`setup` has not been called yet.
        """
        self._require_setup()

        import tensorflow as tf

        resolved_loss = loss_fn or masked_spectral_distance
        resolved_optimizer = optimizer or tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        logger.info(
            "Starting fine-tuning: epochs=%d, lr=%g, loss=%s",
            self.epochs,
            self.learning_rate,
            getattr(resolved_loss, "__name__", repr(resolved_loss)),
        )

        self.model.compile(
            optimizer=resolved_optimizer,
            loss=resolved_loss,
            metrics=metrics or [],
        )

        history = self.model.fit(
            self.dataset.tensor_train_data,
            epochs=self.epochs,
            validation_data=self.dataset.tensor_val_data,
            callbacks=callbacks or [],
        )

        logger.info("Fine-tuning complete.")
        return history

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Save the fine-tuned model weights to disk.

        Parameters
        ----------
        path:
            Destination directory or file path.  Falls back to
            ``self.output_model_path`` when omitted.

        Returns
        -------
        str
            The path the model was saved to.
        """
        self._require_setup()
        destination = path or self.output_model_path
        logger.info("Saving model to '%s' …", destination)
        self.model.save(destination)
        return destination

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_setup(self) -> None:
        """Raise a clear error when the user forgot to call setup()."""
        if self.model is None or self.dataset is None:
            raise RuntimeError(
                "Pipeline is not ready. Call pipeline.setup() before "
                "calling finetune() or save()."
            )

    def __repr__(self) -> str:
        status = "ready" if self.model is not None else "not set up"
        return (
            f"FineTunePipeline("
            f"model={self.base_model_name or Path(self.base_model_weights_filepath).name!r}, "
            f"epochs={self.epochs}, "
            f"lr={self.learning_rate}, "
            f"status={status!r})"
        )
