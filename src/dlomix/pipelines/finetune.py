"""
FineTunePipeline
================

Typical usage
-------------
    # Option A — constructor directly
    pipeline = FineTunePipeline(
        finetune_dataset_path="data/finetuning.parquet",
        base_model_name="my-org/prosit-intensity",  # Hugging Face Hub repo id
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

    # Option D — from a pretrained inference bundle and Hugging Face data: the
    # bundle's preprocessing settings build the dataset, the tokens new to the
    # model are found in the data and added to its vocabulary
    pipeline = FineTunePipeline.from_pretrained(
        "my-org/prosit-intensity", data, initialization_strategy="best-fit"
    )
    history = pipeline.finetune()

    # Option E — every step of option D by hand, to change any of them
    base = InferencePipeline.from_pretrained("my-org/prosit-intensity")
    alphabet = extend_alphabet(
        base.model.alphabet,
        find_new_tokens(data["train"]["modified_sequence"], base.model.alphabet),
    )
    dataset = FragmentIonIntensityDataset(
        data_source=data, data_format="hf", alphabet=alphabet,
        **base.preprocessor.dataset_settings,
    )
    model = adapt_pretrained_model(base.model, alphabet, initialization_strategy="mean")
    pipeline = FineTunePipeline.from_dataset_and_model(dataset, model)
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from ..config import _BACKEND, PYTORCH_BACKEND
from ..data import (
    FragmentIonIntensityDataset,
    PeptideDataset,
    extend_alphabet,
    find_new_tokens,
)
from ..losses import masked_spectral_distance

logger = logging.getLogger(__name__)

_IS_TORCH = _BACKEND in PYTORCH_BACKEND

# The pretrained-model helpers live in the TensorFlow-only ``models.model_utils``,
# which imports TensorFlow at module level, so they are only imported on that
# backend. Fine-tuning is TensorFlow-only for now: the methods that need these
# helpers call ``_require_tensorflow`` first.
if _IS_TORCH:
    adapt_pretrained_model = None
    download_remote_model_weights = None
    load_and_adapt_pretrained_model = None
else:
    from ..models import (
        adapt_pretrained_model,
        download_remote_model_weights,
        load_and_adapt_pretrained_model,
    )

TRAINABLE_MODES = ("all", "embedding", "new_tokens")


def _require_tensorflow(operation: str) -> None:
    """Raise :exc:`NotImplementedError` for ``operation`` on the PyTorch backend."""
    if _IS_TORCH:
        raise NotImplementedError(
            f"FineTunePipeline.{operation}() currently supports TensorFlow only. "
            "Set DLOMIX_BACKEND=tensorflow before importing dlomix."
        )


def _validated_trainable(trainable: str) -> str:
    if trainable not in TRAINABLE_MODES:
        raise ValueError(
            f"Unknown trainable mode {trainable!r}, expected one of {TRAINABLE_MODES}."
        )
    return trainable


_KERAS_MODEL_SUFFIXES = (".keras", ".h5")


def _ensure_keras_extension(destination: str) -> str:
    """Return ``destination`` with a Keras-writable suffix.

    Keras 3 refuses to save to a path that does not end in ``.keras`` or
    ``.h5``, so a bare path (including the default ``./finetuned_model``) is
    given a ``.keras`` suffix rather than failing at the end of a training run.
    """
    if destination.endswith(_KERAS_MODEL_SUFFIXES):
        return destination
    adjusted = f"{destination}.keras"
    logger.info("Appended '.keras' to the save path: '%s'", adjusted)
    return adjusted


class FineTunePipeline:
    """Fine-tuning pipeline for DLomix fragment ion intensity models.

    Parameters
    ----------
    finetune_dataset_path:
        Path to the Parquet file used for fine-tuning.
    base_model_name:
        Hugging Face Hub repo id of a DLOmix inference bundle saved on the
        TensorFlow backend (:meth:`InferencePipeline.push_to_hub`); its
        ``model.keras`` is downloaded automatically.  Either this *or*
        ``base_model_weights_filepath`` must be supplied.
    base_model_weights_filepath:
        Path to a local saved model (a full ``.keras`` model, not a weights-only
        file).  Takes precedence over ``base_model_name`` when both are given.
    old_model_vocab:
        Amino-acid vocabulary of the *pretrained* model.  ``None`` means the
        pipeline will infer it from the loaded model object provided via the weights file.
    new_model_vocab:
        Target vocabulary for the fine-tuned model.  ``None`` means the
        pipeline will derive it from the dataset's ``extended_alphabet``.  When
        given, the dataset is encoded with it too (as its ``alphabet``), so the
        model and the data use the same token indices.
    initialization_strategy:
        Strategy for initialising the embeddings of tokens that are new after a
        vocabulary expansion: ``"random"`` (Glorot uniform), ``"mean"`` (the mean
        of the pretrained embeddings) or ``"best-fit"`` (the embedding of the
        pretrained token that predicts the new token's spectra best).
    best_fit_kwargs:
        Required when ``initialization_strategy="best-fit"``.  Forwarded
        verbatim to :func:`load_and_adapt_pretrained_model`.  Must contain at
        minimum ``new_hf_data`` (a Hugging Face ``Dataset`` of raw sequences).
        Optionally ``sequence_column``, ``label_column``, ``n_examples_for_eval``,
        ``min_examples_for_eval``, ``eval_metric``, ``return_fit_info`` (bool) and
        ``dataset_kwargs`` (dict).
    seed:
        Random seed for reproducibility.
    trainable:
        What :meth:`finetune` trains: ``"all"`` (default), ``"embedding"`` or
        ``"new_tokens"``; see :meth:`finetune`.
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
        best_fit_kwargs: dict | None = None,
        seed: int = 42,
        output_model_path: str = "./finetuned_model",
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        dataset_kwargs: dict | None = None,
        trainable: str = "all",
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
        self.best_fit_kwargs = best_fit_kwargs
        self.seed = seed
        self.trainable = _validated_trainable(trainable)
        self.new_tokens: list | None = None

        self.output_model_path = output_model_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset_kwargs = dataset_kwargs or {}

        # populated by setup()
        self.dataset: PeptideDataset | None = None
        self.model: Any | None = None
        self.best_fit_info: dict | None = None

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
            base_model_name: my-org/prosit-intensity  # Hugging Face Hub repo id
            epochs: 20
            learning_rate: 0.0003
            trainable: new_tokens  # all (default), embedding or new_tokens
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

    @classmethod
    def from_dataset_and_model(
        cls,
        dataset: PeptideDataset,
        model,
        output_model_path: str = "./finetuned_model",
        epochs: int = 10,
        learning_rate: float = 1e-4,
        trainable: str = "all",
        new_tokens: list | None = None,
    ) -> "FineTunePipeline":
        """Create a ready-to-use pipeline from an already-built dataset and model.

        Useful when the dataset has been prepared and the model has been adapted
        (e.g. via :func:`~dlomix.models.load_and_adapt_pretrained_model`) outside
        the pipeline.  :meth:`setup` does **not** need to be called — the pipeline
        is immediately ready for :meth:`finetune` and :meth:`save`.

        Parameters
        ----------
        dataset:
            A processed :class:`~dlomix.data.PeptideDataset` (or subclass).
        model:
            A compiled or uncompiled model to fine-tune.
        output_model_path:
            Default save destination used by :meth:`save` when no path is given.
        epochs:
            Number of training epochs passed to :meth:`finetune`.
        learning_rate:
            Initial learning rate for the default Adam optimiser.
        trainable:
            What :meth:`finetune` trains: ``"all"`` (default), ``"embedding"`` or
            ``"new_tokens"``.
        new_tokens:
            The tokens added to the model's vocabulary, whose embedding rows
            ``trainable="new_tokens"`` trains. Required for that mode here, because
            the pipeline did not expand the vocabulary itself.
        """
        instance = cls.__new__(cls)
        # Config attributes — None/defaults since dataset + model are pre-built
        instance.finetune_dataset_path = None
        instance.base_model_name = None
        instance.base_model_weights_filepath = None
        instance.old_model_vocab = None
        instance.new_model_vocab = None
        instance.initialization_strategy = None
        instance.best_fit_kwargs = None
        instance.seed = None
        instance.trainable = _validated_trainable(trainable)
        instance.new_tokens = list(new_tokens) if new_tokens is not None else None
        instance.output_model_path = output_model_path
        instance.epochs = epochs
        instance.batch_size = getattr(dataset, "batch_size", 64)
        instance.learning_rate = learning_rate
        instance.dataset_kwargs = {}
        # Pre-populated — no setup() needed
        instance.dataset = dataset
        instance.model = model
        instance.best_fit_info = None
        return instance

    @classmethod
    def from_pretrained(
        cls,
        base,
        data,
        label_column: str = "intensities_raw",
        initialization_strategy: str = "random",
        best_fit_kwargs: dict | None = None,
        trainable: str = "all",
        seed: int = 42,
        output_model_path: str = "./finetuned_model",
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        revision: str | None = None,
        dataset_kwargs: dict | None = None,
    ) -> "FineTunePipeline":
        """Create a ready-to-use pipeline from a pretrained bundle and new data.

        The bundle's preprocessing settings (sequence column, sequence length,
        termini, features) build the dataset; the tokens of the data that the model
        does not know are added to its vocabulary, after the known ones; and their
        embeddings are initialized by ``initialization_strategy``. :meth:`setup`
        does not need to be called.

        This composes public steps, which can be called one by one to change any of
        them: :func:`~dlomix.data.find_new_tokens`,
        :func:`~dlomix.data.extend_alphabet`, the dataset class with
        ``base.preprocessor.dataset_settings``,
        :func:`~dlomix.models.adapt_pretrained_model` and
        :meth:`from_dataset_and_model`.

        Parameters
        ----------
        base:
            The pretrained model with its preprocessing: an
            :class:`~dlomix.pipelines.InferencePipeline`, the path of a saved one,
            or the Hugging Face Hub repo id of one (see ``revision``).
        data:
            The fine-tuning data: a Hugging Face ``DatasetDict`` with a ``train``
            split (and ``val`` / ``test``), or a single ``Dataset`` to be split with
            ``dataset_kwargs`` (e.g. ``{"val_ratio": 0.1}``). It needs the bundle's
            sequence and feature columns and ``label_column``.
        label_column:
            Column with the observed intensities.
        initialization_strategy:
            ``"random"``, ``"mean"`` or ``"best-fit"``, as in
            :func:`~dlomix.models.load_and_adapt_pretrained_model`.
        best_fit_kwargs:
            Options of the best-fit search, e.g. ``n_examples_for_eval``. The
            examples (the training split, or every row of a single ``Dataset``),
            the columns, the dataset settings and
            ``return_fit_info`` default to this pipeline's; the search is stored in
            ``best_fit_info``.
        trainable:
            What :meth:`finetune` trains: ``"all"`` (default), ``"embedding"`` or
            ``"new_tokens"``.
        seed:
            Seed of the embedding initialization and the best-fit sample.
        output_model_path, epochs, batch_size, learning_rate:
            As in the constructor.
        revision:
            Commit, branch or tag of the Hub repo, when ``base`` is a repo id.
        dataset_kwargs:
            Further arguments of :class:`~dlomix.data.FragmentIonIntensityDataset`,
            e.g. ``val_ratio``, ``dataset_columns_to_keep`` or ``shuffle``. The
            bundle's settings and the alphabet cannot be changed here.

        Attributes set: ``dataset``, ``model``, ``new_tokens``, ``best_fit_info``.
        """
        _require_tensorflow("from_pretrained")
        from .predictor import InferencePipeline

        if isinstance(base, (str, Path)):
            base = (
                InferencePipeline.load(str(base))
                if Path(base).exists()
                else InferencePipeline.from_pretrained(str(base), revision=revision)
            )
        settings = base.preprocessor.dataset_settings
        dataset_kwargs = dict(dataset_kwargs or {})
        fixed = sorted({"alphabet", "data_source", "data_format", *settings})
        if clash := sorted(set(fixed) & dataset_kwargs.keys()):
            raise ValueError(
                f"dataset_kwargs sets {clash}, which come from the pretrained bundle "
                "and the data. To change them, build the dataset and the model "
                "yourself and use from_dataset_and_model()."
            )

        # the alphabet is learned from the training and validation data only
        splits = (
            [data[name] for name in ("train", "val") if name in data]
            if isinstance(data, dict)
            else [data]
        )
        sequence_column = settings["sequence_column"]
        base_alphabet = dict(base.model.alphabet)
        new_tokens = find_new_tokens(
            (sequence for split in splits for sequence in split[sequence_column]),
            base_alphabet,
            with_termini=settings["with_termini"],
        )
        alphabet = extend_alphabet(base_alphabet, new_tokens)

        dataset = FragmentIonIntensityDataset(
            data_source=data,
            data_format="hf",
            label_column=label_column,
            alphabet=alphabet,
            batch_size=batch_size,
            **settings,
            **dataset_kwargs,
        )

        best_fit_info = None
        if initialization_strategy == "best-fit":
            search_settings = {
                key: value
                for key, value in settings.items()
                if key != "sequence_column"
            }
            best_fit_kwargs = {
                "new_hf_data": splits[0],
                "sequence_column": sequence_column,
                "label_column": label_column,
                "dataset_kwargs": search_settings,
                **(best_fit_kwargs or {}),
                "return_fit_info": True,
            }
        model = adapt_pretrained_model(
            base.model,
            new_alphabet=alphabet,
            initialization_strategy=initialization_strategy,
            random_seed=seed,
            best_fit_kwargs=best_fit_kwargs,
        )
        if isinstance(model, tuple):
            model, best_fit_info = model

        instance = cls.from_dataset_and_model(
            dataset,
            model,
            output_model_path=output_model_path,
            epochs=epochs,
            learning_rate=learning_rate,
            trainable=trainable,
            new_tokens=new_tokens,
        )
        instance.initialization_strategy = initialization_strategy
        instance.best_fit_kwargs = best_fit_kwargs
        instance.best_fit_info = best_fit_info
        instance.seed = seed
        return instance

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> FineTunePipeline:
        """Prepare the dataset and load / adapt the model.

        Must be called before :meth:`finetune`.  Returns *self* so calls can
        be chained::

            history = FineTunePipeline(...).setup().finetune()
        """
        # fail before the (potentially slow) dataset preparation
        _require_tensorflow("setup")
        logger.info("Setting up FineTunePipeline …")
        self._prepare_dataset()
        self._resolve_vocab()
        self._resolve_weights()
        self._load_and_adapt_model()
        logger.info("Setup complete.")
        return self

    def _prepare_dataset(self) -> None:
        logger.info("Loading dataset from '%s' …", self.finetune_dataset_path)
        dataset_kwargs = dict(self.dataset_kwargs)
        if self.new_model_vocab is not None:
            # encode the data with the model's vocabulary; a dataset left to learn
            # its own alphabet would assign other indices to the same tokens
            given = dataset_kwargs.setdefault("alphabet", self.new_model_vocab)
            if dict(given) != dict(self.new_model_vocab):
                raise ValueError(
                    "dataset_kwargs['alphabet'] differs from new_model_vocab; the "
                    "dataset must be encoded with the model's vocabulary. Pass only "
                    "new_model_vocab."
                )
        self.dataset = FragmentIonIntensityDataset(
            data_format="parquet",
            data_source=self.finetune_dataset_path,
            batch_size=self.batch_size,
            **dataset_kwargs,
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
        result = load_and_adapt_pretrained_model(
            model_path=self.base_model_weights_filepath,
            new_alphabet=self.new_model_vocab,
            old_alphabet=self.old_model_vocab,
            initialization_strategy=self.initialization_strategy,
            random_seed=self.seed,
            best_fit_kwargs=self.best_fit_kwargs,
        )
        # When best_fit_kwargs contains return_fit_info=True the function returns
        # (model, fit_info); otherwise it returns just the model.
        if isinstance(result, tuple):
            self.model, self.best_fit_info = result
        else:
            self.model = result

    def to_inference_pipeline(self):
        """Wrap the fine-tuned model in an :class:`~dlomix.pipelines.InferencePipeline`.

        The preprocessor is derived from ``self.dataset`` so the inference pipeline
        reproduces the exact training-time preprocessing.  The natural workflow after
        fine-tuning is::

            history = pipeline.setup().finetune()
            inference = pipeline.to_inference_pipeline()
            inference.save("path/to/bundle")          # or .push_to_hub(...)

        Raises
        ------
        RuntimeError
            If :meth:`setup` (or :meth:`from_dataset_and_model`) has not been called.
        """
        self._require_setup()
        from .predictor import InferencePipeline

        return InferencePipeline.from_model_and_dataset(self.model, self.dataset)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def finetune(
        self,
        loss_fn=None,
        metrics: list | None = None,
        callbacks: list | None = None,
        optimizer=None,
        trainable: str | None = None,
        keep_starting_model: bool = True,
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
        trainable:
            Which weights train; defaults to the pipeline's ``trainable``.

            - ``"all"``: the model as it is, i.e. every layer that is trainable.
            - ``"embedding"``: only the token embedding, all of its rows.
            - ``"new_tokens"``: only the embedding rows of the tokens added by the
              vocabulary expansion. Every other weight, including the rows of the
              known tokens, is exactly the same after training, so predictions
              for peptides without the new tokens do not change.

            The model's ``trainable`` flags are the same after training as before.
        keep_starting_model:
            If True (default), the validation loss of the model before training is
            recorded, and when no epoch has a lower one, the starting weights are
            restored: with little data, every epoch can be worse than the model
            fine-tuning started from, and ``EarlyStopping(restore_best_weights=True)``
            only returns the best trained epoch.

        Returns
        -------
        keras.callbacks.History
            The Keras training history object returned by ``model.fit``. With
            ``keep_starting_model`` it also has ``starting_val_loss`` and
            ``kept_model``: ``"starting"`` if the starting weights were restored,
            else ``"trained"``.

        Raises
        ------
        RuntimeError
            If :meth:`setup` has not been called yet.
        """
        self._require_setup()
        _require_tensorflow("finetune")

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

        with self._trainable_weights(trainable or self.trainable):
            self.model.compile(
                optimizer=resolved_optimizer,
                loss=resolved_loss,
                metrics=metrics or [],
            )

            if keep_starting_model:
                starting_weights = self.model.get_weights()
                starting_val_loss = self.model.evaluate(
                    self.dataset.tensor_val_data, verbose=0, return_dict=True
                )["loss"]

            history = self.model.fit(
                self.dataset.tensor_train_data,
                epochs=self.epochs,
                validation_data=self.dataset.tensor_val_data,
                callbacks=callbacks or [],
            )

        if keep_starting_model:
            history.starting_val_loss = starting_val_loss
            history.kept_model = "trained"
            # also when the loss became NaN: no epoch beat the starting model
            if not min(history.history["val_loss"]) < starting_val_loss:
                self.model.set_weights(starting_weights)
                history.kept_model = "starting"
                logger.warning(
                    "No epoch had a lower validation loss than the starting model "
                    "(%.6f): its weights were restored.",
                    starting_val_loss,
                )

        logger.info("Fine-tuning complete.")
        return history

    def _trainable_weights(self, trainable: str):
        """Context in which only the weights of the ``trainable`` mode can train."""
        if _validated_trainable(trainable) == "all":
            return contextlib.nullcontext()

        from ..models.model_utils import embedding_only_training

        rows = None
        if trainable == "new_tokens":
            rows = [self.model.alphabet[token] for token in self._resolve_new_tokens()]
        return embedding_only_training(self.model, rows=rows)

    def _resolve_new_tokens(self) -> list:
        """The tokens the vocabulary expansion added to the pretrained model's."""
        if self.new_tokens is None:
            if self.base_model_weights_filepath is None:
                raise ValueError(
                    "trainable='new_tokens' needs the tokens that are new to the "
                    "model: pass new_tokens=[...] to from_dataset_and_model()."
                )
            old_vocab = self.old_model_vocab
            if old_vocab is None:
                import tensorflow as tf

                from ..models.model_utils import get_alphabet_from_model

                old_vocab = get_alphabet_from_model(
                    tf.keras.models.load_model(self.base_model_weights_filepath)
                )
            self.new_tokens = [t for t in self.model.alphabet if t not in old_vocab]
        if not self.new_tokens:
            raise ValueError(
                "trainable='new_tokens': the model's vocabulary has no new tokens, "
                "so nothing would be trained."
            )
        return self.new_tokens

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, path: str | None = None, overwrite: bool = False) -> str:
        """Save the fine-tuned model weights to disk.

        Parameters
        ----------
        path:
            Destination file path.  Falls back to ``self.output_model_path``
            when omitted.  Keras 3 only writes to a path ending in ``.keras``
            (or ``.h5``), so a ``.keras`` suffix is appended when the path has
            neither; the returned value reflects the file actually written.
        overwrite:
            If False (default) raise :exc:`FileExistsError` when ``path``
            already exists, matching the behaviour of
            :meth:`~dlomix.pipelines.InferencePipeline.save`.

        Returns
        -------
        str
            The path the model was saved to.
        """
        self._require_setup()
        _require_tensorflow("save")
        destination = _ensure_keras_extension(path or self.output_model_path)
        if Path(destination).exists() and not overwrite:
            raise FileExistsError(
                f"'{destination}' already exists. Set overwrite=True to replace it."
            )
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
        model_label = self.base_model_name or (
            Path(self.base_model_weights_filepath).name
            if self.base_model_weights_filepath
            else "<provided>"
        )
        return (
            f"FineTunePipeline("
            f"model={model_label!r}, "
            f"epochs={self.epochs}, "
            f"lr={self.learning_rate}, "
            f"status={status!r})"
        )
