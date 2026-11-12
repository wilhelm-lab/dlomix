"""
TF/Keras Model utility functions for transferring embeddings for fine-tuning and transfer learning

This module provides utilities for adapting pre-trained models to new datasets,
particularly when vocabularies/alphabets differ between training and fine-tuning.
"""

import contextlib
import logging
import warnings
from typing import Dict, Literal, Optional, Sequence

import keras
import numpy as np
import tensorflow as tf

from dlomix.data import FragmentIonIntensityDataset, sequence_tokens
from dlomix.data.processing.processors import unmodified_token

logger = logging.getLogger(__name__)


# The padding and unknown tokens carry no residue information, and their embeddings
# are not trained like a residue's: they never initialize a new token, neither as a
# best-fit candidate nor as part of the mean.
_NON_RESIDUE_TOKENS = ("-", "X")


def load_and_adapt_pretrained_model(
    model_path: str,
    new_alphabet: Dict[str, int],
    old_alphabet: Optional[Dict[str, int]] = None,
    embedding_layer_name: str = "embedding",
    initialization_strategy: Literal["random", "mean", "best-fit"] = "random",
    random_seed: Optional[int] = None,
    custom_objects: Optional[Dict] = None,
    best_fit_kwargs: Optional[Dict] = None,
) -> tf.keras.Model:
    """
    Load a pre-trained model and adapt it to a new vocabulary in one step.

    This is a convenience wrapper that combines model loading and vocabulary expansion.
    It's the recommended way to prepare a pre-trained model for fine-tuning on a dataset
    with a different vocabulary.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (.keras, or SavedModel directory).
    new_alphabet : Dict[str, int]
        Dictionary mapping tokens to indices in the new vocabulary.
    old_alphabet : Dict[str, int], optional
        Dictionary mapping tokens to indices in the original vocabulary.
        If None, attempts to extract from the loaded model. Defaults to None.
    embedding_layer_name : str, optional
        Name of the embedding layer to expand. Defaults to "embedding".
    initialization_strategy : {'random', 'mean', 'best-fit'}, optional
        Strategy for initializing new token embeddings: Glorot-uniform random
        values, the mean of the pretrained residue embeddings (the padding and
        unknown tokens are left out), or ('best-fit') the
        embedding of the pretrained token whose substitution predicts the new
        token's spectra best (requires ``best_fit_kwargs``). Defaults to 'random'.
    random_seed : int, optional
        Random seed for reproducible initialization. Defaults to None.
    custom_objects : Dict, optional
        Custom objects needed for loading the model (e.g., custom layers, losses).
        Defaults to None.
    best_fit_kwargs : Dict, optional
        Additional arguments for best-fit initialization strategy. Required if initialization_strategy is 'best-fit'. Should include:
        - new_hf_data: Hugging Face
            Dataset containing sequences with new tokens for evaluation.
        - sequence_column: str
            Name of the column in new_hf_data containing the sequences
            (default: "modified_sequence").
        - label_column: str
            Name of the column in new_hf_data containing the labels for evaluation
            (default: "intensities_raw").
        - n_examples_for_eval: int
            The most spectra searched per new token (default: 100). Of more, a
            random sample is taken, seeded with ``random_seed``.
        - min_examples_for_eval: int
            How many spectra with the token as their only new token are enough to
            search on those alone (default: 20; at most ``n_examples_for_eval``).
            With fewer, spectra that also carry other new tokens are added, up to
            ``n_examples_for_eval`` in total, the other new tokens encoded as their
            unmodified residue. 0 never adds them.
        - eval_metric: Callable
            Loss function that takes (labels, predictions) and returns one value per
            sample; the fit is scored as 1 - its median (default:
            ``masked_spectral_distance``).
        - return_fit_info: bool
            Whether to also return the search per new token (default: False): the
            pick, every candidate's score, the number of spectra searched and left
            out, which examples were used, and whether the token fell back to the
            mean initialization for lack of examples.
        - dataset_kwargs: Dict
            Additional keyword arguments to pass when creating the FragmentIonIntensityDataset for evaluation.


    Returns
    -------
    tf.keras.Model
        The loaded model with expanded embedding vocabulary, ready for fine-tuning.

    Raises
    ------
    ValueError
        If the model cannot be loaded or vocabulary expansion fails.

    Examples
    --------
    >>> # Simple usage - alphabet auto-extracted from model
    >>> new_alphabet = {'A': 0, 'C': 1, ..., 'M(ox)': 20}
    >>> model = load_and_adapt_pretrained_model(
    ...     model_path='pretrained_prosit.keras',
    ...     new_alphabet=new_alphabet,
    ...     initialization_strategy='mean'
    ... )
    >>>
    >>> # Compile and fine-tune
    >>> model.compile(optimizer='adam', loss='mse')
    >>> model.fit(new_dataset, epochs=10)

    Notes
    -----
    After loading and adapting, you should compile the model before training.
    The optimizer state from the pre-trained model is not preserved.
    """
    logger.info(f"Loading pre-trained model from: {model_path}")

    return adapt_pretrained_model(
        _load_model_with_custom_objects(model_path, custom_objects),
        new_alphabet=new_alphabet,
        old_alphabet=old_alphabet,
        embedding_layer_name=embedding_layer_name,
        initialization_strategy=initialization_strategy,
        random_seed=random_seed,
        best_fit_kwargs=best_fit_kwargs,
    )


def adapt_pretrained_model(
    model: tf.keras.Model,
    new_alphabet: Dict[str, int],
    old_alphabet: Optional[Dict[str, int]] = None,
    embedding_layer_name: str = "embedding",
    initialization_strategy: Literal["random", "mean", "best-fit"] = "random",
    random_seed: Optional[int] = None,
    best_fit_kwargs: Optional[Dict] = None,
) -> tf.keras.Model:
    """
    Adapt a loaded pre-trained model to a new vocabulary.

    The step of :func:`load_and_adapt_pretrained_model` after loading, for a model
    that is already in memory, e.g. ``InferencePipeline.from_pretrained(...).model``.
    ``model`` is left unchanged; a new model is returned. The arguments and the
    return value are those of :func:`load_and_adapt_pretrained_model`.
    """
    # Extract old alphabet if not provided
    if old_alphabet is None:
        old_alphabet = get_alphabet_from_model(model)
        if old_alphabet is None:
            raise ValueError(
                "Could not extract alphabet from loaded model and old_alphabet was not provided. "
                "Please provide old_alphabet explicitly."
            )
        logger.info(f"Extracted alphabet from model: {len(old_alphabet)} tokens")

    first_initialization_strategy = initialization_strategy
    if initialization_strategy == "best-fit":
        if best_fit_kwargs is None:
            best_fit_kwargs = {}
        logger.info("Using best-fit initialization strategy for new embeddings")
        first_initialization_strategy = "mean"  # Fallback to mean for new tokens

    # Expand the vocabulary
    adapted = expand_embedding_vocabulary(
        model=model,
        new_alphabet=new_alphabet,
        old_alphabet=old_alphabet,
        embedding_layer_name=embedding_layer_name,
        initialization_strategy=first_initialization_strategy,
        random_seed=random_seed,
    )

    if initialization_strategy == "best-fit":
        old_model = model  # unchanged: the expansion builds a new model

        # forward only the options given, so the search's defaults apply otherwise
        search_kwargs = {
            key: best_fit_kwargs[key]
            for key in (
                "sequence_column",
                "label_column",
                "n_examples_for_eval",
                "min_examples_for_eval",
                "eval_metric",
            )
            if best_fit_kwargs.get(key) is not None
        }
        best_fit_dict = _find_best_fit_tokens_for_new_tokens(
            new_hf_data=best_fit_kwargs.get("new_hf_data"),
            alphabet_old=old_alphabet,
            alphabet_new=new_alphabet,
            old_model=old_model,
            random_seed=random_seed,
            **search_kwargs,
            **best_fit_kwargs.get("dataset_kwargs", {}),
        )

        old_weights = _get_embedding_layer(
            old_model, embedding_layer_name
        ).get_weights()[0]
        new_weights = (
            _get_embedding_layer(adapted, embedding_layer_name).get_weights()[0].copy()
        )

        for new_token, fit_info in best_fit_dict.items():
            if fit_info["old_token"] is None:  # no examples: keep the mean fallback
                logger.warning(
                    "No evaluation examples found for new token '%s'; "
                    "keeping mean-initialization fallback.",
                    new_token,
                )
                continue
            new_idx = new_alphabet[new_token]
            old_idx = fit_info["old_token_idx"]

            # Copy the embeddings in numpy to avoid issues with TensorFlow variable assignment and devices
            new_weights[new_idx] = old_weights[old_idx]

            logger.info(
                f"Best-fit transferred embedding for new token '{new_token}' from old token '{fit_info['old_token']}' with eval={fit_info['eval']:.4f}"
            )

        # set the embedding weights one final time to apply all updates
        _get_embedding_layer(adapted, embedding_layer_name).set_weights([new_weights])

        if best_fit_kwargs.get("return_fit_info", False):
            return adapted, best_fit_dict

    return adapted


def expand_embedding_vocabulary(
    model: tf.keras.Model,
    new_alphabet: Dict[str, int],
    old_alphabet: Optional[Dict[str, int]] = None,
    embedding_layer_name: str = "embedding",
    initialization_strategy: Literal["random", "mean"] = "random",
    random_seed: Optional[int] = None,
) -> tf.keras.Model:
    """
    Expand embedding vocabulary by transferring weights to a new model or in-place modification.

    This function creates an expanded vocabulary size, transfers embeddings for common
    tokens from the old vocabulary, and initializes embeddings for new tokens. For
    subclassed models (dlomix models), it performs safe in-place modification. For
    functional/sequential models, it creates a new model to ensure proper graph construction.

    Parameters
    ----------
    model : tf.keras.Model
        The model with the embedding layer to expand.
    new_alphabet : Dict[str, int]
        Dictionary mapping tokens to indices in the new vocabulary.
    old_alphabet : Dict[str, int], optional
        Dictionary mapping tokens to indices in the original vocabulary.
        If None, attempts to extract from model.alphabet. Defaults to None.
    embedding_layer_name : str, optional
        Name of the embedding layer to expand. Defaults to "embedding".
    initialization_strategy : {'random', 'mean'}, optional
        Strategy for initializing new token embeddings: Glorot-uniform random
        values or the mean of the existing residue embeddings (the padding and
        unknown tokens are left out). Defaults to 'random'.
        ('best-fit' is available through :func:`load_and_adapt_pretrained_model`.)
    random_seed : int, optional
        Random seed for reproducible initialization. Defaults to None.

    Returns
    -------
    tf.keras.Model
        Model with expanded embedding vocabulary.

    Raises
    ------
    ValueError
        If old_alphabet cannot be determined or expansion fails.
    AttributeError
        If the specified embedding layer is not found.

    Examples
    --------
    >>> # Basic usage
    >>> new_alphabet = {'A': 0, 'C': 1, ..., 'M(ox)': 20}
    >>> adapted_model = expand_embedding_vocabulary(
    ...     model=pretrained_model,
    ...     new_alphabet=new_alphabet,
    ...     initialization_strategy='mean'
    ... )
    """

    # Extract old alphabet if not provided
    if old_alphabet is None:
        if hasattr(model, "alphabet"):
            old_alphabet = model.alphabet
            logger.info(
                f"Extracted old alphabet from model.alphabet with {len(old_alphabet)} tokens"
            )
        else:
            raise ValueError(
                "old_alphabet is None and model does not have an 'alphabet' attribute. "
                "Please provide old_alphabet explicitly."
            )

    # Get the embedding layer and extract weights
    embedding_layer = _get_embedding_layer(model, embedding_layer_name)

    if not embedding_layer.weights:
        raise ValueError(
            f"Embedding layer '{embedding_layer_name}' has no weights yet, so there is "
            "nothing to transfer. Build the model before expanding its vocabulary, by "
            "calling model.build(input_shape) or running one forward pass."
        )

    old_weights = embedding_layer.get_weights()[0]
    embedding_dim = old_weights.shape[1]
    old_vocab_size = old_weights.shape[0]
    new_vocab_size = len(new_alphabet)

    logger.info(
        f"Expanding vocabulary: old_vocab={old_vocab_size}, new_vocab={new_vocab_size}, "
        f"embedding_dim={embedding_dim}"
    )

    # Keras 3 forbids attaching new state (variables or sub-layers) to a model
    # that is already built, so the embedding cannot be swapped in place with
    # setattr any more. Instead rebuild the model from its own config with the
    # new alphabet and carry every other weight across.
    new_model = _rebuild_model_with_alphabet(model, new_alphabet)

    # Each model applies its own vocabulary convention when sizing the embedding
    # (len(alphabet), or len(alphabet) + 1 / + 2), so take the row count from the
    # layer that will actually hold the weights rather than assuming it.
    new_embedding_layer = _get_embedding_layer(new_model, embedding_layer_name)
    target_vocab_size = int(new_embedding_layer.weights[0].shape[0])

    new_embedding_weights = _create_expanded_embedding_weights(
        old_weights=old_weights,
        old_alphabet=old_alphabet,
        new_alphabet=new_alphabet,
        initialization_strategy=initialization_strategy,
        random_seed=random_seed,
        target_vocab_size=target_vocab_size,
    )

    _transfer_weights(
        old_model=model,
        new_model=new_model,
        embedding_layer_name=embedding_layer_name,
        new_embedding_weights=new_embedding_weights,
    )

    logger.info("Vocabulary expansion completed successfully")
    return new_model


def _rebuild_model_with_alphabet(
    model: tf.keras.Model, new_alphabet: Dict[str, int]
) -> tf.keras.Model:
    """Recreate ``model`` from its config with ``new_alphabet``, built to the same shape.

    The returned model has freshly initialized weights; the caller is responsible
    for transferring them across.
    """
    config = model.get_config()
    if "alphabet" not in config:
        raise ValueError(
            f"{type(model).__name__}.get_config() does not expose an 'alphabet' entry, "
            "so its vocabulary cannot be expanded. Vocabulary expansion requires a model "
            "that serializes its alphabet (PrositIntensityPredictor, "
            "PrositRetentionTimePredictor, ChargeStatePredictor, ...)."
        )
    config["alphabet"] = dict(new_alphabet)
    new_model = type(model).from_config(config)

    build_config = model.get_build_config() or {}
    if not build_config.get("input_shape"):
        raise ValueError(
            "The model must be built before its vocabulary can be expanded, so that the "
            "rebuilt model can be given the same input shape. Call model.build(input_shape) "
            "or run one forward pass first."
        )
    new_model.build_from_config(build_config)
    return new_model


def _transfer_weights(
    old_model: tf.keras.Model,
    new_model: tf.keras.Model,
    embedding_layer_name: str,
    new_embedding_weights: np.ndarray,
) -> None:
    """Copy every weight from ``old_model`` to ``new_model`` except the embedding.

    The two models are the same class built from the same config, so their
    ``weights`` lists line up positionally. Names cannot be used for matching:
    Keras appends a global counter to auto-generated layer names, so the same
    layer is called ``sequential_3`` in one instance and ``sequential_9`` in the
    next. Every pair is shape-checked, and the embedding receives
    ``new_embedding_weights`` instead of a copy.
    """
    old_weights, new_weights = old_model.weights, new_model.weights
    if len(old_weights) != len(new_weights):
        raise ValueError(
            f"The rebuilt model has {len(new_weights)} weight tensors but the original has "
            f"{len(old_weights)}. This usually means get_config() omits a parameter that "
            "changes the architecture."
        )

    embedding_variable_ids = {
        id(variable)
        for variable in _get_embedding_layer(new_model, embedding_layer_name).weights
    }

    transferred = 0
    for old_weight, new_weight in zip(old_weights, new_weights):
        if id(new_weight) in embedding_variable_ids:
            new_weight.assign(new_embedding_weights)
            continue

        if tuple(old_weight.shape) != tuple(new_weight.shape):
            raise ValueError(
                f"Shape mismatch while transferring '{new_weight.path}': the original weight "
                f"is {tuple(old_weight.shape)} but the rebuilt one is {tuple(new_weight.shape)}. "
                "Only the embedding is expected to change shape during vocabulary expansion."
            )

        new_weight.assign(old_weight)
        transferred += 1

    logger.info(
        f"Transferred {transferred} non-embedding weight tensors to the rebuilt model"
    )


def _create_expanded_embedding_weights(
    old_weights: np.ndarray,
    old_alphabet: Dict[str, int],
    new_alphabet: Dict[str, int],
    initialization_strategy: str,
    random_seed: Optional[int],
    target_vocab_size: Optional[int] = None,
) -> np.ndarray:
    """Create expanded embedding weights matrix with transferred and initialized embeddings.

    ``target_vocab_size`` is the number of rows the matrix must have. It defaults
    to ``len(new_alphabet)``, but models that reserve extra slots (e.g. for
    padding or termini tokens) size their embedding larger than the alphabet.
    """

    embedding_dim = old_weights.shape[1]
    new_vocab_size = (
        len(new_alphabet) if target_vocab_size is None else target_vocab_size
    )

    max_index = max(new_alphabet.values(), default=-1)
    if max_index >= new_vocab_size:
        raise ValueError(
            f"Alphabet index {max_index} is out of range for an embedding with "
            f"{new_vocab_size} rows. The alphabet indices must fit the embedding size."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize new embeddings
    if initialization_strategy == "mean":
        residue_rows = [
            idx
            for token, idx in old_alphabet.items()
            if token not in _NON_RESIDUE_TOKENS
        ]
        if not residue_rows:
            raise ValueError(
                "initialization_strategy='mean' needs at least one token other than "
                f"{_NON_RESIDUE_TOKENS} in the old alphabet to average."
            )
        mean_embedding = np.mean(old_weights[residue_rows], axis=0)
        new_weights = np.tile(mean_embedding, (new_vocab_size, 1))
        logger.info(
            "Initialized new embeddings with the mean of the residue embeddings"
        )
    elif initialization_strategy == "random":
        limit = np.sqrt(6.0 / (new_vocab_size + embedding_dim))
        new_weights = np.random.uniform(
            -limit, limit, size=(new_vocab_size, embedding_dim)
        ).astype(np.float32)
        logger.info("Initialized new embeddings with random Xavier uniform")
    else:
        raise ValueError(f"Unknown initialization_strategy: {initialization_strategy}")

    # Transfer common embeddings
    transferred_count = 0
    new_token_count = 0

    for token, new_idx in new_alphabet.items():
        if token in old_alphabet:
            old_idx = old_alphabet[token]
            new_weights[new_idx] = old_weights[old_idx]
            transferred_count += 1
        else:
            new_token_count += 1

    logger.info(
        f"Transfer complete: {transferred_count} common embeddings transferred, "
        f"{new_token_count} new tokens initialized"
    )

    return new_weights


def get_alphabet_from_model(model: tf.keras.Model) -> Optional[Dict[str, int]]:
    """
    Extract the alphabet/vocabulary from a model's configuration.

    This utility function retrieves the alphabet dictionary from a dlomix model
    that stores it in its config (e.g., PrositIntensityPredictor).

    Parameters
    ----------
    model : tf.keras.Model
        The model to extract the alphabet from.

    Returns
    -------
    Dict[str, int] or None
        The alphabet dictionary mapping tokens to indices, or None if not found.

    Examples
    --------
    >>> model = tf.keras.models.load_model('model.keras')
    >>> alphabet = get_alphabet_from_model(model)
    >>> print(f"Vocabulary size: {len(alphabet)}")
    """
    if hasattr(model, "alphabet"):
        return dict(model.alphabet)

    if hasattr(model, "get_config"):
        try:
            config = model.get_config()
            if "alphabet" in config:
                return dict(config["alphabet"])
        except Exception as e:
            logger.warning(f"Could not extract alphabet from model config: {e}")

    return None


def _get_embedding_layer(
    model: tf.keras.Model, embedding_layer_name: str
) -> tf.keras.layers.Layer:
    """Get an embedding layer via Python attribute first, then Keras name registry.

    Using the attribute path (getattr) works even for models whose Keras layer
    name string differs from the attribute name (e.g. models saved before a
    layer rename).  Falling back to get_layer() handles functional models where
    no matching attribute exists.
    """
    if hasattr(model, embedding_layer_name):
        return getattr(model, embedding_layer_name)
    try:
        return model.get_layer(embedding_layer_name)
    except ValueError as e:
        raise AttributeError(
            f"Embedding layer '{embedding_layer_name}' not found in model. "
            f"Available layers: {[layer.name for layer in model.layers]}"
        ) from e


class _KeepRows(keras.constraints.Constraint):
    """Applied after every optimizer step: all rows but ``trainable_rows`` go back
    to their ``start`` values, so only these rows learn."""

    def __init__(self, start: np.ndarray, trainable_rows: Sequence[int]):
        self.start = np.array(start)
        self.trainable = np.zeros((len(self.start), 1), dtype=bool)
        self.trainable[list(trainable_rows)] = True

    def __call__(self, w):
        return keras.ops.where(self.trainable, w, self.start)


@contextlib.contextmanager
def embedding_only_training(
    model: tf.keras.Model,
    rows: Optional[Sequence[int]] = None,
    embedding_layer_name: str = "embedding",
):
    """Within this context, only the model's token embedding is trainable.

    Every other layer is frozen individually; in Keras 3, ``model.trainable = False``
    followed by ``model.embedding.trainable = True`` would leave no trainable weights
    at all. With ``rows``, only these rows of the embedding change during training
    (e.g. the rows of tokens added by a vocabulary expansion); all other rows keep
    their values exactly. Compile the model inside the context. On exit the layers'
    ``trainable`` flags are as they were before.
    """
    if not model.trainable:
        raise ValueError(
            "The model itself is frozen (model.trainable is False), so none of its "
            "layers can train. Set model.trainable = True first."
        )
    embedding = _get_embedding_layer(model, embedding_layer_name)
    if rows is not None and len(rows) == 0:
        raise ValueError("`rows` is empty: no embedding row would be trained.")

    # all nested layers, parents before their children, so that restoring each
    # layer's own flag in this order gives back the exact previous state
    flags = [(layer, layer.trainable) for layer in model._flatten_layers(False)]
    try:
        for layer in model.layers:
            layer.trainable = layer is embedding
        if rows is not None:
            # on the variable, not the layer: it is not saved with the model
            embedding.embeddings.constraint = _KeepRows(
                embedding.embeddings.numpy(), rows
            )
        yield
    finally:
        embedding.embeddings.constraint = None
        for layer, trainable in flags:
            layer.trainable = trainable


def _load_model_with_custom_objects(
    model_path: str, custom_objects: Optional[Dict]
) -> tf.keras.Model:
    # Load the model
    try:
        if custom_objects is not None:
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects
            )
        else:
            model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}") from e
    return model


def _find_best_fit_tokens_for_new_tokens(
    new_hf_data,
    alphabet_old,
    alphabet_new,
    old_model,
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    n_examples_for_eval=100,
    min_examples_for_eval=20,
    eval_metric=None,
    random_seed=None,
    **dataset_kwargs,
):
    """Search, for every new token, the pretrained token that predicts its spectra best.

    A candidate is scored by encoding the new token as the candidate and taking
    ``1 - median(eval_metric)`` of the pretrained model over the token's examples.
    Every pretrained token but padding and unknown is a candidate; the first of
    equal scores wins.

    Two numbers decide which spectra a new token is searched on:

    - ``n_examples_for_eval`` is the most spectra searched; of more, a random sample
      (``random_seed``) is taken.
    - ``min_examples_for_eval`` is how many spectra with the token as their only new
      token are enough (at most ``n_examples_for_eval``, which is all the search
      would use). With that many, only such spectra are searched, so that no
      candidate is scored on inputs with other, still unknown tokens. With fewer,
      all of them are kept and spectra that also carry other new tokens fill up to
      ``n_examples_for_eval``, the other new tokens encoded as their unmodified
      residue (as the inference preprocessor encodes unseen modifications).

    Returns one dict per new token:

    - ``old_token``, ``old_token_idx``, ``eval``: the pick and its score, or None
      when the token has no examples; it then keeps its mean initialization.
    - ``initialization``: ``"best-fit"`` or ``"mean"``.
    - ``scores``: every candidate's score.
    - ``examples``: ``"sole-token"``, ``"unmodified-fallback"`` or None.
    - ``n_with_token``: spectra containing the token; ``n_left_out``: those of them
      excluded for carrying other new tokens; ``n_searched``: spectra scored.
    """
    import datasets

    if not isinstance(new_hf_data, datasets.Dataset):
        raise ValueError("new_hf_data must be a Hugging Face Dataset object.")
    if n_examples_for_eval < 1 or min_examples_for_eval < 0:
        raise ValueError(
            "n_examples_for_eval must be at least 1 and min_examples_for_eval at "
            f"least 0, got {n_examples_for_eval} and {min_examples_for_eval}."
        )
    enough_sole_token = min(min_examples_for_eval, n_examples_for_eval)

    def sample(rows, size):
        # the rows Dataset.shuffle(seed=random_seed) would put first
        if len(rows) <= size:
            return rows
        order = np.random.default_rng(random_seed).permutation(len(rows))
        return [rows[i] for i in order[:size]]

    if eval_metric is None:
        from dlomix.losses import masked_spectral_distance

        eval_metric = masked_spectral_distance

    new_tokens = list(alphabet_new.keys() - alphabet_old.keys())
    candidates = {
        token: idx
        for token, idx in alphabet_old.items()
        if token not in _NON_RESIDUE_TOKENS
    }

    # the other new tokens as their unmodified residue, where the model knows it
    unmodified_fallback = {
        token: alphabet_old[unmodified_token(token)]
        for token in new_tokens
        if unmodified_token(token) in alphabet_old
    }

    new_tokens_per_row = [
        set(sequence_tokens(sequence)) & set(new_tokens)
        for sequence in new_hf_data[sequence_column]
    ]

    def score(examples, alphabet):
        # throwaway dataset, built once per candidate: its warnings only repeat noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            data = FragmentIonIntensityDataset(
                data_format="hf",
                test_data_source=examples,
                sequence_column=sequence_column,
                label_column=label_column,
                alphabet=alphabet,
                **dataset_kwargs,
            )
        per_spectrum = [
            1 - np.asarray(eval_metric(labels, old_model.predict(inputs, verbose=0)))
            for inputs, labels in data.tensor_test_data
        ]
        return float(np.median(np.concatenate([s.reshape(-1) for s in per_spectrum])))

    datasets.disable_progress_bar()
    try:
        fit_info = {}
        for new in new_tokens:
            with_token = [i for i, row in enumerate(new_tokens_per_row) if new in row]
            sole_token = [i for i in with_token if new_tokens_per_row[i] == {new}]
            with_others = [i for i in with_token if new_tokens_per_row[i] != {new}]

            use_fallback = bool(with_others) and len(sole_token) < enough_sole_token
            base_alphabet = dict(alphabet_old)
            if use_fallback:
                # every spectrum with the token alone, filled up with the others
                rows = sole_token + sample(
                    with_others, n_examples_for_eval - len(sole_token)
                )
                base_alphabet.update(unmodified_fallback)
            else:
                rows = sample(sole_token, n_examples_for_eval)
            examples = new_hf_data.select(sorted(rows))

            scores = {}
            if len(examples) > 0:
                scores = {
                    candidate: score(examples, {**base_alphabet, new: candidate_idx})
                    for candidate, candidate_idx in candidates.items()
                }
            # max returns the first of equal scores
            best = max(scores, key=scores.get) if scores else None

            fit_info[new] = {
                "old_token": best,
                "old_token_idx": candidates[best] if best else None,
                "eval": scores[best] if best else None,
                "initialization": "best-fit" if best else "mean",
                "scores": scores,
                "examples": (
                    None
                    if not scores
                    else "unmodified-fallback" if use_fallback else "sole-token"
                ),
                "n_with_token": len(with_token),
                "n_left_out": 0 if use_fallback else len(with_others),
                "n_searched": len(examples),
            }
    finally:
        datasets.enable_progress_bar()

    return fit_info


def download_remote_model_weights(
    model_name: str, revision: Optional[str] = None
) -> str:
    """Download a TensorFlow model from the Hugging Face Hub and return its local path.

    ``model_name`` is the repo id of a DLOmix inference bundle, as written by
    :meth:`~dlomix.pipelines.InferencePipeline.push_to_hub` on the TensorFlow
    backend; its ``model.keras`` file is downloaded (and cached by
    ``huggingface_hub``).

    Parameters
    ----------
    model_name : str
        Hub repo id, e.g. ``"my-org/prosit-intensity"``.
    revision : str, optional
        Branch, tag or commit to download. Defaults to the main branch.

    Returns
    -------
    str
        Local path of the downloaded ``model.keras`` file, which
        :func:`load_and_adapt_pretrained_model` accepts as ``model_path``.
    """
    from huggingface_hub import hf_hub_download

    # the file name InferencePipeline.save writes for TensorFlow models
    # (dlomix.pipelines.predictor.TF_MODEL_FILE; not imported, pipelines imports models)
    try:
        return hf_hub_download(
            repo_id=model_name, filename="model.keras", revision=revision
        )
    except Exception as e:
        raise ValueError(
            f"Could not download 'model.keras' from the Hugging Face Hub repo "
            f"'{model_name}'. base_model_name must be the repo id of a DLOmix "
            f"inference bundle saved on the TensorFlow backend "
            f"(InferencePipeline.push_to_hub): {e}"
        ) from e
