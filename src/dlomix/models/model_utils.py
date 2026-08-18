"""
TF/Keras Model utility functions for transferring embeddings for fine-tuning and transfer learning

This module provides utilities for adapting pre-trained models to new datasets,
particularly when vocabularies/alphabets differ between training and fine-tuning.
"""

import logging
from typing import Dict, Literal, Optional

import numpy as np
import tensorflow as tf

from dlomix.data import FragmentIonIntensityDataset

logger = logging.getLogger(__name__)


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
    initialization_strategy : {'random', 'mean'}, optional
        Strategy for initializing new token embeddings. Defaults to 'random'.
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
            Name of the column in new_hf_data containing the sequences.
        - label_column: str
            Name of the column in new_hf_data containing the labels for evaluation.
        - n_examples_for_eval: int
            Number of examples to use for evaluating each old token fit.
        - eval_metric: Callable
            Evaluation metric function that takes (labels, predictions) and returns a score.
        - return_fit_info: bool
            Whether to return detailed fit information for each new token (default: False).
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

    model = _load_model_with_custom_objects(model_path, custom_objects)

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
    model = expand_embedding_vocabulary(
        model=model,
        new_alphabet=new_alphabet,
        old_alphabet=old_alphabet,
        embedding_layer_name=embedding_layer_name,
        initialization_strategy=first_initialization_strategy,
        random_seed=random_seed,
    )

    if initialization_strategy == "best-fit":
        old_model = _load_model_with_custom_objects(model_path, custom_objects)

        best_fit_dict = _find_best_fit_tokens_for_new_tokens(
            new_hf_data=best_fit_kwargs.get("new_hf_data"),
            sequence_column=best_fit_kwargs.get("sequence_column"),
            label_column=best_fit_kwargs.get("label_column"),
            alphabet_old=old_alphabet,
            alphabet_new=new_alphabet,
            old_model=old_model,
            n_examples_for_eval=best_fit_kwargs.get("n_examples_for_eval"),
            eval_metric=best_fit_kwargs.get("eval_metric"),
            **best_fit_kwargs.get("dataset_kwargs", {}),
        )

        # best fit dict has the format
        # {'M[UNIMOD:35]': {'eval': 0.9156792, 'old_token': 'V', 'old_token_idx': 6}, 'C[UNIMOD:4]': {'eval': 0.9577802, 'old_token': '-', 'old_token_idx': 0}}

        old_weights = _get_embedding_layer(
            old_model, embedding_layer_name
        ).get_weights()[0]
        new_weights = (
            _get_embedding_layer(model, embedding_layer_name).get_weights()[0].copy()
        )

        for new_token, fit_info in best_fit_dict.items():
            if (
                not fit_info
            ):  # no examples found for this token — keep mean-init fallback
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
        _get_embedding_layer(model, embedding_layer_name).set_weights([new_weights])

        if best_fit_kwargs.get("return_fit_info", False):
            return model, best_fit_dict

    return model


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
        Strategy for initializing new token embeddings. Defaults to 'random'.
    random_seed : int, optional
        Random seed for reproducible initialization. Defaults to None.

    Returns
    -------
    tf.keras.Model
        Model with expanded embedding vocabulary.
    dict
        If initialization_strategy is 'best-fit' and return_fit_info is True, also returns a dictionary with best fit information for each new token.

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
        mean_embedding = np.mean(old_weights, axis=0)
        new_weights = np.tile(mean_embedding, (new_vocab_size, 1))
        logger.info("Initialized new embeddings with mean of existing embeddings")
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
    sequence_column,
    label_column,
    alphabet_old,
    alphabet_new,
    old_model,
    n_examples_for_eval=100,
    eval_metric=None,
    **dataset_kwargs,
):
    new_tokens = list(alphabet_new.keys() - alphabet_old.keys())
    best_fit_dict = {}
    best_sa = 0

    import datasets

    datasets.disable_progress_bar()

    if not isinstance(new_hf_data, datasets.Dataset):
        raise ValueError("new_hf_data must be a Hugging Face Dataset object.")

    for new in new_tokens:
        best_fit_dict[new] = {}
        best_sa = -np.inf  # ensure the first valid SA (even negative) is recorded

        filtered_data = new_hf_data.filter(lambda x: new in x[sequence_column])

        if len(filtered_data) == 0:
            continue
        if len(filtered_data) <= n_examples_for_eval:
            example_data = filtered_data
        else:
            example_data = filtered_data.take(n_examples_for_eval)

        for current_old_token, current_old_token_idx in alphabet_old.items():
            temp_alphabet = alphabet_old.copy()
            temp_alphabet.update({new: current_old_token_idx})
            test_data_current_token = FragmentIonIntensityDataset(
                data_format="hf",
                test_data_source=example_data,
                sequence_column=sequence_column,
                label_column=label_column,
                alphabet=temp_alphabet,
                **dataset_kwargs,
            )

            sa = []
            for inputs, labels in test_data_current_token.tensor_test_data:
                preds = old_model.predict(inputs, verbose=0)
                current_sa = 1 - eval_metric(labels, preds)
                sa.extend(current_sa)
            sa = np.median(sa)
            if sa > best_sa:
                best_sa = sa
                best_fit_dict[new]["eval"] = sa
                best_fit_dict[new]["old_token"] = current_old_token
                best_fit_dict[new]["old_token_idx"] = current_old_token_idx

    datasets.enable_progress_bar()

    return best_fit_dict


def download_remote_model_weights(model_name):
    # Download the model weights from a remote source (e.g., Hugging Face Hub, PRIDE, etc.)
    raise NotImplementedError(
        "Downloading remote model weights is not implemented yet."
    )
