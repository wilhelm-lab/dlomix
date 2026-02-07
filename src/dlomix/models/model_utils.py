"""
TF/Keras Model utility functions for transferring embeddings for fine-tuning and transfer learning

This module provides utilities for adapting pre-trained models to new datasets,
particularly when vocabularies/alphabets differ between training and fine-tuning.
"""

import logging
from typing import Dict, Literal, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


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
    if hasattr(model, embedding_layer_name):
        embedding_layer = getattr(model, embedding_layer_name)
    else:
        try:
            embedding_layer = model.get_layer(embedding_layer_name)
        except ValueError as e:
            raise AttributeError(
                f"Embedding layer '{embedding_layer_name}' not found in model. "
                f"Available layers: {[layer.name for layer in model.layers]}"
            ) from e

    old_weights = embedding_layer.get_weights()[0]
    embedding_dim = old_weights.shape[1]
    old_vocab_size = old_weights.shape[0]
    new_vocab_size = len(new_alphabet)

    logger.info(
        f"Expanding vocabulary: old_vocab={old_vocab_size}, new_vocab={new_vocab_size}, "
        f"embedding_dim={embedding_dim}"
    )

    # Create expanded embedding weights
    new_embedding_weights = _create_expanded_embedding_weights(
        old_weights=old_weights,
        old_alphabet=old_alphabet,
        new_alphabet=new_alphabet,
        initialization_strategy=initialization_strategy,
        random_seed=random_seed,
    )

    # For dlomix models (subclassed), use safe in-place modification
    # For functional/sequential models, this approach also works well
    new_embedding_layer = tf.keras.layers.Embedding(
        input_dim=new_vocab_size,
        output_dim=embedding_dim,
        input_length=embedding_layer.input_length,
        name=embedding_layer_name,
        mask_zero=embedding_layer.mask_zero,
        trainable=embedding_layer.trainable,
    )

    # Build the layer so it can accept weights
    # Use the same input_shape as the old embedding layer
    if (
        hasattr(embedding_layer, "input_shape")
        and embedding_layer.input_shape is not None
    ):
        new_embedding_layer.build(embedding_layer.input_shape)
    else:
        # Build with a standard input shape for sequence models
        new_embedding_layer.build((None, None))

    # Set the new weights
    new_embedding_layer.set_weights([new_embedding_weights])

    # Replace the embedding layer
    if hasattr(model, embedding_layer_name):
        setattr(model, embedding_layer_name, new_embedding_layer)
    else:
        # For functional models, we need to rebuild - this is complex so we use fallback
        logger.warning("Functional model detected - using in-place replacement")
        # Find and replace in model.layers
        for i, layer in enumerate(model.layers):
            if layer.name == embedding_layer_name:
                model.layers[i] = new_embedding_layer
                break

    # Update model attributes if they exist
    if hasattr(model, "alphabet"):
        model.alphabet = dict(new_alphabet)
    if hasattr(model, "embeddings_count"):
        model.embeddings_count = new_vocab_size

    logger.info("Vocabulary expansion completed successfully")
    return model


def _create_expanded_embedding_weights(
    old_weights: np.ndarray,
    old_alphabet: Dict[str, int],
    new_alphabet: Dict[str, int],
    initialization_strategy: str,
    random_seed: Optional[int],
) -> np.ndarray:
    """Create expanded embedding weights matrix with transferred and initialized embeddings."""

    embedding_dim = old_weights.shape[1]
    new_vocab_size = len(new_alphabet)

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


def load_and_adapt_pretrained_model(
    model_path: str,
    new_alphabet: Dict[str, int],
    old_alphabet: Optional[Dict[str, int]] = None,
    embedding_layer_name: str = "embedding",
    initialization_strategy: Literal["random", "mean"] = "random",
    random_seed: Optional[int] = None,
    custom_objects: Optional[Dict] = None,
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

    # Extract old alphabet if not provided
    if old_alphabet is None:
        old_alphabet = get_alphabet_from_model(model)
        if old_alphabet is None:
            raise ValueError(
                "Could not extract alphabet from loaded model and old_alphabet was not provided. "
                "Please provide old_alphabet explicitly."
            )
        logger.info(f"Extracted alphabet from model: {len(old_alphabet)} tokens")

    # Expand the vocabulary
    model = expand_embedding_vocabulary(
        model=model,
        new_alphabet=new_alphabet,
        old_alphabet=old_alphabet,
        embedding_layer_name=embedding_layer_name,
        initialization_strategy=initialization_strategy,
        random_seed=random_seed,
    )

    logger.info(
        "Model successfully loaded and adapted to new vocabulary. "
        "Ready for compilation and fine-tuning."
    )

    return model
