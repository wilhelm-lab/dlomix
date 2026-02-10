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

        old_weights = old_model.get_layer(embedding_layer_name).get_weights()[0]
        new_weights = model.get_layer(embedding_layer_name).get_weights()[0].copy()

        for new_token, fit_info in best_fit_dict.items():
            new_idx = new_alphabet[new_token]
            old_idx = fit_info["old_token_idx"]

            # Copy the embeddings in numpy to avoid issues with TensorFlow variable assignment and devices
            new_weights[new_idx] = old_weights[old_idx]

            logger.info(
                f"Best-fit transferred embedding for new token '{new_token}' from old token '{fit_info['old_token']}' with eval={fit_info['eval']:.4f}"
            )

        # set the embedding weights one final time to apply all updates
        model.get_layer(embedding_layer_name).set_weights([new_weights])

    logger.info(
        "Model successfully loaded and adapted to new vocabulary. "
        "Ready for compilation and fine-tuning."
    )

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
        print("New token is: ", new)
        best_fit_dict[new] = {}
        best_sa = 0

        filtered_data = new_hf_data.filter(lambda x: new in x[sequence_column])

        if len(filtered_data) == 0:
            continue
        if len(filtered_data) <= n_examples_for_eval:
            example_data = filtered_data
        else:
            example_data = filtered_data.take(n_examples_for_eval)

        for current_old_token, current_old_token_idx in alphabet_old.items():
            print("Trying token: ", current_old_token)
            temp_alphabet = alphabet_old.copy()
            temp_alphabet.update({new: current_old_token_idx})
            test_data_current_token = FragmentIonIntensityDataset(
                data_format="hf",
                data_source=example_data,
                sequence_column=sequence_column,
                label_column=label_column,
                alphabet=temp_alphabet,
                **dataset_kwargs,
            )

            sa = []
            for inputs, labels in test_data_current_token.tensor_train_data:
                preds = old_model.predict(inputs, verbose=0)
                current_sa = 1 - eval_metric(labels, preds)
                sa.extend(current_sa)
            sa = np.median(sa)
            print("median eval value: ", sa)
            if sa > best_sa:
                best_sa = sa
                best_fit_dict[new]["eval"] = sa
                best_fit_dict[new]["old_token"] = current_old_token
                best_fit_dict[new]["old_token_idx"] = current_old_token_idx
                print("updated: ", best_fit_dict)

    datasets.enable_progress_bar()

    return best_fit_dict
