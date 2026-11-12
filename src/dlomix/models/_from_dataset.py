"""``Model.from_dataset``: the model arguments that a dataset already knows.

Shared by both backends. Each function below gives, for one kind of model, the
constructor arguments that must agree with the dataset; :func:`from_dataset_method`
turns it into the model's ``from_dataset`` classmethod.
"""

import numpy as np

from ..data.processing.chain import padded_sequence_length


def sequence_model_arguments(dataset, explicit):
    """``alphabet`` (its ``extended_alphabet``) and ``seq_length`` (the width of its
    encoded sequences: ``max_seq_len``, plus 2 with termini)."""
    return {
        "alphabet": dict(dataset.extended_alphabet),
        "seq_length": padded_sequence_length(dataset.max_seq_len, dataset.with_termini),
    }


def deeplc_arguments(dataset, explicit):
    """``alphabet`` (its ``extended_alphabet``), ``seq_length`` (the width of its
    encoded sequences: ``max_seq_len``, plus 2 with termini) and
    ``sequence_input_key`` (its sequence column)."""
    return {
        **sequence_model_arguments(dataset, explicit),
        "sequence_input_key": dataset.sequence_column,
    }


def prosit_intensity_arguments(dataset, explicit):
    """``alphabet`` (its ``extended_alphabet``), ``seq_length`` (its ``max_seq_len``),
    ``with_termini``, the sequence column in ``input_keys``, and ``meta_data_keys`` /
    ``use_meta_data`` (its ``model_features``)."""
    meta_data_keys = list(dataset.model_features or [])
    return {
        "alphabet": dict(dataset.extended_alphabet),
        "seq_length": dataset.max_seq_len,
        "with_termini": dataset.with_termini,
        "meta_data_keys": meta_data_keys,
        "use_meta_data": bool(meta_data_keys),
        # further input keys (e.g. the instrument type) can be given explicitly
        "input_keys": {
            **(explicit.get("input_keys") or {}),
            "SEQUENCE_KEY": dataset.sequence_column,
        },
    }


def detectability_arguments(dataset, explicit):
    """``padding_char``: the one-hot vector of its padding token, as long as its
    ``extended_alphabet``."""
    alphabet = dataset.extended_alphabet
    padding_char = np.zeros(len(alphabet))
    padding_char[alphabet[dataset.padding_value]] = 1
    return {"padding_char": padding_char}


def detectability_torch_arguments(dataset, explicit):
    """``alphabet_size`` (the size of its ``extended_alphabet``) and ``padding_idx``
    (the index of its padding token)."""
    alphabet = dataset.extended_alphabet
    return {
        "alphabet_size": len(alphabet),
        "padding_idx": alphabet[dataset.padding_value],
    }


def ionmob_arguments(dataset, explicit):
    """``num_tokens`` (the size of its ``extended_alphabet``) and ``sequence_key``
    (its sequence column)."""
    return {
        "num_tokens": len(dataset.extended_alphabet),
        "sequence_key": dataset.sequence_column,
    }


def ionmob_torch_arguments(dataset, explicit):
    """``num_tokens`` (the size of its ``extended_alphabet``)."""
    return {"num_tokens": len(dataset.extended_alphabet)}


def _comparable(name, value):
    if name == "meta_data_keys" and isinstance(value, dict):
        return list(value.values())
    if name == "input_keys":
        return dict(value or {})
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def dataset_arguments(derive, dataset, explicit: dict) -> dict:
    """``explicit`` with the arguments ``derive`` takes from ``dataset``.

    An explicit argument that contradicts the dataset raises.
    """
    from_dataset = derive(dataset, explicit)
    conflicts = {
        name: (explicit[name], value)
        for name, value in from_dataset.items()
        if name in explicit
        and _comparable(name, explicit[name]) != _comparable(name, value)
    }
    if conflicts:
        details = "; ".join(
            f"{name}={given!r} but the dataset has {value!r}"
            for name, (given, value) in conflicts.items()
        )
        raise ValueError(
            f"Arguments contradict the dataset: {details}. Leave them out to take "
            "them from the dataset."
        )
    return {**explicit, **from_dataset}


def from_dataset_method(derive):
    """The ``from_dataset`` classmethod of a model whose dataset arguments are
    ``derive(dataset, explicit)``; ``derive``'s docstring names them."""

    def from_dataset(cls, dataset, **kwargs):
        return cls(**dataset_arguments(derive, dataset, kwargs))

    from_dataset.__doc__ = f"""Build the model for a dlomix dataset.

        The arguments that must agree with the dataset are taken from it:
        {derive.__doc__}

        Pass everything else, e.g. the layer sizes, as keyword arguments::

            model = Model.from_dataset(dataset, **other_arguments)

        Raises
        ------
        ValueError
            If a keyword argument contradicts the dataset.
        """
    return classmethod(from_dataset)
