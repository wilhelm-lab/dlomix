"""Model arguments that a dataset already knows, shared by both backends."""


def prosit_intensity_arguments(dataset, explicit: dict) -> dict:
    """Constructor arguments of ``PrositIntensityPredictor`` for ``dataset``.

    The arguments that must agree with the dataset are taken from it: ``alphabet``,
    ``seq_length``, ``with_termini``, the sequence entry of ``input_keys``, and
    ``meta_data_keys`` / ``use_meta_data`` from its ``model_features``. ``explicit``
    arguments are passed through; one that contradicts the dataset raises.
    """
    sequence_key = {"SEQUENCE_KEY": dataset.sequence_column}
    meta_data_keys = list(dataset.model_features or [])
    from_dataset = {
        "alphabet": dict(dataset.extended_alphabet),
        "seq_length": dataset.max_seq_len,
        "with_termini": dataset.with_termini,
        "meta_data_keys": meta_data_keys,
        "use_meta_data": bool(meta_data_keys),
        # further input keys (e.g. the instrument type) can be given explicitly
        "input_keys": {**(explicit.get("input_keys") or {}), **sequence_key},
    }

    given = dict(explicit)
    if isinstance(given.get("meta_data_keys"), dict):
        given["meta_data_keys"] = list(given["meta_data_keys"].values())
    if "input_keys" in given:
        given["input_keys"] = dict(given["input_keys"] or {})
    conflicts = {
        name: (given[name], value)
        for name, value in from_dataset.items()
        if name in given and given[name] != value
    }
    if conflicts:
        details = "; ".join(
            f"{name}={given_value!r} but the dataset has {value!r}"
            for name, (given_value, value) in conflicts.items()
        )
        raise ValueError(
            f"Arguments contradict the dataset: {details}. Leave them out to take "
            "them from the dataset."
        )
    return {**explicit, **from_dataset}
