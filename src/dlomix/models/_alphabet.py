"""Alphabet validation shared by the TensorFlow models."""


def validate_alphabet_size(alphabet, model_name):
    """Raise when ``alphabet`` is too small for the indices it maps to.

    The models size their embedding as ``len(alphabet)``, which holds when the
    vocabulary carries every index it uses -- including the padding and unknown
    tokens that :class:`~dlomix.data.PeptideDataset` inserts explicitly. Passing
    a raw alphabet constant leaves those out, and the resulting out-of-range
    lookups return zeros silently on GPU instead of raising, so the affected
    tokens train to nothing with no error anywhere.
    """
    if not alphabet:
        return

    max_index = max(alphabet.values())
    if max_index >= len(alphabet):
        raise ValueError(
            f"{model_name} received an alphabet of {len(alphabet)} tokens whose highest "
            f"index is {max_index}, so an embedding of len(alphabet) rows cannot "
            f"represent it -- {max_index + 1} rows are needed. This usually means a raw "
            "alphabet constant was passed instead of the vocabulary learned by the "
            "dataset. Use `dataset.extended_alphabet`, which includes the padding and "
            "unknown tokens."
        )
