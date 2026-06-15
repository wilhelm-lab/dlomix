# most of this can go away, as it is not used anymore

from enum import Enum


class EncodingScheme(str, Enum):
    """
    Enum for encoding schemes.
    """

    UNMOD = "unmod"
    NAIVE_MODS = "naive-mods"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"Provided {cls.__name__} is not defined. Valid values are: {', '.join([e.value for e in cls])}"
        )


def get_num_processors():
    """
    Get the number of processors available on the system.

    Returns
    -------
    int
        Number of processors available.
    """

    import multiprocessing

    return multiprocessing.cpu_count()


def validate_num_proc_value(num_proc):
    """
    Validate the user-provided num_proc value.

    Allowed values:
    - None: force single-process execution
    - -1: use all available processors
    - positive integer: explicit number of processors
    """

    if num_proc is None:
        return

    if not isinstance(num_proc, int):
        raise ValueError(
            "The num_proc parameter should be None, -1, or a positive integer."
        )

    if num_proc != -1 and num_proc <= 0:
        raise ValueError(
            "Invalid num_proc value. Use None for single-process mode, -1 for all available processors, or a positive integer."
        )


def resolve_num_proc(num_proc, n_processors):
    """
    Resolve num_proc to the effective value used by HF Dataset map/filter calls.

    Returns
    -------
    tuple
        (resolved_num_proc, was_capped)
    """

    validate_num_proc_value(num_proc)

    if num_proc is None:
        return None, False

    if num_proc == -1:
        return n_processors, False

    if num_proc > n_processors:
        return n_processors, True

    return num_proc, False
