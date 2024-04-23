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
