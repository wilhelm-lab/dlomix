from ..config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

if _BACKEND in TENSORFLOW_BACKEND:
    from .chargestate import adjusted_mean_absolute_error, adjusted_mean_squared_error
    from .rt_eval import TimeDeltaMetric, timedelta
elif _BACKEND in PYTORCH_BACKEND:
    from .chargestate_torch import adjusted_mean_absolute_error, adjusted_mean_squared_error
    from .rt_eval_torch import TimeDeltaMetric, timedelta

__all__ = [
    "adjusted_mean_absolute_error",
    "adjusted_mean_squared_error",
    "timedelta",
    "TimeDeltaMetric",
]
