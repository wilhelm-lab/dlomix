from .predictor import InferencePipeline

__all__ = ["InferencePipeline"]

# Legacy TF-only pipeline; keep importable where its dependencies resolve.
try:
    from .pipeline import RetentionTimePipeline

    __all__.append("RetentionTimePipeline")
except Exception:
    pass
