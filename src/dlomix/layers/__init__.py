from ..config import _BACKEND, PYTORCH_BACKEND, TENSORFLOW_BACKEND

__all__ = []

if _BACKEND in TENSORFLOW_BACKEND:
    from .attention import AttentionLayer, DecoderAttentionLayer
    from .meta_encoder import MetaEncoder
    from .positional_embedding import PositionalEmbedding
    from .transformer import TransformerEncoder
elif _BACKEND in PYTORCH_BACKEND:
    from .attention_torch import AttentionLayer, DecoderAttentionLayer
    from .bi_gru_seq_encoder_torch import BiGRUSequentialEncoder

    __all__.append("BiGRUSequentialEncoder")

__all__.extend(
    [
        "AttentionLayer",
        "DecoderAttentionLayer",
    ]
)
