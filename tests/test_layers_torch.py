import logging

import torch

from dlomix.layers.attention_torch import AttentionLayer, DecoderAttentionLayer

logger = logging.getLogger(__name__)


def test_attention_layer_shapes():
    batch_size = 2
    seq_len = 5
    feature_dim = 8

    dummy_input = torch.randn(batch_size, seq_len, feature_dim)

    layer = AttentionLayer(
        feature_dim=feature_dim, seq_len=seq_len, context=False, bias=True
    )
    output = layer(dummy_input)

    logger.info(
        "Original Shape: {}, Layer output shape".format(
            (batch_size, seq_len, feature_dim), output.shape
        )
    )

    assert output.detach().numpy().shape == (batch_size, feature_dim)


def test_decoder_attention_layer_shapes():
    batch_size = 2
    time_steps = 30
    features = 5

    dummy_input = torch.randn(batch_size, time_steps, features)
    layer = DecoderAttentionLayer(time_steps=time_steps)
    output = layer(dummy_input)

    logger.info(
        "Original Shape: {}, Layer output shape".format(
            (batch_size, time_steps, features), output.shape
        )
    )

    assert output.detach().numpy().shape == (batch_size, time_steps, features)
