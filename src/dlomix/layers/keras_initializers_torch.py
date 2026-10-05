"""Initialize PyTorch layers the way Keras initializes their counterparts.

PyTorch and Keras use different default initializers, and for some models this
changes how well training converges (for Ionmob, the test error of PyTorch models
trained from PyTorch's defaults was about 7% higher). All DLOmix PyTorch models use
these helpers so that both backends start from the same distribution of weights:

=================  ====================================  ==============================
Layer              Keras default                         PyTorch default
=================  ====================================  ==============================
Embedding          uniform(-0.05, 0.05)                  normal(0, 1)
Dense / Conv1D     Glorot uniform, zero bias             Kaiming uniform, uniform bias
GRU                Glorot uniform input kernel,          uniform(±1/sqrt(hidden)) for
                   orthogonal recurrent kernel,          all weights and biases
                   zero biases
=================  ====================================  ==============================

Layers whose size is inferred on the first forward pass (``nn.LazyLinear``,
``nn.LazyConv1d``) only get their weights then, so :class:`KerasLazyLinear` and
:class:`KerasLazyConv1d` apply the Keras initialization at that point.
"""

import torch.nn as nn

EMBEDDING_INIT_RANGE = 0.05  # keras.initializers.RandomUniform(-0.05, 0.05)


def init_dense_like_keras(layer: nn.Module) -> None:
    """Glorot-uniform weights and zero bias, for ``nn.Linear`` and ``nn.Conv1d``.

    PyTorch's Xavier initialization computes the fans of a convolution kernel from
    the receptive field as Keras does, so the bounds are the same.
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_embedding_like_keras(layer: nn.Embedding) -> None:
    nn.init.uniform_(layer.weight, -EMBEDDING_INIT_RANGE, EMBEDDING_INIT_RANGE)
    if layer.padding_idx is not None:
        nn.init.zeros_(layer.weight[layer.padding_idx])


def init_gru_like_keras(
    layer: nn.GRU, recurrent_initializer: str = "orthogonal"
) -> None:
    """Glorot-uniform input weights, orthogonal recurrent weights, zero biases.

    Keras draws one orthogonal matrix of shape (hidden, 3 * hidden) for the three
    gates together; PyTorch stores its transpose with the gates in another order,
    which is still orthogonal, so drawing the full matrix here is equivalent.
    ``recurrent_initializer="glorot_uniform"`` matches a Keras GRU created with that
    initializer (the fans of the transposed matrix are the same).
    """
    init_recurrent = {
        "orthogonal": nn.init.orthogonal_,
        "glorot_uniform": nn.init.xavier_uniform_,
    }[recurrent_initializer]
    for name, parameter in layer.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(parameter)
        elif name.startswith("weight_hh"):
            init_recurrent(parameter)
        elif name.startswith("bias"):
            nn.init.zeros_(parameter)


def init_like_keras(
    module: nn.Module, gru_recurrent_initializer: str = "orthogonal"
) -> None:
    """Apply the Keras initialization to every supported layer inside ``module``.

    Layers that do not have their weights yet (lazy layers before the first forward
    pass) and parameters of other layer types are left untouched; custom layers
    initialize their own parameters like their Keras counterparts.
    """
    for layer in module.modules():
        if isinstance(layer, nn.modules.lazy.LazyModuleMixin):
            continue  # initialized on materialization, see the Keras* lazy layers
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            init_dense_like_keras(layer)
        elif isinstance(layer, nn.Embedding):
            init_embedding_like_keras(layer)
        elif isinstance(layer, nn.GRU):
            init_gru_like_keras(layer, gru_recurrent_initializer)


class _KerasInitLinear(nn.Linear):
    def reset_parameters(self):
        init_dense_like_keras(self)


class KerasLazyLinear(nn.LazyLinear):
    """``nn.LazyLinear`` that is initialized like a Keras ``Dense`` layer."""

    cls_to_become = _KerasInitLinear

    def reset_parameters(self):
        if not self.has_uninitialized_params() and self.in_features != 0:
            init_dense_like_keras(self)


class _KerasInitConv1d(nn.Conv1d):
    def reset_parameters(self):
        init_dense_like_keras(self)


class KerasLazyConv1d(nn.LazyConv1d):
    """``nn.LazyConv1d`` that is initialized like a Keras ``Conv1D`` layer."""

    cls_to_become = _KerasInitConv1d

    def reset_parameters(self):
        if not self.has_uninitialized_params() and self.in_channels != 0:
            init_dense_like_keras(self)
