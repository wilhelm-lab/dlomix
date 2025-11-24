from typing import Union

import tensorflow as tf
import torch  # ! to figure out how to avoid importing both

Tensor = Union[torch.Tensor, tf.Tensor]
