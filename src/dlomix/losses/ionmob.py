"""Ionmob loss for the TensorFlow backend, written against ``keras.ops``.

Same constructor and call signature as the PyTorch ``nn.Module`` in
:mod:`dlomix.losses.ionmob_torch`, and the same values for the same inputs.
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="dlomix")
class MaskedIonmobLoss:
    """Loss for an ion-mobility predictor that outputs the CCS mean and standard deviation.

    The CCS standard deviation term ignores the targets set to -1 (no measured std).

    Args:
        use_mse: if false, MAE will be used instead of MSE.
    """

    def __init__(self, use_mse: bool = True):
        self.use_mse = use_mse

    def _error(self, prediction, target):
        difference = prediction - target
        return ops.square(difference) if self.use_mse else ops.abs(difference)

    def __call__(self, outputs, targets):
        """
        Computes loss for Ionmob model with masked CCS standard deviation loss.

        Args:
            outputs: Tuple[CCS, CCS_STD]
            targets: Tuple[CCS, CCS_STD]

        Returns:
            Combined loss of predicted CCS and masked CCS STD.
        """
        ccs_output, ccs_std_output = outputs
        target_ccs, target_ccs_std = targets

        # targets may come as (batch,) from a dataset; match the (batch, 1) outputs
        target_ccs = ops.reshape(
            ops.cast(target_ccs, ccs_output.dtype), ops.shape(ccs_output)
        )
        target_ccs_std = ops.reshape(
            ops.cast(target_ccs_std, ccs_std_output.dtype), ops.shape(ccs_std_output)
        )

        loss_ccs = ops.mean(self._error(ccs_output, target_ccs))

        # mean over the targets that are not -1; 0 if there are none
        mask = ops.cast(ops.not_equal(target_ccs_std, -1), ccs_std_output.dtype)
        masked_error = self._error(ccs_std_output, target_ccs_std) * mask
        loss_ccs_std = ops.sum(masked_error) / ops.maximum(ops.sum(mask), 1.0)

        return loss_ccs + loss_ccs_std

    def get_config(self):
        return {"use_mse": self.use_mse}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
