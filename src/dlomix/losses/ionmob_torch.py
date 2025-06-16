import torch
import torch.nn as nn


class MaskedIonmobLoss(nn.Module):
    def __init__(self, use_mse: bool = True):
        """
        This loss can be used for the training of an ion-mobility predictor that
        outputs both expected ion mean and ion standard deviation.

        Args:
            use_mse: if false, MAE will be used instead of MSE.
        """
        super(MaskedIonmobLoss, self).__init__()
        self.loss_function = nn.MSELoss() if use_mse else nn.L1Loss()

    def forward(self, outputs, targets):
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

        # Ensure all tensors are on the same device
        device = ccs_output.device
        target_ccs = target_ccs.to(device)
        target_ccs_std = target_ccs_std.to(device)

        # Compute MSE/MAE loss for CCS
        loss_ccs = self.loss_function(ccs_output, target_ccs)

        # Masked MSE/MAE loss for CCS-STD (ignore -1 values)
        mask = target_ccs_std != -1

        if mask.any():  # Ensure at least one valid target exists
            loss_ccs_std = self.loss_function(
                ccs_std_output[mask], target_ccs_std[mask]
            )
        else:
            loss_ccs_std = torch.tensor(0.0, device=device)

        loss_total = loss_ccs + loss_ccs_std
        return loss_total
