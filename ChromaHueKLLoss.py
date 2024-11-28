import torch 
import torch.nn as nn
import torch.nn.functional as F

class CromaHueKLLoss(nn.Module):
    def __init__(self, lambda_hue=5.0) -> None:
        super(CromaHueKLLoss, self).__init__()
        self.lambda_hue = lambda_hue

    def forward(self, rho_chroma, rho_hat_chroma, rho_hue, rho_hat_hue, chroma):

        """
        Compute the combined KL divergence loss.

        Args:
            rho_chroma (Tensor): Ground truth Chroma distribution.
            rho_hat_chroma (Tensor): Predicted Chroma distribution.
            rho_hue (Tensor): Ground truth Hue distribution.
            rho_hat_hue (Tensor): Predicted Hue distribution.
            chroma (Tensor): Chroma values for weighting the Hue loss.

        Returns:
            Tensor: Combined KL divergence loss.
        """

        # Normalize distributions (to ensure they are valid probability distributions)
        rho_chroma = F.softmax(rho_chroma, dim=-1)
        rho_hat_chroma = F.softmax(rho_hat_chroma, dim=-1)
        rho_hue = F.softmax(rho_hue, dim=-1)
        rho_hat_hue = F.softmax(rho_hat_hue, dim=-1)

        # Reshape distributions to match expected input shape for KL divergence
        rho_chroma = rho_chroma.permute(0, 2, 3, 1).reshape(-1, rho_chroma.shape[1])  # Flatten to [N, C]
        rho_hat_chroma = rho_hat_chroma.permute(0, 2, 3, 1).reshape(-1, rho_hat_chroma.shape[1])  # Flatten to [N, C]
        rho_hue = rho_hue.permute(0, 2, 3, 1).reshape(-1, rho_hue.shape[1])  # Flatten to [N, C]
        rho_hat_hue = rho_hat_hue.permute(0, 2, 3, 1).reshape(-1, rho_hat_hue.shape[1])  # Flatten to [N, C]

        # Compute KL divergence for Chroma
        kl_chroma = F.kl_div(torch.log(rho_hat_chroma + 1e-8), rho_chroma, reduction='batchmean')
        
        # Compute KL divergence for Hue
        kl_hue = F.kl_div(torch.log(rho_hat_hue + 1e-8), rho_hue, reduction='batchmean')
        
        # Weight the Hue KL divergence by Chroma
        weighted_kl_hue = torch.mean(chroma * kl_hue)

        # Combine the losses
        #replaces the L1 loss of discriminator
        total_loss = kl_chroma + self.lambda_hue * weighted_kl_hue
        return total_loss
