import torch 
import torch.nn as nn
import torch.nn.functional as F

class CromaHueKLLoss(nn.Module):
    def __init__(self, lambda_hue=5.0) -> None:
        super(CromaHueKLLoss, self).__init__()
        self.lambda_hue = lambda_hue

    def forward(self, gt_chroma, pred_chroma, gt_hue, pred_hue, chroma_weights):
        """
        Compute the combined KL divergence loss.
        
        Args:
            gt_chroma: Ground truth Chroma distribution [0,1]
            pred_chroma: Predicted Chroma distribution [0,1]
            gt_hue: Ground truth Hue distribution [0,1]
            pred_hue: Predicted Hue distribution [0,1]
            chroma_weights: Chroma values for weighting the Hue loss [0,1]
        """
        # For KL div, input should be log probabilities and target should be probabilities
        pred_chroma = F.softmax(pred_chroma, dim=1)
        pred_hue = F.softmax(pred_hue, dim=1)
        
        # Convert ground truth to probability distributions
        gt_chroma = F.softmax(gt_chroma, dim=1)
        gt_hue = F.softmax(gt_hue, dim=1)
        
        # Compute KL divergence for Chroma
        kl_chroma = F.kl_div(
            torch.log(pred_chroma + 1e-8),
            gt_chroma,
            reduction='none'
        ).sum(dim=1).mean()

        # Compute KL divergence for Hue
        kl_hue = F.kl_div(
            torch.log(pred_hue + 1e-8),
            gt_hue,
            reduction='none'
        ).sum(dim=1).mean()
        
        # Weight the Hue loss by Chroma values
        weighted_kl_hue = chroma_weights.mean() * kl_hue

        # Combine losses
        total_loss = kl_chroma + self.lambda_hue * weighted_kl_hue
        
        return total_loss