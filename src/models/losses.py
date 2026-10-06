"""
Loss functions and segmentation evaluation metrics for histological glomeruli segmentation.

Includes:
- Soft Dice Loss with Laplace smoothing
- Binary Cross Entropy + Dice Combined Loss (BCEDiceLoss)
- Binary Focal Loss (for extreme foreground-background class imbalance)
- Unified Loss Factory `build_criterion`
- Robust pixel-level metrics computation (Dice, IoU, Precision, Recall, Specificity)
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ComboLoss(nn.Module):
    """
    Three-part composite loss: BCE + Soft Dice + Focal Loss.
    Uses optimized SMP implementations under the hood.
    Offers maximum gradient stability and boundary precision for histology.
    """
    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.5,
        focal_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss()
        # from_logits=True tells SMP that our U-Net has no Sigmoid at the end
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.focal = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets.float())
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(logits, targets)
        return loss


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate the loss function based on config dictionary.
    """
    loss_name = config.get("loss_type", "combo").lower()

    if loss_name in ("combo", "bce_dice_focal"):
        return ComboLoss(
            bce_weight=float(config.get("bce_weight", 0.3)),
            dice_weight=float(config.get("dice_weight", 0.5)),
            focal_weight=float(config.get("focal_weight", 0.2)),
        )
    elif loss_name in ("bce_dice", "bcedice"):
        return ComboLoss(
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
            focal_weight=0.0, # Disable Focal Loss
        )
    elif loss_name == "dice":
        return smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
    elif loss_name in ("bce", "cross_entropy"):
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Choose from: combo, bce_dice, dice, bce")

# ==============================================================================
# Evaluation Metrics Computation
# ==============================================================================

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Compute pixel-level Dice (F1), IoU (Jaccard), Precision, Recall, and Specificity.

    Args:
        logits: Tensor of raw model outputs (B, 1, H, W).
        targets: Binary ground truth tensor (B, 1, H, W) with values 0 or 1.
        threshold: Sigmoid probability cutoff for positive classification.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary containing 'dice', 'iou', 'precision', 'recall', and 'specificity'.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }