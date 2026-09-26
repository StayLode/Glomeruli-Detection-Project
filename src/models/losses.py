"""
Loss functions and segmentation evaluation metrics for histological glomeruli segmentation.

Includes:
- Soft Dice Loss with Laplace smoothing
- Binary Cross Entropy + Dice Combined Loss (BCEDiceLoss)
- Binary Focal Loss (for extreme foreground-background class imbalance)
- Focal Tversky Loss (penalizing false negatives to prevent under-segmentation)
- Unified Loss Factory `build_criterion`
- Robust pixel-level metrics computation (Dice, IoU, Precision, Recall, Specificity)
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Loss Functions
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Computes Soft Dice Loss for binary segmentation directly from raw logits.
    """

    def __init__(self, smooth: float = 1e-6, square_denominator: bool = False) -> None:
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        if self.square_denominator:
            denominator = (probs ** 2).sum() + (targets ** 2).sum()
        else:
            denominator = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification from logits.
    Downweights easy negative background pixels and focuses on hard boundary pixels.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view(-1)
        logits = logits.view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced biomedical segmentation.
    Setting beta > alpha (e.g. beta=0.7, alpha=0.3) penalizes False Negatives
    heavily, preventing the model from under-segmenting glomerular margins.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - tversky) ** self.gamma
        return loss


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy (with logits) and Soft Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class ComboLoss(nn.Module):
    """
    Three-part composite loss: BCE + Soft Dice + Binary Focal Loss.
    Offers maximum gradient stability and boundary precision for histology.
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.5,
        focal_weight: float = 0.2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets.float())
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        return loss


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate loss function based on config dictionary.
    """
    loss_name = config.get("loss_type", "combo").lower()

    if loss_name in ("combo", "bce_dice_focal"):
        return ComboLoss(
            bce_weight=float(config.get("bce_weight", 0.3)),
            dice_weight=float(config.get("dice_weight", 0.5)),
            focal_weight=float(config.get("focal_weight", 0.2)),
        )
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=float(config.get("tversky_alpha", 0.3)),
            beta=float(config.get("tversky_beta", 0.7)),
            gamma=float(config.get("tversky_gamma", 1.33)),
        )
    elif loss_name in ("bce_dice", "bcedice"):
        return BCEDiceLoss(
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        )
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name in ("bce", "cross_entropy"):
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Choose from: combo, bce_dice, focal_tversky, dice, bce")


# ==============================================================================
# 2. Evaluation Metrics Computation
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
