import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: raw segmentation output [B, 1, H, W]
        targets: binary masks [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)

        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        dice = self.dice_loss(logits, targets)

        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss


class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=0.5):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

        self.segmentation_loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, seg_logits, masks, cls_logits, labels):
        seg_loss = self.segmentation_loss(seg_logits, masks)
        cls_loss = self.classification_loss(cls_logits, labels.long())

        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        loss_dict = {
            "total_loss": total_loss.detach(),
            "seg_loss": seg_loss.detach(),
            "cls_loss": cls_loss.detach()
        }

        return total_loss, loss_dict
