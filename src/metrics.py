import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def _binary_predictions_from_logits(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    return preds


def dice_coefficient(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Computes mean Dice score on a batch.

    Empty-mask behavior:
    - ground truth empty and prediction empty => Dice = 1
    - ground truth empty and prediction not empty => Dice close to 0
    """
    preds = _binary_predictions_from_logits(logits, threshold)
    targets = targets.float()

    dims = (1, 2, 3)

    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Computes mean Intersection over Union on a batch.

    Empty-mask behavior:
    - ground truth empty and prediction empty => IoU = 1
    - ground truth empty and prediction not empty => IoU close to 0
    """
    preds = _binary_predictions_from_logits(logits, threshold)
    targets = targets.float()

    dims = (1, 2, 3)

    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds + targets, dim=dims) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


def segmentation_precision(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Computes mean segmentation precision on a batch.

    Precision = TP / predicted positives.
    If no pixel is predicted positive:
    - precision = 1 if ground truth is also empty
    - precision = 0 if ground truth contains lesion pixels
    """
    preds = _binary_predictions_from_logits(logits, threshold)
    targets = targets.float()

    dims = (1, 2, 3)

    true_positive = torch.sum(preds * targets, dim=dims)
    predicted_positive = torch.sum(preds, dim=dims)
    actual_positive = torch.sum(targets, dim=dims)

    precision = torch.where(
        predicted_positive > 0,
        (true_positive + smooth) / (predicted_positive + smooth),
        torch.where(actual_positive == 0, torch.ones_like(actual_positive), torch.zeros_like(actual_positive))
    )

    return precision.mean().item()


def segmentation_recall(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Computes mean segmentation recall on a batch.

    Recall = TP / actual positives.
    If ground truth is empty:
    - recall = 1 if prediction is also empty
    - recall = 0 if model predicts false positive lesion pixels
    """
    preds = _binary_predictions_from_logits(logits, threshold)
    targets = targets.float()

    dims = (1, 2, 3)

    true_positive = torch.sum(preds * targets, dim=dims)
    predicted_positive = torch.sum(preds, dim=dims)
    actual_positive = torch.sum(targets, dim=dims)

    recall = torch.where(
        actual_positive > 0,
        (true_positive + smooth) / (actual_positive + smooth),
        torch.where(predicted_positive == 0, torch.ones_like(predicted_positive), torch.zeros_like(predicted_positive))
    )

    return recall.mean().item()


def segmentation_metrics(logits, targets, threshold=0.5):
    """
    Returns all segmentation metrics in one dictionary.
    """
    return {
        "dice": dice_coefficient(logits, targets, threshold),
        "iou": iou_score(logits, targets, threshold),
        "precision": segmentation_precision(logits, targets, threshold),
        "recall": segmentation_recall(logits, targets, threshold)
    }


def classification_metrics(cls_logits, labels):
    """
    Computes classification metrics for one batch.
    cls_logits: raw classification outputs [B, num_classes]
    labels: integer labels [B]
    """
    preds = torch.argmax(cls_logits, dim=1)

    y_true = labels.detach().cpu().numpy()
    y_pred = preds.detach().cpu().numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0)
    }


def classification_metrics_from_arrays(y_true, y_pred):
    """
    Computes classification metrics from full arrays.
    Useful at the end of validation or testing.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0)
    }


def compute_confusion_matrix(y_true, y_pred):
    """
    Computes confusion matrix from class labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return confusion_matrix(y_true, y_pred)
