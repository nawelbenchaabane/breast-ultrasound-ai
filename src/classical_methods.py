import cv2
import numpy as np
from skimage.segmentation import chan_vese


def preprocess_ultrasound(image):
    """
    Preprocess a grayscale breast ultrasound image using contrast enhancement
    and median filtering.
    """
    image = image.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)
    denoised = cv2.medianBlur(enhanced, 5)

    return denoised


def keep_relevant_components(
    binary_mask,
    image,
    min_area=80,
    max_area_ratio=0.15,
    border_margin=5,
    keep_top_k=2,
    min_contrast=0.035,
    min_score=0.08
):
    """
    Keep only plausible lesion-like connected components.

    This filtering removes:
    - tiny noisy components,
    - very large dark regions,
    - components touching image borders,
    - low-contrast structures.
    """
    h, w = binary_mask.shape
    max_area = max_area_ratio * h * w

    binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    candidates = []

    for label_id in range(1, num_labels):
        x, y, width, height, area = stats[label_id]

        if area < min_area:
            continue

        if area > max_area:
            continue

        if (
            x <= border_margin
            or y <= border_margin
            or x + width >= w - border_margin
            or y + height >= h - border_margin
        ):
            continue

        component_mask = labels == label_id

        component_mean = image[component_mask].mean()
        outside_mean = image[~component_mask].mean()

        contrast = (outside_mean - component_mean) / 255.0

        if contrast < min_contrast:
            continue

        component_uint8 = component_mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            component_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        extent = area / (width * height + 1e-6)

        if extent < 0.15:
            continue

        cx, cy = centroids[label_id]

        center_distance = np.sqrt(
            ((cx - w / 2) / (w / 2)) ** 2
            + ((cy - h / 2) / (h / 2)) ** 2
        )

        center_score = max(0.0, 1.0 - center_distance)

        score = (
            0.55 * contrast
            + 0.30 * compactness
            + 0.15 * center_score
        )

        if score < min_score:
            continue

        candidates.append({
            "label_id": label_id,
            "score": score,
            "area": area,
            "contrast": contrast
        })

    if len(candidates) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )[:keep_top_k]

    output_mask = np.zeros((h, w), dtype=np.uint8)

    for candidate in candidates:
        output_mask[labels == candidate["label_id"]] = 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    output_mask = cv2.morphologyEx(
        output_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    output_mask = cv2.morphologyEx(
        output_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return output_mask.astype(np.uint8)


def otsu_segmentation(image):
    """
    Otsu thresholding with lesion-like component filtering.
    """
    preprocessed = preprocess_ultrasound(image)

    _, raw_mask = cv2.threshold(
        preprocessed,
        0,
        1,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    final_mask = keep_relevant_components(
        raw_mask,
        preprocessed,
        min_area=80,
        max_area_ratio=0.15,
        min_contrast=0.035,
        min_score=0.08
    )

    return final_mask


def adaptive_segmentation(image):
    """
    Adaptive thresholding with lesion-like component filtering.
    """
    preprocessed = preprocess_ultrasound(image)

    raw_mask = cv2.adaptiveThreshold(
        preprocessed,
        1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        3
    )

    final_mask = keep_relevant_components(
        raw_mask,
        preprocessed,
        min_area=80,
        max_area_ratio=0.15,
        min_contrast=0.035,
        min_score=0.08
    )

    return final_mask


def chan_vese_segmentation(image, max_iter=80):
    """
    Chan-Vese active contour initialized from the filtered Otsu candidate.
    If no plausible lesion candidate is found, an empty mask is returned.
    """
    preprocessed = preprocess_ultrasound(image)

    init_mask = otsu_segmentation(image)

    if init_mask.sum() == 0:
        return np.zeros_like(init_mask, dtype=np.uint8)

    image_norm = preprocessed.astype(np.float32) / 255.0

    cv_mask = chan_vese(
        image_norm,
        mu=0.25,
        lambda1=1.0,
        lambda2=1.0,
        tol=1e-3,
        max_num_iter=max_iter,
        dt=0.5,
        init_level_set=init_mask.astype(float),
        extended_output=False
    )

    cv_mask = cv_mask.astype(np.uint8)

    final_mask = keep_relevant_components(
        cv_mask,
        preprocessed,
        min_area=80,
        max_area_ratio=0.15,
        min_contrast=0.025,
        min_score=0.06
    )

    return final_mask
