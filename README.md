# Breast Ultrasound AI: Segmentation and Classification

This project focuses on breast ultrasound image analysis using deep learning and classical image processing methods.

The objective is to segment breast lesions and classify ultrasound images into three diagnostic categories:

* benign
* malignant
* normal

The project compares classical segmentation methods, a baseline U-Net segmentation model, and a multi-task Attention U-Net model for joint segmentation and classification.

---

## Project Overview

Breast ultrasound images are challenging because of:

* low contrast between lesions and surrounding tissue,
* speckle noise,
* irregular lesion boundaries,
* high variability in lesion size and appearance,
* class imbalance between benign, malignant, and normal cases.

This project addresses these challenges through:

1. classical segmentation baselines,
2. deep learning-based lesion segmentation,
3. joint segmentation and classification,
4. quantitative evaluation using Dice, IoU, precision, recall, accuracy, and macro F1-score,
5. visual inspection of predicted segmentation masks.

---

## Dataset

This project uses the Breast Ultrasound Images Dataset, also known as BUSI.

The dataset contains ultrasound images divided into three classes:

* benign
* malignant
* normal

For benign and malignant images, lesion masks are provided.
For normal images, no lesion is present, so empty masks are generated during preprocessing.

The raw dataset is not included in this repository. See `data/README.md` for the expected dataset structure.

---

## Repository Structure

```text
breast-ultrasound-ai/
├── data/
│   └── README.md
├── notebooks/
│   └── Breast_Ultrasound_AI_Project.ipynb
├── results/
│   ├── final_model_comparison.csv
│   └── final_model_comparison.md
├── src/
│   ├── classical_methods.py
│   ├── config.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── models.py
├── .gitignore
├── requirements.txt
└── README.md
```

Raw images, generated masks, train/validation/test CSV files, and model checkpoints are excluded from GitHub.

---

## Methods

Three families of methods are implemented and compared.

---

## 1. Classical Segmentation Methods

Classical image processing methods are used as non-deep-learning baselines.

They do not require training and rely only on image intensity, contrast, morphology, and connected component filtering.

Implemented methods include:

* Otsu thresholding with morphology,
* adaptive thresholding,
* Chan-Vese active contour segmentation.

### Preprocessing

Before applying classical segmentation, the ultrasound image is converted to grayscale and enhanced using:

* CLAHE contrast enhancement,
* median filtering to reduce speckle noise.

This preprocessing step improves local contrast and reduces small noisy artifacts.

### Otsu + Morphology

Otsu thresholding automatically finds an intensity threshold that separates dark lesion-like regions from the background.

Since breast ultrasound images contain many dark regions unrelated to lesions, the raw Otsu mask is filtered using connected component analysis.

The filtering removes:

* very small noisy regions,
* very large regions,
* components touching the image borders,
* low-contrast components,
* implausible lesion-like candidates.

Morphological opening and closing are then applied to smooth the final mask.

### Adaptive Thresholding

Adaptive thresholding computes local thresholds instead of a single global threshold.

This can be useful when illumination and contrast vary across the ultrasound image.

However, in practice, this method tends to generate many false positives because ultrasound images contain many local dark structures.

### Chan-Vese Active Contour

Chan-Vese segmentation is an active contour method that evolves a contour to separate regions according to intensity homogeneity.

In this project, Chan-Vese is initialized using the filtered Otsu mask.

If no plausible lesion candidate is found, the method returns an empty mask.

### Interpretation

Classical methods are useful as baselines, but they are not robust enough for this task.

They struggle with:

* speckle noise,
* weak lesion boundaries,
* shadow artifacts,
* normal images with no lesion,
* irregular malignant lesion shapes.

---

## 2. U-Net Baseline

The baseline deep learning model is a U-Net architecture designed for medical image segmentation.

U-Net is based on an encoder-decoder structure.

### Encoder

The encoder progressively extracts image features at different spatial resolutions.

At each level, convolutional blocks learn increasingly abstract representations:

* low-level edges and textures,
* local tissue patterns,
* lesion-like structures,
* high-level semantic features.

Downsampling reduces spatial resolution while increasing the number of feature channels.

### Bottleneck

The bottleneck is the deepest part of the network.

It contains the most compressed and abstract representation of the ultrasound image.

This representation captures global context and lesion-related information.

### Decoder

The decoder progressively upsamples the feature maps to recover spatial resolution.

Its role is to produce a pixel-level segmentation mask.

### Skip Connections

Skip connections transfer high-resolution features from the encoder to the decoder.

This helps the model recover fine lesion boundaries and spatial details.

They are essential for medical segmentation because lesion contours can be small, subtle, and irregular.

### Output

The U-Net outputs segmentation logits.

These logits are converted into probabilities using a sigmoid function.

A threshold is then applied to obtain a binary lesion mask:

* pixels above the threshold are predicted as lesion,
* pixels below the threshold are predicted as background.

Two thresholds are compared:

* `0.50`: more sensitive, better lesion recall,
* `0.75`: more conservative, fewer false positives on normal images.

### Loss Function

The segmentation model is trained using a combination of binary cross-entropy and Dice loss.

Binary cross-entropy helps optimize pixel-wise classification.

Dice loss helps address class imbalance because lesion pixels are usually much fewer than background pixels.

---

## 3. MultiTask Attention U-Net

The MultiTask Attention U-Net extends the baseline U-Net by performing two tasks at the same time:

1. lesion segmentation,
2. image-level classification.

The model predicts both:

* a segmentation mask,
* a class label among benign, malignant, and normal.

### Shared Encoder

The encoder is shared by both tasks.

It extracts visual features from the ultrasound image that are useful for both segmentation and classification.

This shared representation allows the model to learn from both pixel-level and image-level supervision.

### Attention Gates

Attention gates are added to the skip connections.

Their role is to help the decoder focus on relevant image regions while suppressing less useful background information.

In breast ultrasound images, this is useful because the image can contain many confusing structures unrelated to the lesion.

The attention mechanism learns to weight encoder features before passing them to the decoder.

### Segmentation Branch

The segmentation branch works similarly to U-Net.

It uses the decoder to reconstruct a pixel-level lesion mask.

The output is a binary segmentation map.

### Classification Branch

The classification branch uses features from the bottleneck.

These deep features are passed through a classification head to predict one of three classes:

* benign,
* malignant,
* normal.

The classification head produces class logits.

These logits are converted into probabilities using a softmax function.

The predicted class is the class with the highest probability.

### Multi-Task Loss

The model is trained using a combined loss:

* segmentation loss for the lesion mask,
* classification loss for the image-level class.

The segmentation loss optimizes the predicted mask.

The classification loss optimizes the benign/malignant/normal prediction.

This allows the model to learn both local lesion boundaries and global diagnostic category information.

### Interpretation

The MultiTask Attention U-Net provides a richer output than the baseline U-Net because it predicts both segmentation and classification.

However, in this project, the baseline U-Net achieved the best pure segmentation performance, while the multi-task model added classification capability.

---

## Segmentation Visualization

Segmentation results are visualized using color overlays.

The colors have the following meaning:

* Red: ground-truth manual lesion mask,
* Green: predicted lesion mask from the model.

When red and green overlap, this indicates that the model prediction matches the manual annotation.

If green appears without red, the model predicted a false positive region.

If red appears without green, the model missed part of the lesion.

For normal images, the expected mask is empty.
A good prediction should therefore contain no green region.

---

## Meaning of `Conf`

In the classification visualizations, `Conf` means classification confidence.

It corresponds to the highest softmax probability assigned by the model to the predicted class.

For example, if the model outputs the following probabilities:

```text
benign:    0.82
malignant: 0.12
normal:    0.06
```

Then the predicted class is `benign`, and:

```text
Conf = 0.82
```

This value indicates how confident the model is in its predicted class according to its own probability distribution.

Important note: `Conf` is not a clinical certainty score.
It is only the model's internal confidence for the classification prediction.

---

## Evaluation Metrics

### Segmentation Metrics

The segmentation models are evaluated using:

* Dice coefficient,
* Intersection over Union,
* precision,
* recall.

### Dice Coefficient

Dice measures the overlap between the predicted mask and the ground-truth mask.

A Dice score close to 1 indicates strong overlap.

### IoU

Intersection over Union measures the ratio between the intersection and the union of the predicted and ground-truth masks.

It is stricter than Dice.

### Precision

Precision measures how much of the predicted lesion area is actually correct.

High precision means fewer false positives.

### Recall

Recall measures how much of the true lesion area was detected.

High recall means fewer missed lesion pixels.

### Classification Metrics

For classification, the model is evaluated using:

* accuracy,
* macro F1-score,
* class-wise precision,
* class-wise recall,
* class-wise F1-score.

Macro F1-score is important because the dataset is imbalanced across benign, malignant, and normal classes.

---

## Final Results

The table below summarizes the main test results.

The segmentation results are reported on lesion images only, meaning benign and malignant cases.

| Model                     | Task                          | Threshold | Test Lesion Dice | Test Lesion IoU | Test Lesion Precision | Test Lesion Recall | Classification Accuracy | Classification Macro F1 |
| ------------------------- | ----------------------------- | --------: | ---------------: | --------------: | --------------------: | -----------------: | ----------------------: | ----------------------: |
| Otsu + Morphology         | Classical segmentation        |         - |           0.1584 |          0.1255 |                0.2372 |             0.1720 |                       - |                       - |
| U-Net Baseline            | Segmentation                  |      0.50 |           0.7711 |          0.6866 |                0.8023 |             0.7908 |                       - |                       - |
| U-Net Baseline            | Segmentation                  |      0.75 |           0.7622 |          0.6799 |                0.8329 |             0.7389 |                       - |                       - |
| MultiTask Attention U-Net | Segmentation + Classification |      0.50 |           0.7389 |          0.6484 |                0.8026 |             0.7351 |                    0.76 |                    0.75 |
| MultiTask Attention U-Net | Segmentation + Classification |      0.75 |           0.6973 |          0.6088 |                0.8449 |             0.6511 |                    0.76 |                    0.75 |

---

## Key Findings

The classical segmentation methods performed poorly compared with deep learning methods.

This confirms that simple threshold-based approaches are not sufficient for robust breast ultrasound lesion segmentation.

The U-Net baseline achieved the best segmentation performance.

The threshold `0.50` provided the best balance for lesion detection, especially in terms of Dice and recall.

The threshold `0.75` was more conservative and improved precision, but reduced recall.

The MultiTask Attention U-Net achieved slightly lower segmentation performance than the baseline U-Net, but it also provided image-level classification.

This makes it useful when both lesion localization and diagnostic category prediction are required.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nawelbenchaabane/breast-ultrasound-ai.git
cd breast-ultrasound-ai
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

The project uses:

* Python,
* PyTorch,
* NumPy,
* Pandas,
* OpenCV,
* Albumentations,
* Scikit-learn,
* Scikit-image,
* Matplotlib,
* Streamlit.

---

## How to Use

The main workflow is available in:

```text
notebooks/Breast_Ultrasound_AI_Project.ipynb
```

The notebook covers:

1. dataset preparation,
2. mask preprocessing,
3. train/validation/test split,
4. classical segmentation experiments,
5. U-Net baseline training,
6. threshold optimization,
7. MultiTask Attention U-Net training,
8. segmentation and classification evaluation,
9. qualitative visualization,
10. final model comparison.

---

## Project Limitations

This project is experimental and intended for research and portfolio purposes.

The models are not intended for clinical use.

Main limitations include:

* relatively small dataset size,
* class imbalance,
* limited external validation,
* no prospective clinical evaluation,
* possible variability in ultrasound acquisition conditions,
* no expert radiologist validation beyond the available dataset annotations.

Future improvements could include:

* using larger multi-center datasets,
* external validation,
* stronger architectures such as U-Net++, DeepLabV3+, or Swin-UNet,
* uncertainty estimation,
* calibration of classification confidence,
* explainability methods,
* deployment through a lightweight web application.

---

## Author

Nawel Ben Chaabane

AI / Data Science project focused on medical image segmentation and classification.

