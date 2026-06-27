
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMG_SIZE, CLASS_TO_IDX


def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_valid_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


class BreastUltrasoundDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _read_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def _read_mask(self, mask_path, image_shape):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)

        mask = (mask > 0).astype(np.float32)
        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["processed_mask_path"]
        class_name = row["class_name"]

        image = self._read_image(image_path)
        mask = self._read_mask(mask_path, image.shape)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float()
        label = CLASS_TO_IDX[class_name]

        return {
            "image": image.float(),
            "mask": mask,
            "label": label,
            "class_name": class_name,
            "image_path": image_path
        }
