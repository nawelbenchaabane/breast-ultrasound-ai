
import torch

PROJECT_NAME = "Breast Ultrasound AI"

SEED = 42

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10

CLASSES = ["benign", "malignant", "normal"]
CLASS_TO_IDX = {
    "benign": 0,
    "malignant": 1,
    "normal": 2
}

NUM_CLASSES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
