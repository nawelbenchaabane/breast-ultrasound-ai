# Data

The raw BUSI dataset is not included in this repository.

## Dataset

This project uses the Breast Ultrasound Images Dataset, also known as BUSI.

The dataset contains breast ultrasound images grouped into three classes:

- benign
- malignant
- normal

For benign and malignant cases, segmentation masks are provided.

## Expected structure

After downloading the dataset, the expected local structure is:

```text
data/
└── BUSI_raw/
    └── Dataset_BUSI_with_GT/
        ├── benign/
        ├── malignant/
        └── normal/
