# Multi-Task Pancreas Segmentation and Classification

A multi-task deep learning model for pancreas segmentation and cancer subtype classification using nnU-Net v2 framework.

## Overview

This repository implements a multi-task neural network that simultaneously performs:
- **Pancreas segmentation**: Separating normal pancreas (label 1) and pancreas lesions (label 2) from background
- **Cancer subtype classification**: Classifying lesions into 3 subtypes (0, 1, 2)

The model extends nnU-Net v2's ResidualEncoderUNet with a shared encoder and dual decoder heads for joint optimization.

**Note**: The training and evaluation scripts provided are wrapper scripts built on top of the nnU-Net v2 framework. These wrappers facilitate environment setup, provide simplified command-line interfaces, and enable multi-task customizations while leveraging the original nnUNetv2's preprocessing, training, and inference pipelines.

## Requirements

Install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

Key dependencies:
- nnU-Net v2
- PyTorch >= 1.9.0
- SimpleITK
- scikit-learn
- pandas

## Data Preparation

1. Prepare the dataset in nnU-Net format:

```bash
python prepare_data_and_plans.py
```

This script will:
- Convert the provided dataset to nnU-Net format
- Generate dataset fingerprints
- Create experiment plans
- Run preprocessing

## Usage

### Training

Train the multi-task model with default settings:

```bash
python train_multitask.py
```

#### Training Options

- **Classification loss weight**: `--cls_loss_weight 0.3` (default: 0.3)
- **Cross-validation fold**: `--fold 0` (default: 0)
- **Training epochs**: `--num_epochs 200`
- **Device selection**: `--device cuda` or `--device cpu` (default: auto)
- **Verbose logging**: `--verbose`

Example with custom parameters:

```bash
python train_multitask.py --fold 1 --cls_loss_weight 0.5 --num_epochs 150 --verbose
```

### Evaluation

#### Validation Evaluation

Evaluate on validation set with ground truth comparison:

```bash
python eval.py
```

#### Test Prediction

Generate predictions for test set (submission format):

```bash
python eval.py --test_mode
```

#### Evaluation Options

- **Fast inference**: `--fast` (disables TTA, increases speed by ~30%)
- **Timing display**: `--show_timing` (shows per-case inference time)
- **Custom model**: `--model_folder path/to/model`
- **Output directory**: `--output_dir custom_results`

Example fast test prediction:

```bash
python eval.py --test_mode --fast --show_timing
```

## Results

### Segmentation Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Whole Pancreas DSC | 0.91 | 0.91 |
| Pancreas Lesion DSC | 0.31 | 0.59 |

### Classification Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Macro-average F1 | 0.70 | 0.73 |

### Inference Speed

| Mode | Total Time | Speed Improvement |
|------|------------|------------------|
| Standard | 56.38s | Baseline |
| Fast Mode | 40.39s | 28.4% faster |

To reproduce results:

```bash
# Train model
python train_multitask.py --fold 0

# Evaluate on validation set
python eval.py --verbose

# Generate test predictions
python eval.py --test_mode --output_dir submission_results
```

## Model Architecture

The model extends nnU-Net v2 with:

- **Shared Encoder**: ResidualEncoderUNet backbone for feature extraction
- **Segmentation Head**: Standard nnU-Net decoder with deep supervision
- **Classification Head**: Multi-scale feature fusion with adaptive pooling and MLP

Key features:
- Multi-scale feature fusion using the last 3 encoder outputs
- Feature compression and adaptive pooling for each scale
- Weighted loss combination (segmentation + classification)
- Direct encoder feature access for reliable training


## File Structure

```
├── train_multitask.py                      # Training script
├── eval.py                                 # Evaluation script  
├── prepare_data_and_plans.py               # Data preparation
├── requirements.txt                        # Dependencies
├── nnunetv2/                               # Local nnU-Net v2 library
│   └── training/
│       └── nnUNetTrainer/
│           ├── multitask_trainer.py        # Custom multi-task trainer
│           └── src/
│               └── models/
│                   └── nnunet_multitask.py # Multi-task model architecture
└── nnUNet_raw_data/                        # Raw dataset (nnU-Net format)
    └── Dataset001_Pancreas/
        ├── imagesTr/                       # Training images
        ├── labelsTr/                       # Training labels
        ├── imagesTs/                       # Test images
        ├── dataset.json                    # Dataset metadata
        └── subtype_info.json               # Classification labels
```
