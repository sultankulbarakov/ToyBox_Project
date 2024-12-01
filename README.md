# Multi-View vs Single-View Object Classification

## Project Overview
This project compares multi-view and single-view approaches for object classification using deep learning. Using the COIL-100 dataset, which provides 72 rotated views of 100 objects, we implement and evaluate both approaches:
- Multi-view: Uses 3 views per object with late fusion strategy
- Single-view: Uses single views of objects

## Project Structure
```
project/
├── src/
│   ├── data_preprocessing.py  # Dataset handling and preprocessing
│   ├── model.py               # Multi-view model architecture
│   ├── single_view_train.py   # Single-view training implementation
│   ├── train.py               # Multi-view training implementation
│   └── evaluate.py            # Evaluation and comparison scripts
├── data/                      # Dataset directory
├── results/                   # Results and model checkpoints
├── ML_Final_ProjectReport.pdf 
└── README.md
```

## Environment Setup
1. Create and activate virtual environment:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (MacOS/Linux)
source .venv/bin/activate
```

2. Install required packages:
```bash
# Install main packages
pip install kagglehub torch torchvision numpy Pillow scikit-learn

# Install visualization packages
pip install matplotlib seaborn scikit-learn
```

## Code Modules Description

### data_preprocessing.py
Handles dataset preparation and loading.

Key Components:
- `COIL100Preprocessor`: Manages dataset organization
  - Input: Base directory path
  - Output: Organized dataset splits
  - Methods: `download_dataset()`, `organize_views()`, `create_splits()`

- `COIL100Dataset`: Custom dataset class
  - Input: Image paths dictionary, number of views
  - Output: Image tensors and labels

### model.py
Implements multi-view model architecture.

Key Components:
- `ViewEncoder`: Processes individual views using ResNet50
- `LateFusionModel`: Main model architecture
  - Input: Multiple view images (batch_size × num_views × channels × height × width)
  - Output: Class predictions
  - Features: Supports different fusion methods (concat, mean, max)

### single_view_train.py
Implements single-view model and training.

Key Components:
- `SingleViewCOIL100Dataset`: Dataset class for single views
- `SingleViewModel`: ResNet50-based single-view classifier
- `train_single_view()`: Training loop implementation

### train.py
Implements multi-view model training.

Key Components:
- Training configuration and optimization
- Metrics tracking
- Model checkpointing
- Validation procedures

### evaluate.py
Handles model evaluation and comparisons.

Key Features:
- Generates comparison plots
- Computes evaluation metrics
- Creates confusion matrices
- Outputs classification reports

## Usage Instructions

1. Data Preprocessing:
```bash
python3 src/data_preprocessing.py --base_dir ./data
```

2. Train Multi-view Model:
```bash
python3 src/train.py --data_dir ./data --save_dir ./results --epochs 8
```

3. Train Single-view Model:
```bash
python3 src/single_view_train.py --data_dir ./data --save_dir ./results --epochs 8
```

4. Compare Models:
```bash
python3 src/evaluate.py --data_dir ./data --save_dir ./results --compare_models
```

## Current Results
Based on 8 epochs of training:

Multi-view (3 views):
- Final training loss: 1.85
- Final validation loss: 11.49
- Final validation accuracy: 6.67%

Single-view:
- Final training loss: 1.12
- Final validation loss: 8.17
- Final validation accuracy: 6.67%

Both approaches show evidence of overfitting, with decreasing training loss but increasing validation loss.

## The Final Project is Done but we still have some Future Improvements
1. Experiment with different fusion methods
2. Add regularization techniques
3. Reduce model complexity
4. Increase training time / epochs