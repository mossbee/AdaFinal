# AdaFace Twin Fine-tuning Implementation Plan

## Overview
This document outlines the implementation plan for fine-tuning the AdaFace model on identical twin face verification using triplet loss.

## Key Requirements

### 1. Data Handling Strategy
- **Single image handling**: 
  - Skip twin pairs where both persons have only 1 image
  - If one twin has 1 image and the other has 2+: use twin's multiple images as anchor/positive, single image as negative
- **Triplet sampling**: (anchor, positive, negative) = (person_A_img1, person_A_img2, twin_of_A_img)
- **Data percentages**: Support sampling 10%, 25%, 50%, 75%, 100% of maximum possible triplets per twin pairs.
- **Validation split**: 10-15% of training data for validation

### 2. Fine-tuning Modes
- **Full model**: Train entire network end-to-end
- **Frozen backbone**: Freeze conv layers, only train embedding layer (512-dim feature output)

### 3. Training Configuration
- **Triplet margin**: Configurable (default: 0.3)
- **Hard negative mining**: Sample negatives from twin persons
- **Batch size**: 16 triplets
- **Learning rates**: 1e-4 (full), 1e-3 (frozen backbone)
- **Separate alignment**: Train and test use different aligned folders

## Files to Create

### 1. `train_twins.py` (Main Training Script)
**Purpose**: Two-stage fine-tuning with comprehensive options

**Stage 1: Image Alignment**
- Align training images → `train_aligned_images/`
- Generate `id_to_train_aligned_images.json`
- Option to skip alignment and use existing data

**Stage 2: Fine-tuning**
- Load aligned images and twin pairs
- Generate triplets with sampling ratio (10% 20% 100% et)
- Train with triplet loss and validation
- Save models with descriptive names

**Key Components**:
- `TwinTrainer` class with initialization, alignment, and training methods
- Triplet sampling strategy respecting single-image constraints
- Configurable data percentage sampling
- Two fine-tuning modes (full vs backbone-frozen)
- Validation split and periodic evaluation
- Model checkpointing with descriptive names

### 2. `triplet_dataset.py` (Dataset and Sampling Logic)
**Purpose**: Efficient triplet generation and loading

**Key Features**:
- `TwinTripletDataset` class extending PyTorch Dataset
- Smart triplet sampling:
  - Skip twin pairs where both have only 1 image
  - Implement percentage-based subsampling
- No - data augmentation needed
- Validation set creation (10-15% split)
- Efficient batch loading and caching

### 3. `triplet_loss.py` (Loss Function)
**Purpose**: Configurable triplet loss implementation

**Features**:
- Standard triplet margin loss with configurable margin
- Distance metrics and logging utilities
- Support for different distance metrics (cosine, euclidean)
- **No hard negative mining needed** - twins provide natural hard negatives

## Architecture Understanding

### AdaFace Model Structure
- **Backbone**: ResNet-based feature extractor
- **Embedding layer**: Final layer producing 512-dim features
- **No classification head**: We only need embeddings for verification

### Fine-tuning Strategies
- **Full training**: Train entire network end-to-end (all parameters)
- **Frozen backbone**: Freeze all conv/residual blocks, only train final embedding layer
- **Learning rates**: Different rates for different strategies

## Training Configuration Details

### Data Splits
```
Training: 85-90% of available data
Validation: 10-15% of training data
Test: Separate test set (not used during training)
```

### Hyperparameters
```
Triplet margin: 0.3 (configurable)
Batch size: 16-32 triplets
Epochs: 10-50 (early stopping based on validation)
Learning rates:
  - Full model: 1e-4
  - Frozen backbone: 1e-3
Optimizer: Adam with weight decay
```

### Data Percentages
Support training with different data sizes for faster experimentation:
- 10%: Quick prototyping
- 25%: Small-scale experiments
- 50%: Medium-scale training
- 75%: Large-scale training
- 100%: Full dataset training

## Output Organization

### Model Naming Convention
```
models/
├── twin_tuned_ir50_full_100pct_margin0.3_epoch10.pth
├── twin_tuned_ir50_frozen_50pct_margin0.5_epoch15.pth
└── ...
```

### Training Logs Structure
```
training_logs/
├── full_100pct_margin0.3/
│   ├── training_log.json
│   ├── validation_curves.png
│   ├── loss_curves.png
│   └── config.json
└── frozen_50pct_margin0.5/
    ├── training_log.json
    ├── validation_curves.png
    ├── loss_curves.png
    └── config.json
```

## Command Line Interface

### Basic Usage
```bash
# Full training with 50% data
python train_twins.py \
    --train_percentage 50 \
    --freeze_backbone False \
    --margin 0.3 \
    --epochs 20

# Frozen backbone with 25% data  
python train_twins.py \
    --train_percentage 25 \
    --freeze_backbone True \
    --margin 0.5 \
    --epochs 15

# Skip alignment and use existing
python train_twins.py \
    --skip_alignment \
    --train_percentage 100 \
    --aligned_json id_to_train_aligned_images.json
```

### Full Parameter List
```bash
python train_twins.py \
    --id_to_images id_to_images.json \
    --train_pairs train_twin_id_pairs.json \
    --aligned_folder train_aligned_images \
    --aligned_json id_to_train_aligned_images.json \
    --skip_alignment \
    --train_percentage 75 \
    --freeze_backbone True \
    --margin 0.3 \
    --batch_size 48 \
    --epochs 25 \
    --learning_rate 1e-3 \
    --validation_split 0.15 \
    --output_dir models \
    --log_dir training_logs \
    --device cuda \
    --max_workers 6
```

## Integration with Evaluation

### Model Loading
- Fine-tuned models can be loaded directly into `evaluate_twins.py`
- Consistent model architecture and checkpoint format
- Support for evaluating multiple fine-tuned models

### Data Consistency
- Consistent alignment pipeline ensures compatibility
- Separate train/test aligned folders prevent data leakage
- Same JSON format for aligned image mappings

### Validation During Training
- Periodic evaluation on validation set
- Early stopping based on validation metrics
- Track training progress with comprehensive logging

## Implementation Steps

### Phase 1: Core Infrastructure
1. Create `triplet_loss.py` with configurable triplet loss
2. Implement `triplet_dataset.py` with smart sampling logic
3. Set up basic training loop structure

### Phase 2: Training Pipeline
1. Implement `TwinTrainer` class in `train_twins.py`
2. Add alignment stage (reusing existing functions)
3. Integrate triplet sampling and data loading

### Phase 3: Configuration and Logging
1. Add comprehensive command-line arguments
2. Implement training logging and visualization
3. Add model checkpointing with descriptive names

### Phase 4: Testing and Optimization
1. Test with small data percentages
2. Validate integration with evaluation pipeline
3. Optimize performance and memory usage

## Expected Outcomes

### Model Variants
- Multiple fine-tuned models with different configurations
- Comparison between full and frozen backbone training
- Analysis of optimal data percentage vs performance trade-offs
- Make sure fine-tund models work with current model loading and inference

### Performance Improvements
- Better twin face verification accuracy
- Improved feature representations for challenging twin pairs
- Reduced false positive rates on twin negatives

### Research Insights
- Understanding of optimal fine-tuning strategies for face verification
- Analysis of triplet loss effectiveness on twin data
- Comparison of different training data sizes and their impact
