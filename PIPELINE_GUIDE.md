# AdaFace Twin Fine-tuning Pipeline - Quick Start Guide

## Overview
This pipeline performs complete fine-tuning and evaluation of AdaFace models on identical twin face verification:

1. **Training Image Alignment**: Aligns training images for consistency
2. **Model Fine-tuning**: Trains models with different configurations using triplet loss
3. **Test Image Alignment**: Aligns test images for evaluation
4. **Model Evaluation**: Tests all fine-tuned models and compares performance

## Prerequisites

### Required Files
Ensure these files are in the AdaFace directory:
- `id_to_images.json` - Maps person IDs to their image paths
- `train_twin_id_pairs.json` - Training twin pairs
- `test_twin_id_pairs.json` - Test twin pairs

### Dependencies
Install required packages:
```bash
pip install torch torchvision numpy opencv-python Pillow matplotlib seaborn scikit-learn scipy tqdm pandas
```

## Quick Start

### Run Full Pipeline
```bash
./run_full_pipeline.sh
```

This will:
- Align training and test images
- Train models with different configurations
- Evaluate all trained models
- Generate comprehensive results

### Pipeline Configuration
The script tests these configurations by default:

**Training Data**: 25%, 50%, 75%, 100% of available triplets
**Fine-tuning Modes**: 
- Frozen backbone (only train embedding layer)
- Full model training (all parameters)
**Triplet Margins**: 0.3, 0.5
**Training**: 15 epochs, batch size 16

### Output Structure
```
models/                           # Trained model checkpoints
├── twin_tuned_ir50_frozen_25pct_margin0.3_epoch15.pth
├── twin_tuned_ir50_full_50pct_margin0.3_epoch12.pth
└── ...

training_logs/                    # Training logs and curves
├── frozen_25pct_margin0.3/
│   ├── config.json
│   ├── training_log.json
│   └── training_curves.png
└── ...

evaluation_results/               # Evaluation results
├── twin_tuned_ir50_frozen_25pct_margin0.3_epoch15/
│   ├── evaluation_results.json
│   ├── roc_curve.png
│   ├── similarity_distributions.png
│   ├── pair_results.csv
│   └── confusion_matrix.png
└── ...
```

## Manual Usage

### Individual Training
```bash
# Quick test with 25% data, frozen backbone
python train_twins.py --train_percentage 25 --freeze_backbone --epochs 10

# Full model training with 100% data
python train_twins.py --train_percentage 100 --epochs 20
```

### Individual Evaluation
```bash
# Evaluate pretrained model
python evaluate_twins.py --test_pairs test_twin_id_pairs.json

# Evaluate fine-tuned model
python evaluate_twins.py \
    --custom_model_path models/twin_tuned_ir50_full_100pct_margin0.3_epoch15.pth \
    --skip_alignment
```

## Key Features

### Smart Triplet Sampling
- Automatically skips twin pairs where both persons have only 1 image
- Generates balanced triplets: (anchor, positive, negative) = (person_A_img1, person_A_img2, twin_of_A_img)
- Supports percentage-based subsampling for faster experimentation

### Two Fine-tuning Modes
- **Frozen Backbone**: Freezes feature extraction layers, only trains embedding layer
- **Full Training**: Trains entire network end-to-end

### Comprehensive Evaluation
- ROC curves and AUC metrics
- Equal Error Rate (EER) analysis
- Precision, Recall, F1-score
- True Acceptance Rate at specific False Acceptance Rates
- Detailed pair-by-pair results in CSV format

### Integration Ready
- Fine-tuned models work seamlessly with existing inference pipeline
- Consistent checkpoint format for easy model loading
- Automatic model architecture detection

## Results Analysis

### Best Model Selection
The pipeline automatically identifies the best performing model:
```bash
# View results summary
cat evaluation_results/*/evaluation_results.json | grep -A 5 '"auc"'
```

### Training Progress
Check training curves:
```bash
# View training logs
ls training_logs/*/training_curves.png
```

### Detailed Analysis
Analyze specific failure cases:
```bash
# Load pair results
python -c "
import pandas as pd
df = pd.read_csv('evaluation_results/MODEL_NAME/pair_results.csv')
print('Hardest positive pairs:')
print(df[df['pair_type']=='positive'].head(10))
print('\nEasiest negative pairs:')  
print(df[df['pair_type']=='negative'].tail(10))
"
```

## Customization

### Modify Configurations
Edit `run_full_pipeline.sh`:
```bash
# Training configurations to test
TRAIN_PERCENTAGES=(10 25 50)  # Reduce for faster testing
FREEZE_OPTIONS=(true)         # Only test frozen backbone
MARGINS=(0.3)                 # Single margin value
EPOCHS=10                     # Reduce epochs
```

### Skip Steps
```bash
# Clean everything and start fresh
./run_full_pipeline.sh clean

# Get help
./run_full_pipeline.sh help
```

## Troubleshooting

### Common Issues

**"Required file not found"**
- Ensure `id_to_images.json`, `train_twin_id_pairs.json`, `test_twin_id_pairs.json` exist
- Check file paths in JSON files point to existing images

**"No training data available"**
- Check that twin pairs reference IDs that exist in `id_to_images.json`
- Ensure at least one person in each twin pair has 2+ images

**"CUDA out of memory"**
- Reduce batch size: edit `BATCH_SIZE=8` in pipeline script
- Use CPU: edit `DEVICE="cpu"` in pipeline script

**Training fails**
- Check `training_logs/*/training_log.json` for detailed error messages
- Verify image paths in aligned JSON files

### Performance Tips

**Faster Training**
- Use smaller data percentages (25%, 50%) for initial experiments
- Reduce epochs for quick prototyping
- Use frozen backbone mode (faster than full training)

**Better Results**
- Use 100% training data for final models
- Try different margin values (0.2, 0.3, 0.5)
- Train for more epochs with early stopping

## Expected Results

### Performance Improvements
After fine-tuning, expect to see:
- **Improved AUC**: 0.02-0.05 increase over pretrained model
- **Better EER**: Lower equal error rates on twin verification
- **Reduced False Positives**: Fewer twin faces incorrectly classified as same person

### Training Insights
- **Frozen backbone**: Faster training, good for limited data
- **Full training**: Better performance with sufficient data
- **Margin effects**: Larger margins = more conservative similarity thresholds
- **Data percentage**: More data generally = better performance, with diminishing returns

This pipeline provides a complete solution for fine-tuning AdaFace on challenging twin face verification tasks!
