#!/bin/bash

# Example usage of the twin face verification training script

echo "AdaFace Twin Face Verification Training"
echo "======================================="

# Check if required files exist
if [ ! -f "id_to_images.json" ]; then
    echo "Error: id_to_images.json not found!"
    exit 1
fi

if [ ! -f "train_twin_id_pairs.json" ]; then
    echo "Error: train_twin_id_pairs.json not found!"
    exit 1
fi

echo "Files found. Starting training..."

# Quick test with 25% data, frozen backbone
echo ""
echo "Running quick test (25% data, frozen backbone, 10 epochs)..."
python train_twins.py \
    --train_percentage 25 \
    --freeze_backbone \
    --margin 0.3 \
    --batch_size 16 \
    --epochs 10 \
    --output_dir models \
    --log_dir training_logs \
    --device auto

echo ""
echo "Quick test completed!"

echo ""
echo "To run full training experiments:"
echo ""
echo "# Full model training with 50% data:"
echo "python train_twins.py --train_percentage 50 --freeze_backbone False --epochs 20"
echo ""
echo "# Frozen backbone with 100% data:"
echo "python train_twins.py --train_percentage 100 --freeze_backbone True --epochs 15"
echo ""
echo "# Skip alignment (if already done):"
echo "python train_twins.py --skip_alignment --train_percentage 75"
echo ""
echo "Results will be saved in:"
echo "- models/ (trained model checkpoints)"
echo "- training_logs/ (training curves and logs)"