#!/bin/bash

# Example usage of the twin face verification evaluation script

echo "AdaFace Twin Face Verification Evaluation"
echo "========================================"

# Basic usage with default settings (includes both alignment and evaluation)
echo "Running full evaluation (alignment + evaluation)..."
python evaluate_twins.py \
    --id_to_images id_to_images.json \
    --test_pairs test_twin_id_pairs.json \
    --aligned_folder aligned_test_images \
    --aligned_json id_to_aligned_images.json \
    --output_dir evaluation_results \
    --max_workers 4 \
    --batch_size 32 \
    --device auto

echo ""
echo "To skip alignment and run evaluation only:"
echo "python evaluate_twins.py --skip_alignment --aligned_json id_to_aligned_images.json"

echo ""
echo "To use GPU if available:"
echo "python evaluate_twins.py --device cuda"

echo ""
echo "To overwrite existing aligned images:"
echo "python evaluate_twins.py --overwrite"

echo ""
echo "Results will be saved in:"
echo "- evaluation_results/evaluation_results.json (metrics)"
echo "- evaluation_results/roc_curve.png"
echo "- evaluation_results/similarity_distributions.png" 
echo "- evaluation_results/confusion_matrix.png"
