#!/bin/bash

# Run AdaFace Twin Verification Evaluation
# This script runs the complete evaluation pipeline

echo "=== AdaFace Identical Twin Verification Evaluation ==="
echo ""

# Set default parameters
TWIN_PAIRS="test_twin_id_pairs.json"
ID_TO_IMAGES="id_to_images.json"
ALIGNED_DIR="aligned_test_images"
RESULTS_DIR="twin_verification_results"
MODEL_ARCH="ir_50"
BATCH_SIZE=32
MAX_WORKERS=4

# Check if required files exist
if [ ! -f "$TWIN_PAIRS" ]; then
    echo "Error: $TWIN_PAIRS not found!"
    echo "Make sure you have the twin pairs JSON file in the current directory."
    exit 1
fi

if [ ! -f "$ID_TO_IMAGES" ]; then
    echo "Error: $ID_TO_IMAGES not found!"
    echo "Make sure you have the ID to images mapping JSON file in the current directory."
    exit 1
fi

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, using CUDA"
    DEVICE="cuda"
else
    echo "No NVIDIA GPU detected, using CPU"
    DEVICE="cpu"
fi

echo "Configuration:"
echo "  Twin pairs file: $TWIN_PAIRS"
echo "  ID to images file: $ID_TO_IMAGES"
echo "  Aligned images directory: $ALIGNED_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Model architecture: $MODEL_ARCH"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Max workers: $MAX_WORKERS"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run the evaluation
echo "Starting evaluation..."
python evaluate_twins.py \
    --twin_pairs "$TWIN_PAIRS" \
    --id_to_images "$ID_TO_IMAGES" \
    --aligned_dir "$ALIGNED_DIR" \
    --results_dir "$RESULTS_DIR" \
    --model_arch "$MODEL_ARCH" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --max_workers "$MAX_WORKERS"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Evaluation completed successfully! ==="
    echo "Results saved to: $RESULTS_DIR"
    echo ""
    echo "Generated files:"
    echo "  - verification_results.json: Detailed metrics"
    echo "  - evaluation_plots.png: ROC, PR curves, and distributions"
    echo "  - detailed_distributions.png: Similarity score distributions"
    echo "  - id_to_aligned_images.json: Mapping of alignable images"
else
    echo ""
    echo "=== Evaluation failed! ==="
    echo "Check the error messages above for details."
    exit 1
fi
