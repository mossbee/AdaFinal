#!/bin/bash

# Full Pipeline: AdaFace Twin Fine-tuning and Evaluation
# This script performs:
# 1. Train image alignment (if needed)
# 2. Model fine-tuning on training data
# 3. Test image alignment (if needed) 
# 4. Evaluation on test data with fine-tuned models

set -e  # Exit on any error

echo "=========================================="
echo "AdaFace Twin Fine-tuning & Evaluation Pipeline"
echo "=========================================="

# Configuration - Modify these as needed
ID_TO_IMAGES="id_to_images.json"
TRAIN_PAIRS="train_twin_id_pairs.json"
TEST_PAIRS="test_twin_id_pairs.json"

# Directories
TRAIN_ALIGNED_FOLDER="train_aligned_images"
TEST_ALIGNED_FOLDER="aligned_test_images"
MODELS_DIR="models"
LOGS_DIR="training_logs"
EVAL_DIR="evaluation_results"

# Training configurations to test
TRAIN_PERCENTAGES=(25 50 75 100)
FREEZE_OPTIONS=(true false)
MARGINS=(0.3 0.5)

# Other settings
MAX_WORKERS=4
BATCH_SIZE=16
EPOCHS=15
DEVICE="auto"

echo "Configuration:"
echo "  ID to Images: $ID_TO_IMAGES"
echo "  Train Pairs: $TRAIN_PAIRS"
echo "  Test Pairs: $TEST_PAIRS"
echo "  Device: $DEVICE"
echo "  Max Workers: $MAX_WORKERS"
echo "  Epochs: $EPOCHS"
echo ""

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: Required file '$1' not found!"
        echo "Please ensure all required files are present:"
        echo "  - $ID_TO_IMAGES"
        echo "  - $TRAIN_PAIRS"
        echo "  - $TEST_PAIRS"
        exit 1
    fi
}

# Function to run training with specific configuration
run_training() {
    local percentage=$1
    local freeze=$2
    local margin=$3
    
    local mode
    if [ "$freeze" = "true" ]; then
        mode="frozen"
    else
        mode="full"
    fi
    
    echo "----------------------------------------"
    echo "Training: ${mode} model, ${percentage}% data, margin=${margin}"
    echo "----------------------------------------"
    
    local freeze_flag=""
    if [ "$freeze" = "true" ]; then
        freeze_flag="--freeze_backbone"
    fi
    
    # Run training
    python train_twins.py \
        --id_to_images "$ID_TO_IMAGES" \
        --train_pairs "$TRAIN_PAIRS" \
        --aligned_folder "$TRAIN_ALIGNED_FOLDER" \
        --aligned_json "id_to_train_aligned_images.json" \
        --skip_alignment \
        --train_percentage $percentage \
        $freeze_flag \
        --margin $margin \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --output_dir "$MODELS_DIR" \
        --log_dir "$LOGS_DIR" \
        --device "$DEVICE" \
        --max_workers $MAX_WORKERS
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully: ${mode}_${percentage}pct_margin${margin}"
    else
        echo "✗ Training failed: ${mode}_${percentage}pct_margin${margin}"
        return 1
    fi
}

# Function to run evaluation on a specific model
run_evaluation() {
    local model_path=$1
    local model_name=$(basename "$model_path" .pth)
    local eval_output="${EVAL_DIR}/${model_name}"
    
    echo "----------------------------------------"
    echo "Evaluating: $model_name"
    echo "----------------------------------------"
    
    # Extract model architecture from filename (assuming format contains ir_50, ir_101, etc.)
    local architecture="ir_50"  # default
    if [[ "$model_name" == *"ir_101"* ]]; then
        architecture="ir_101"
    elif [[ "$model_name" == *"ir_se_50"* ]]; then
        architecture="ir_se_50"
    fi
    
    # Run evaluation
    python evaluate_twins.py \
        --id_to_images "$ID_TO_IMAGES" \
        --test_pairs "$TEST_PAIRS" \
        --aligned_folder "$TEST_ALIGNED_FOLDER" \
        --aligned_json "id_to_aligned_images.json" \
        --skip_alignment \
        --model "$architecture" \
        --output_dir "$eval_output" \
        --batch_size 32 \
        --device "$DEVICE" \
        --custom_model_path "$model_path"
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed: $model_name"
        echo "  Results saved to: $eval_output"
    else
        echo "✗ Evaluation failed: $model_name"
        return 1
    fi
}

# Function to summarize results
summarize_results() {
    echo ""
    echo "=========================================="
    echo "PIPELINE SUMMARY"
    echo "=========================================="
    
    echo ""
    echo "Trained Models:"
    if [ -d "$MODELS_DIR" ]; then
        find "$MODELS_DIR" -name "*.pth" -type f | while read model; do
            echo "  $(basename "$model")"
        done
    else
        echo "  No models found"
    fi
    
    echo ""
    echo "Training Logs:"
    if [ -d "$LOGS_DIR" ]; then
        find "$LOGS_DIR" -name "training_log.json" -type f | while read log; do
            local dir=$(dirname "$log")
            echo "  $(basename "$dir")"
        done
    else
        echo "  No logs found"
    fi
    
    echo ""
    echo "Evaluation Results:"
    if [ -d "$EVAL_DIR" ]; then
        find "$EVAL_DIR" -name "evaluation_results.json" -type f | while read result; do
            local dir=$(dirname "$result")
            echo "  $(basename "$dir")"
        done
    else
        echo "  No evaluation results found"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Check training curves: $LOGS_DIR/*/training_curves.png"
    echo "2. Compare evaluation metrics: $EVAL_DIR/*/evaluation_results.json"
    echo "3. Analyze pair results: $EVAL_DIR/*/pair_results.csv"
}

# Main pipeline execution
main() {
    echo "Starting full pipeline..."
    
    # Check required files
    echo "Checking required files..."
    check_file "$ID_TO_IMAGES"
    check_file "$TRAIN_PAIRS"
    check_file "$TEST_PAIRS"
    echo "✓ All required files found"
    
    # Create directories
    mkdir -p "$MODELS_DIR" "$LOGS_DIR" "$EVAL_DIR"
    
    # Step 1: Align training images (if not already done)
    echo ""
    echo "=========================================="
    echo "STEP 1: TRAIN IMAGE ALIGNMENT"
    echo "=========================================="
    
    if [ ! -f "id_to_train_aligned_images.json" ]; then
        echo "Aligning training images..."
        python train_twins.py \
            --id_to_images "$ID_TO_IMAGES" \
            --train_pairs "$TRAIN_PAIRS" \
            --aligned_folder "$TRAIN_ALIGNED_FOLDER" \
            --aligned_json "id_to_train_aligned_images.json" \
            --train_percentage 10 \
            --epochs 1 \
            --max_workers $MAX_WORKERS \
            --device "$DEVICE" \
            > /dev/null 2>&1 || true  # Just for alignment, ignore training errors
        
        if [ -f "id_to_train_aligned_images.json" ]; then
            echo "✓ Training images aligned successfully"
        else
            echo "✗ Failed to align training images"
            exit 1
        fi
    else
        echo "✓ Training images already aligned"
    fi
    
    # Step 2: Align test images (if not already done)
    echo ""
    echo "=========================================="
    echo "STEP 2: TEST IMAGE ALIGNMENT"
    echo "=========================================="
    
    if [ ! -f "id_to_aligned_images.json" ]; then
        echo "Aligning test images..."
        python evaluate_twins.py \
            --id_to_images "$ID_TO_IMAGES" \
            --test_pairs "$TEST_PAIRS" \
            --aligned_folder "$TEST_ALIGNED_FOLDER" \
            --aligned_json "id_to_aligned_images.json" \
            --max_workers $MAX_WORKERS \
            --device "$DEVICE" \
            > /dev/null 2>&1 || true  # Just for alignment, ignore evaluation errors
        
        if [ -f "id_to_aligned_images.json" ]; then
            echo "✓ Test images aligned successfully"
        else
            echo "✗ Failed to align test images"
            exit 1
        fi
    else
        echo "✓ Test images already aligned"
    fi
    
    # Step 3: Model Fine-tuning
    echo ""
    echo "=========================================="
    echo "STEP 3: MODEL FINE-TUNING"
    echo "=========================================="
    
    local training_success=0
    local training_total=0
    
    # Run training with different configurations
    for percentage in "${TRAIN_PERCENTAGES[@]}"; do
        for freeze in "${FREEZE_OPTIONS[@]}"; do
            for margin in "${MARGINS[@]}"; do
                training_total=$((training_total + 1))
                
                # Skip some combinations to reduce training time
                # Only test full combinations for 50% and 100% data
                if [ "$percentage" -lt 50 ] && [ "$freeze" = "false" ]; then
                    echo "Skipping: full model with ${percentage}% data (time optimization)"
                    continue
                fi
                
                if run_training "$percentage" "$freeze" "$margin"; then
                    training_success=$((training_success + 1))
                fi
                
                echo ""
            done
        done
    done
    
    echo "Training Summary: $training_success successful out of attempted configurations"
    
    # Step 4: Model Evaluation
    echo ""
    echo "=========================================="
    echo "STEP 4: MODEL EVALUATION"
    echo "=========================================="
    
    # Find all trained models
    local models_found=0
    local eval_success=0
    
    if [ -d "$MODELS_DIR" ]; then
        while IFS= read -r -d '' model_path; do
            models_found=$((models_found + 1))
            echo ""
            if run_evaluation "$model_path"; then
                eval_success=$((eval_success + 1))
            fi
        done < <(find "$MODELS_DIR" -name "*.pth" -type f -print0 2>/dev/null)
    fi
    
    if [ $models_found -eq 0 ]; then
        echo "No trained models found for evaluation"
        echo "Training may have failed. Check training logs in: $LOGS_DIR"
    else
        echo ""
        echo "Evaluation Summary: $eval_success/$models_found models evaluated successfully"
    fi
    
    # Step 5: Results Summary
    echo ""
    echo "=========================================="
    echo "STEP 5: RESULTS SUMMARY"
    echo "=========================================="
    
    summarize_results
    
    # Final status
    echo ""
    echo "=========================================="
    echo "PIPELINE COMPLETED"
    echo "=========================================="
    
    if [ $training_success -gt 0 ] && [ $eval_success -gt 0 ]; then
        echo "✓ Pipeline completed successfully!"
        echo "✓ $training_success models trained"
        echo "✓ $eval_success models evaluated"
        
        # Find best model based on evaluation results
        echo ""
        echo "Finding best performing model..."
        python -c "
import os
import json
import glob

eval_dirs = glob.glob('$EVAL_DIR/*/evaluation_results.json')
best_auc = 0
best_model = None

for eval_file in eval_dirs:
    try:
        with open(eval_file, 'r') as f:
            results = json.load(f)
        
        auc = results['metrics']['auc']
        model_name = os.path.basename(os.path.dirname(eval_file))
        
        print(f'  {model_name}: AUC={auc:.4f}')
        
        if auc > best_auc:
            best_auc = auc
            best_model = model_name
    except:
        continue

if best_model:
    print(f'\\n✓ Best model: {best_model} (AUC={best_auc:.4f})')
else:
    print('\\n! No valid evaluation results found')
" 2>/dev/null || echo "Could not analyze results automatically"
        
    else
        echo "! Pipeline completed with some failures"
        echo "! Training successes: $training_success"
        echo "! Evaluation successes: $eval_success"
        echo "! Check logs for details"
    fi
}

# Handle script arguments
case "${1:-run}" in
    "run")
        main
        ;;
    "clean")
        echo "Cleaning up generated files..."
        rm -rf "$MODELS_DIR" "$LOGS_DIR" "$EVAL_DIR"
        rm -f "id_to_train_aligned_images.json" "id_to_aligned_images.json"
        rm -rf "$TRAIN_ALIGNED_FOLDER" "$TEST_ALIGNED_FOLDER"
        echo "✓ Cleanup completed"
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  run (default) - Run the full pipeline"
        echo "  clean         - Clean up all generated files"
        echo "  help          - Show this help message"
        echo ""
        echo "Configuration (edit script to modify):"
        echo "  Train percentages: ${TRAIN_PERCENTAGES[*]}"
        echo "  Freeze options: ${FREEZE_OPTIONS[*]}"
        echo "  Margins: ${MARGINS[*]}"
        echo "  Epochs: $EPOCHS"
        echo "  Batch size: $BATCH_SIZE"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
