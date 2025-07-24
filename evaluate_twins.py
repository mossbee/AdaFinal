#!/usr/bin/env python3
"""
Identical Twin Faces Verification Evaluation Script for AdaFace

This script evaluates the AdaFace model on identical twin face verification task.
It consists of two stages:
1. Alignment: Align and save all alignable test images
2. Evaluation: Extract features and compute verification metrics

Metrics computed:
- Accuracy (best threshold)
- Equal Error Rate (EER)
- Area Under ROC Curve (AUC-ROC)
- True Acceptance Rate (TAR) at specific False Acceptance Rates (FAR)
- Precision, Recall, F1-score
- Distance distributions visualization
"""

import os
import json
import numpy as np
import torch
import argparse
import warnings
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations
warnings.filterwarnings('ignore')

# Import from existing modules
from inference import load_pretrained_model, get_features_from_aligned_images, align_image_paths_with_structure
from face_alignment import align


def load_twin_pairs(twin_pairs_file):
    """Load twin pairs from JSON file"""
    with open(twin_pairs_file, 'r') as f:
        twin_pairs = json.load(f)
    return twin_pairs


def load_id_to_images(id_to_images_file):
    """Load ID to images mapping from JSON file"""
    with open(id_to_images_file, 'r') as f:
        id_to_images = json.load(f)
    return id_to_images


def get_all_test_images_from_twins(twin_pairs, id_to_images):
    """Extract all unique image paths from twin pairs"""
    all_test_ids = set()
    for id1, id2 in twin_pairs:
        all_test_ids.add(id1)
        all_test_ids.add(id2)
    
    all_image_paths = []
    for test_id in all_test_ids:
        if test_id in id_to_images:
            all_image_paths.extend(id_to_images[test_id])
    
    return list(set(all_image_paths))  # Remove duplicates


def align_test_images(image_paths, aligned_dir, max_workers=4):
    """
    Stage 1: Align all test images preserving original directory structure
    Uses parallel processing for efficiency
    Returns: mapping of original paths to aligned paths and failed images list
    """
    print("=== Stage 1: Aligning Test Images ===")
    
    # Use the parallel alignment function from inference.py
    alignable_images, failed_images = align_image_paths_with_structure(
        image_paths, aligned_dir, max_workers=max_workers, overwrite=False
    )
    
    return alignable_images, failed_images


def create_alignable_id_mapping(twin_pairs, original_id_to_images, alignable_images):
    """Create new ID to images mapping with only alignable images"""
    new_id_to_images = {}
    
    # Get all test IDs from twin pairs
    all_test_ids = set()
    for id1, id2 in twin_pairs:
        all_test_ids.add(id1)
        all_test_ids.add(id2)
    
    # Filter to only alignable images
    for test_id in all_test_ids:
        if test_id in original_id_to_images:
            alignable_paths = []
            for img_path in original_id_to_images[test_id]:
                if img_path in alignable_images:
                    aligned_path = alignable_images[img_path]
                    alignable_paths.append(aligned_path)
            
            if alignable_paths:  # Only include IDs with at least one alignable image
                new_id_to_images[test_id] = alignable_paths
    
    return new_id_to_images


def generate_pairs_and_labels(twin_pairs, id_to_aligned_images):
    """
    Generate positive and negative pairs for evaluation
    Returns: pairs list and labels list (1 for same person, 0 for different person)
    """
    pairs = []
    labels = []
    
    print("=== Generating Evaluation Pairs ===")
    
    # Generate positive pairs (same person)
    for test_id in id_to_aligned_images:
        if len(id_to_aligned_images[test_id]) >= 2:
            # All possible combinations within the same person
            for img1, img2 in combinations(id_to_aligned_images[test_id], 2):
                pairs.append((img1, img2))
                labels.append(1)  # Same person
    
    print(f"Generated {sum(labels)} positive pairs")
    
    # Generate negative pairs (twin pairs - different persons)
    for id1, id2 in twin_pairs:
        if id1 in id_to_aligned_images and id2 in id_to_aligned_images:
            # Sample pairs between twins
            for img1 in id_to_aligned_images[id1]:
                for img2 in id_to_aligned_images[id2]:
                    pairs.append((img1, img2))
                    labels.append(0)  # Different person (twins)
    
    print(f"Generated {len(labels) - sum(labels)} negative pairs (twins)")
    print(f"Total pairs: {len(pairs)}")
    
    return pairs, labels


def extract_features_batch(model, image_paths, device='cuda', batch_size=32):
    """Extract features for a list of images"""
    print(f"Extracting features for {len(image_paths)} images...")
    
    # Load all images
    images = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    
    if not images:
        return {}, []
    
    # Extract features
    features = get_features_from_aligned_images(images, model, device=device, batch_size=batch_size)
    
    # Normalize features
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    # Create path to feature mapping
    path_to_feature = {path: feat for path, feat in zip(valid_paths, features)}
    
    return path_to_feature, valid_paths


def compute_similarities(pairs, path_to_feature):
    """Compute cosine similarities for pairs"""
    similarities = []
    valid_pairs = []
    valid_labels = []
    
    for i, (img1, img2) in enumerate(pairs):
        if img1 in path_to_feature and img2 in path_to_feature:
            feat1 = path_to_feature[img1]
            feat2 = path_to_feature[img2]
            
            # Cosine similarity
            similarity = torch.dot(feat1, feat2).item()
            similarities.append(similarity)
            valid_pairs.append((img1, img2))
            
    return np.array(similarities)


def find_optimal_threshold(similarities, labels):
    """Find optimal threshold that maximizes accuracy"""
    thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
    best_acc = 0
    best_thresh = 0
    
    for thresh in thresholds:
        predictions = (similarities >= thresh).astype(int)
        acc = accuracy_score(labels, predictions)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return best_thresh, best_acc


def compute_eer(similarities, labels):
    """Compute Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer, eer_threshold


def compute_tar_at_far(similarities, labels, target_fars=[0.01, 0.001, 0.0001]):
    """Compute True Acceptance Rate at specific False Acceptance Rates"""
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    
    results = {}
    for target_far in target_fars:
        # Find threshold closest to target FAR
        idx = np.argmin(np.abs(fpr - target_far))
        tar = tpr[idx]
        actual_far = fpr[idx]
        threshold = thresholds[idx]
        
        results[f"TAR@FAR={target_far}"] = {
            'tar': tar,
            'actual_far': actual_far,
            'threshold': threshold
        }
    
    return results


def evaluate_verification(similarities, labels, save_dir):
    """
    Stage 2: Comprehensive evaluation of verification performance
    """
    print("\n=== Stage 2: Verification Evaluation ===")
    
    # Basic metrics
    optimal_thresh, best_acc = find_optimal_threshold(similarities, labels)
    eer, eer_thresh = compute_eer(similarities, labels)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, similarities)
    pr_auc = auc(recall, precision)
    
    # Predictions at optimal threshold
    predictions = (similarities >= optimal_thresh).astype(int)
    precision_score_val = precision_score(labels, predictions)
    recall_score_val = recall_score(labels, predictions)
    f1_score_val = f1_score(labels, predictions)
    
    # TAR @ FAR
    tar_results = compute_tar_at_far(similarities, labels)
    
    # Print results
    print(f"\n=== VERIFICATION RESULTS ===")
    print(f"Best Accuracy: {best_acc:.4f} (threshold: {optimal_thresh:.4f})")
    print(f"Equal Error Rate (EER): {eer:.4f} (threshold: {eer_thresh:.4f})")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    print(f"Precision: {precision_score_val:.4f}")
    print(f"Recall: {recall_score_val:.4f}")
    print(f"F1-Score: {f1_score_val:.4f}")
    
    print(f"\nTAR @ FAR Results:")
    for key, value in tar_results.items():
        print(f"{key}: TAR={value['tar']:.4f}, Actual FAR={value['actual_far']:.6f}")
    
    # Save results
    results = {
        'best_accuracy': best_acc,
        'optimal_threshold': optimal_thresh,
        'eer': eer,
        'eer_threshold': eer_thresh,
        'auc_roc': roc_auc,
        'auc_pr': pr_auc,
        'precision': precision_score_val,
        'recall': recall_score_val,
        'f1_score': f1_score_val,
        'tar_at_far': tar_results
    }
    
    # Save detailed results
    results_file = os.path.join(save_dir, 'verification_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot and save visualizations
    plot_results(similarities, labels, fpr, tpr, roc_auc, precision, recall, pr_auc, save_dir)
    
    return results


def plot_results(similarities, labels, fpr, tpr, roc_auc, precision, recall, pr_auc, save_dir):
    """Create and save visualization plots"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    # Distance Distributions
    pos_similarities = similarities[labels == 1]
    neg_similarities = similarities[labels == 0]
    
    ax3.hist(pos_similarities, bins=50, alpha=0.7, label='Same Person (Positive)', color='blue', density=True)
    ax3.hist(neg_similarities, bins=50, alpha=0.7, label='Different Person (Negative)', color='red', density=True)
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.set_title('Similarity Score Distributions')
    ax3.legend()
    ax3.grid(True)
    
    # Box plot
    data_for_box = [pos_similarities, neg_similarities]
    ax4.boxplot(data_for_box, labels=['Same Person', 'Twins (Different)'])
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Similarity Score Box Plot')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Separate detailed distribution plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(pos_similarities, bins=50, alpha=0.7, label=f'Same Person (n={len(pos_similarities)})', 
             color='blue', density=True)
    plt.hist(neg_similarities, bins=50, alpha=0.7, label=f'Twins (n={len(neg_similarities)})', 
             color='red', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distributions')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Violin plot
    data_df = []
    for sim in pos_similarities:
        data_df.append(['Same Person', sim])
    for sim in neg_similarities:
        data_df.append(['Twins', sim])
    
    df = pd.DataFrame(data_df, columns=['Type', 'Similarity'])
    sns.violinplot(data=df, x='Type', y='Similarity', ax=plt.gca())
    plt.title('Similarity Score Distributions (Violin Plot)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate AdaFace on identical twin verification')
    parser.add_argument('--twin_pairs', type=str, default='test_twin_id_pairs.json',
                       help='Path to twin pairs JSON file')
    parser.add_argument('--id_to_images', type=str, default='id_to_images.json',
                       help='Path to ID to images mapping JSON file')
    parser.add_argument('--aligned_dir', type=str, default='aligned_test_images',
                       help='Directory to save aligned images')
    parser.add_argument('--results_dir', type=str, default='twin_verification_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--model_arch', type=str, default='ir_50', choices=['ir_50', 'ir_101'],
                       help='Model architecture')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of worker threads for parallel alignment')
    parser.add_argument('--skip_alignment', action='store_true',
                       help='Skip alignment stage (assumes aligned images exist)')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    print("Loading twin pairs and ID mappings...")
    twin_pairs = load_twin_pairs(args.twin_pairs)
    original_id_to_images = load_id_to_images(args.id_to_images)
    
    print(f"Loaded {len(twin_pairs)} twin pairs")
    print(f"Loaded {len(original_id_to_images)} IDs with images")
    
    # Stage 1: Alignment (if not skipped)
    if not args.skip_alignment:
        all_test_images = get_all_test_images_from_twins(twin_pairs, original_id_to_images)
        alignable_images, failed_images = align_test_images(all_test_images, args.aligned_dir, max_workers=args.max_workers)
        
        # Create new ID mapping with only alignable images
        id_to_aligned_images = create_alignable_id_mapping(twin_pairs, original_id_to_images, alignable_images)
        
        # Save the new mapping
        aligned_mapping_file = os.path.join(args.results_dir, 'id_to_aligned_images.json')
        with open(aligned_mapping_file, 'w') as f:
            json.dump(id_to_aligned_images, f, indent=2)
        
        print(f"Saved alignable ID mapping to {aligned_mapping_file}")
    else:
        # Load existing aligned mapping
        aligned_mapping_file = os.path.join(args.results_dir, 'id_to_aligned_images.json')
        if not os.path.exists(aligned_mapping_file):
            print(f"Error: {aligned_mapping_file} not found. Run without --skip_alignment first.")
            return
        
        with open(aligned_mapping_file, 'r') as f:
            id_to_aligned_images = json.load(f)
        print(f"Loaded existing aligned mapping with {len(id_to_aligned_images)} IDs")
    
    # Generate pairs and labels
    pairs, labels = generate_pairs_and_labels(twin_pairs, id_to_aligned_images)
    
    if len(pairs) == 0:
        print("Error: No valid pairs generated for evaluation")
        return
    
    # Load model
    print(f"Loading AdaFace model ({args.model_arch})...")
    model = load_pretrained_model(args.model_arch, device=args.device)
    model = model.to(args.device)
    model.eval()
    
    # Extract features for all unique images
    all_unique_images = set()
    for img1, img2 in pairs:
        all_unique_images.add(img1)
        all_unique_images.add(img2)
    all_unique_images = list(all_unique_images)
    
    path_to_feature, valid_paths = extract_features_batch(model, all_unique_images, 
                                                         device=args.device, batch_size=args.batch_size)
    
    # Compute similarities
    print("Computing similarities...")
    similarities = compute_similarities(pairs, path_to_feature)
    
    # Filter pairs to only those with valid features
    valid_indices = []
    valid_labels = []
    for i, (img1, img2) in enumerate(pairs):
        if img1 in path_to_feature and img2 in path_to_feature:
            valid_indices.append(i)
            valid_labels.append(labels[i])
    
    valid_labels = np.array(valid_labels)
    
    if len(similarities) == 0:
        print("Error: No valid similarities computed")
        return
    
    print(f"Valid pairs for evaluation: {len(similarities)}")
    print(f"Positive pairs: {sum(valid_labels)}")
    print(f"Negative pairs: {len(valid_labels) - sum(valid_labels)}")
    
    # Stage 2: Evaluation
    results = evaluate_verification(similarities, valid_labels, args.results_dir)
    
    print(f"\nEvaluation complete! Results saved to {args.results_dir}")


if __name__ == '__main__':
    main()
