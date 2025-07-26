import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from itertools import combinations
import argparse
from tqdm import tqdm
import warnings
import pandas as pd  # Add this import at the top
warnings.filterwarnings('ignore')

# Import from local modules
from inference import (
    load_pretrained_model, 
    align_twin_pairs_images, 
    get_features_from_aligned_images,
    save_aligned_id_mapping
)
from PIL import Image


class TwinEvaluator:
    """
    Evaluator for identical twin face verification task.
    
    Handles two-stage evaluation:
    1. Image alignment and preparation
    2. Model evaluation with comprehensive metrics
    """
    
    def __init__(self, model_architecture='ir_50', device='auto', custom_model_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_architecture (str): Model architecture to use
            device (str): Device to run on ('auto', 'cpu', 'cuda')
            custom_model_path (str): Path to custom fine-tuned model checkpoint
        """
        self.model_architecture = model_architecture
        self.custom_model_path = custom_model_path
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        if custom_model_path:
            print(f"Loading fine-tuned model from: {custom_model_path}")
            self.model = self._load_finetuned_model(custom_model_path)
        else:
            print("Loading pretrained AdaFace model...")
            self.model = load_pretrained_model(model_architecture, device=self.device)
        
        self.model = self.model.to(self.device)
        print("Model loaded successfully!")
    
    def _load_finetuned_model(self, model_path):
        """
        Load a fine-tuned model from checkpoint.
        
        Args:
            model_path (str): Path to the model checkpoint
            
        Returns:
            torch.nn.Module: Loaded model
        """
        # Load the base model architecture
        model = load_pretrained_model(self.model_architecture, device='cpu')
        
        # Load the fine-tuned checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load the fine-tuned state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded fine-tuned model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"Model config: {config.get('train_percentage', 'unknown')}% data, "
                      f"{'frozen' if config.get('freeze_backbone', False) else 'full'} training, "
                      f"margin={config.get('margin', 'unknown')}")
        else:
            # Fallback: assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            print("Loaded model state dict (legacy format)")
        
        model.eval()
        return model
    
    def stage1_align_images(self, id_to_images_json, test_twin_pairs_json, 
                           aligned_folder, aligned_json_output, 
                           max_workers=4, overwrite=False):
        """
        Stage 1: Align all test images and create aligned mapping.
        
        Args:
            id_to_images_json (str): Path to original id_to_images.json
            test_twin_pairs_json (str): Path to test_twin_id_pairs.json
            aligned_folder (str): Folder to save aligned images
            aligned_json_output (str): Path to save aligned id_to_images.json
            max_workers (int): Number of parallel workers
            overwrite (bool): Whether to overwrite existing aligned images
            
        Returns:
            dict: Alignment statistics
        """
        print("=" * 60)
        print("STAGE 1: IMAGE ALIGNMENT")
        print("=" * 60)
        
        # Perform alignment
        stats = align_twin_pairs_images(
            id_to_images_json=id_to_images_json,
            test_twin_pairs_json=test_twin_pairs_json,
            dest_folder=aligned_folder,
            max_workers=max_workers,
            overwrite=overwrite
        )
        
        # Save aligned mapping
        if stats["aligned_id_to_images"]:
            save_aligned_id_mapping(stats["aligned_id_to_images"], aligned_json_output)
        else:
            print("Warning: No aligned images to save!")
            
        return stats
    
    def _load_aligned_images(self, aligned_id_to_images, id_name):
        """Load all aligned images for a given ID."""
        images = []
        valid_paths = []
        
        for img_path in aligned_id_to_images.get(id_name, []):
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
        
        return images, valid_paths
    
    def _generate_pairs(self, aligned_id_to_images, test_twin_pairs):
        """
        Generate positive and negative pairs for evaluation.
        
        Positive pairs: All combinations within the same person
        Negative pairs: All combinations between twin pairs
        """
        positive_pairs = []
        negative_pairs = []
        
        print("Generating evaluation pairs...")
        
        # Get all unique IDs from twin pairs
        all_ids = set()
        for pair in test_twin_pairs:
            all_ids.add(pair[0])
            all_ids.add(pair[1])
        
        # Generate positive pairs (intra-class)
        for id_name in all_ids:
            if id_name in aligned_id_to_images:
                id_images = aligned_id_to_images[id_name]
                if len(id_images) >= 2:
                    # All combinations of images from same person
                    for img1, img2 in combinations(id_images, 2):
                        positive_pairs.append((img1, img2, 1, id_name, id_name))
        
        # Generate negative pairs (inter-class, between twins)
        for twin_pair in test_twin_pairs:
            id1, id2 = twin_pair
            
            if id1 in aligned_id_to_images and id2 in aligned_id_to_images:
                id1_images = aligned_id_to_images[id1]
                id2_images = aligned_id_to_images[id2]
                
                # All combinations between twin pairs
                for img1 in id1_images:
                    for img2 in id2_images:
                        negative_pairs.append((img1, img2, 0, id1, id2))
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        print(f"Generated {len(negative_pairs)} negative pairs")
        
        return positive_pairs, negative_pairs
    
    def _compute_similarities(self, pairs, batch_size=32):
        """Compute cosine similarities for pairs."""
        similarities = []
        labels = []
        pair_info = []  # Add this to store pair information
        
        print("Computing similarities...")
        
        for i in tqdm(range(0, len(pairs), batch_size)):
            batch_pairs = pairs[i:i + batch_size]
            
            # Load images for current batch
            batch_images1 = []
            batch_images2 = []
            batch_labels = []
            batch_pair_info = []  # Store info for current batch
            
            for img_path1, img_path2, label, id1, id2 in batch_pairs:
                try:
                    img1 = Image.open(img_path1).convert('RGB')
                    img2 = Image.open(img_path2).convert('RGB')
                    
                    batch_images1.append(img1)
                    batch_images2.append(img2)
                    batch_labels.append(label)
                    # Store pair information
                    batch_pair_info.append({
                        'image1_path': img_path1,
                        'image2_path': img_path2,
                        'image1_name': os.path.basename(img_path1),
                        'image2_name': os.path.basename(img_path2),
                        'id1': id1,
                        'id2': id2,
                        'true_label': label,
                        'pair_type': 'positive' if label == 1 else 'negative'
                    })
                except Exception as e:
                    print(f"Warning: Failed to load pair ({img_path1}, {img_path2}): {e}")
                    continue
        
            if not batch_images1:
                continue
                
            # Get features
            features1 = get_features_from_aligned_images(
                batch_images1, self.model, device=self.device, batch_size=len(batch_images1)
            )
            features2 = get_features_from_aligned_images(
                batch_images2, self.model, device=self.device, batch_size=len(batch_images2)
            )
            
            # Compute cosine similarities
            features1_norm = torch.nn.functional.normalize(features1, p=2, dim=1)
            features2_norm = torch.nn.functional.normalize(features2, p=2, dim=1)
            batch_similarities = torch.sum(features1_norm * features2_norm, dim=1)
            
            similarities.extend(batch_similarities.cpu().numpy())
            labels.extend(batch_labels)
            pair_info.extend(batch_pair_info)  # Add pair info
    
        return np.array(similarities), np.array(labels), pair_info  # Return pair info too
    
    def _compute_metrics(self, similarities, labels):
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        metrics['auc'] = roc_auc
        
        # Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = 1. - interp1d(fpr, tpr)(eer_threshold)
        metrics['eer'] = eer
        metrics['eer_threshold'] = eer_threshold
        
        # Best accuracy
        accuracies = []
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            acc = accuracy_score(labels, predictions)
            accuracies.append(acc)
        
        best_acc_idx = np.argmax(accuracies)
        metrics['best_accuracy'] = accuracies[best_acc_idx]
        metrics['best_acc_threshold'] = thresholds[best_acc_idx]
        
        # Predictions at best accuracy threshold
        best_predictions = (similarities >= metrics['best_acc_threshold']).astype(int)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, best_predictions, average='binary'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # TAR at specific FARs
        tar_at_far = {}
        for far_target in [0.01, 0.001, 0.0001]:
            # Find closest FAR to target
            far_idx = np.argmin(np.abs(fpr - far_target))
            tar_at_far[f'tar_at_far_{far_target}'] = tpr[far_idx]
        metrics.update(tar_at_far)
        
        # Confusion matrix
        cm = confusion_matrix(labels, best_predictions)
        metrics['confusion_matrix'] = cm
        
        return metrics, fpr, tpr, thresholds
    
    def _plot_results(self, similarities, labels, fpr, tpr, metrics, output_dir):
        """Generate visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Twin Face Verification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distance distributions
        plt.figure(figsize=(10, 6))
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]
        
        plt.hist(negative_similarities, bins=50, alpha=0.7, label='Negative pairs (twins)', 
                color='red', density=True)
        plt.hist(positive_similarities, bins=50, alpha=0.7, label='Positive pairs (same person)', 
                color='blue', density=True)
        
        plt.axvline(metrics['best_acc_threshold'], color='green', linestyle='--', 
                   label=f'Best threshold = {metrics["best_acc_threshold"]:.4f}')
        plt.axvline(metrics['eer_threshold'], color='orange', linestyle='--', 
                   label=f'EER threshold = {metrics["eer_threshold"]:.4f}')
        
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Similarity Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Different Person', 'Same Person'],
                    yticklabels=['Different Person', 'Same Person'])
        plt.title('Confusion Matrix (at Best Accuracy Threshold)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {output_dir}")
    
    def _save_pair_results(self, similarities, labels, pair_info, metrics, output_dir):
        """Save detailed pair results to CSV."""
        # Add similarity scores and predictions to pair info
        best_threshold = metrics['best_acc_threshold']
        eer_threshold = metrics['eer_threshold']
        
        for i, info in enumerate(pair_info):
            info['similarity_score'] = float(similarities[i])
            info['predicted_label_best'] = int(similarities[i] >= best_threshold)
            info['predicted_label_eer'] = int(similarities[i] >= eer_threshold)
            info['correct_best'] = info['predicted_label_best'] == info['true_label']
            info['correct_eer'] = info['predicted_label_eer'] == info['true_label']
        
        # Create DataFrame and save
        df = pd.DataFrame(pair_info)
        
        # Sort by similarity score for easier analysis
        df = df.sort_values('similarity_score', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'pair_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Save separate CSVs for positive and negative pairs
        positive_df = df[df['pair_type'] == 'positive']
        negative_df = df[df['pair_type'] == 'negative']
        
        positive_df.to_csv(os.path.join(output_dir, 'positive_pairs.csv'), index=False)
        negative_df.to_csv(os.path.join(output_dir, 'negative_pairs.csv'), index=False)
        
        # Print summary statistics
        print(f"\nPair Results Summary:")
        print(f"Total pairs: {len(df)}")
        print(f"Positive pairs: {len(positive_df)}")
        print(f"Negative pairs: {len(negative_df)}")
        print(f"Correct predictions (best threshold): {df['correct_best'].sum()}/{len(df)}")
        print(f"Correct predictions (EER threshold): {df['correct_eer'].sum()}/{len(df)}")
        print(f"\nDetailed results saved to:")
        print(f"  - All pairs: {csv_path}")
        print(f"  - Positive pairs: {os.path.join(output_dir, 'positive_pairs.csv')}")
        print(f"  - Negative pairs: {os.path.join(output_dir, 'negative_pairs.csv')}")
        
        return df
    
    def stage2_evaluate(self, aligned_json_path, test_twin_pairs_json, 
                       output_dir, batch_size=32):
        """
        Stage 2: Evaluate on aligned images.
        
        Args:
            aligned_json_path (str): Path to aligned id_to_images.json
            test_twin_pairs_json (str): Path to test_twin_id_pairs.json
            output_dir (str): Directory to save results
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Evaluation metrics
        """
        print("=" * 60)
        print("STAGE 2: MODEL EVALUATION")
        print("=" * 60)
        
        # Load data
        with open(aligned_json_path, 'r') as f:
            aligned_id_to_images = json.load(f)
        
        with open(test_twin_pairs_json, 'r') as f:
            test_twin_pairs = json.load(f)
        
        print(f"Loaded {len(aligned_id_to_images)} IDs with aligned images")
        print(f"Loaded {len(test_twin_pairs)} twin pairs")
        
        # Generate pairs
        positive_pairs, negative_pairs = self._generate_pairs(
            aligned_id_to_images, test_twin_pairs
        )
        
        if not positive_pairs and not negative_pairs:
            raise ValueError("No valid pairs generated! Check your data.")
        
        all_pairs = positive_pairs + negative_pairs
        
        # Compute similarities (now returns pair_info too)
        similarities, labels, pair_info = self._compute_similarities(all_pairs, batch_size)
        
        if len(similarities) == 0:
            raise ValueError("No similarities computed! Check your images.")
        
        print(f"\nProcessed {len(similarities)} pairs")
        print(f"Positive pairs: {np.sum(labels == 1)}")
        print(f"Negative pairs: {np.sum(labels == 0)}")
        
        # Compute metrics
        metrics, fpr, tpr, thresholds = self._compute_metrics(similarities, labels)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed pair results to CSV
        pair_df = self._save_pair_results(similarities, labels, pair_info, metrics, output_dir)
        
        # Save results
        results = {
            'metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'num_positive_pairs': int(np.sum(labels == 1)),
            'num_negative_pairs': int(np.sum(labels == 0)),
            'model_architecture': self.model_architecture,
            'device': self.device
        }
        
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate plots
        self._plot_results(similarities, labels, fpr, tpr, metrics, output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Best Accuracy: {metrics['best_accuracy']:.4f}")
        print(f"AUC-ROC: {metrics['auc']:.4f}")
        print(f"Equal Error Rate (EER): {metrics['eer']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"TAR @ FAR=0.1%: {metrics['tar_at_far_0.001']:.4f}")
        print(f"TAR @ FAR=0.01%: {metrics['tar_at_far_0.0001']:.4f}")
        print(f"\nResults saved to: {results_path}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate AdaFace on identical twin verification task')
    
    # Input files
    parser.add_argument('--id_to_images', type=str, default='id_to_images.json',
                       help='Path to original id_to_images.json')
    parser.add_argument('--test_pairs', type=str, default='test_twin_id_pairs.json',
                       help='Path to test_twin_id_pairs.json')
    
    # Alignment settings
    parser.add_argument('--aligned_folder', type=str, default='aligned_test_images',
                       help='Folder to save aligned images')
    parser.add_argument('--aligned_json', type=str, default='id_to_aligned_images.json',
                       help='Path to save aligned id_to_images.json')
    parser.add_argument('--skip_alignment', action='store_true',
                       help='Skip stage 1 (alignment) and use existing aligned images')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of parallel workers for alignment')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing aligned images')
    
    # Model settings
    parser.add_argument('--model', type=str, default='ir_50',
                       help='Model architecture (ir_50, ir_101, etc.)')
    parser.add_argument('--custom_model_path', type=str, default=None,
                       help='Path to fine-tuned model checkpoint (optional)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TwinEvaluator(
        model_architecture=args.model,
        device=args.device,
        custom_model_path=args.custom_model_path
    )
    
    try:
        # Stage 1: Alignment (if not skipped)
        if not args.skip_alignment:
            alignment_stats = evaluator.stage1_align_images(
                id_to_images_json=args.id_to_images,
                test_twin_pairs_json=args.test_pairs,
                aligned_folder=args.aligned_folder,
                aligned_json_output=args.aligned_json,
                max_workers=args.max_workers,
                overwrite=args.overwrite
            )
            
            if alignment_stats["successful"] == 0:
                print("Warning: No images were successfully aligned!")
                return
        else:
            print("Skipping alignment stage...")
            if not os.path.exists(args.aligned_json):
                raise FileNotFoundError(f"Aligned JSON file not found: {args.aligned_json}")
        
        # Stage 2: Evaluation
        metrics = evaluator.stage2_evaluate(
            aligned_json_path=args.aligned_json,
            test_twin_pairs_json=args.test_pairs,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == '__main__':
    main()