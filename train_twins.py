import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from inference import (
    load_pretrained_model,
    align_twin_pairs_images,
    save_aligned_id_mapping,
    to_input_batch,
    get_features_from_aligned_images
)
from triplet_loss import TripletLossWithMetrics
from triplet_dataset import create_twin_datasets


class TwinTrainer:
    """
    Trainer class for fine-tuning AdaFace on twin face verification.
    
    Supports both full model training and frozen backbone training.
    """
    
    def __init__(self, model_architecture='ir_50', device='auto'):
        """
        Initialize the trainer.
        
        Args:
            model_architecture (str): Model architecture to use
            device (str): Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_architecture = model_architecture
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load pretrained model
        print("Loading pretrained AdaFace model...")
        self.model = load_pretrained_model(model_architecture, device=self.device)
        self.model = self.model.to(self.device)
        print("Model loaded successfully!")
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [], 
            'train_metrics': [],
            'val_metrics': [],
            'epochs': []
        }
        
    def stage1_align_images(self, id_to_images_json, train_twin_pairs_json,
                           aligned_folder, aligned_json_output,
                           max_workers=4, overwrite=False):
        """
        Stage 1: Align all training images and create aligned mapping.
        
        Args:
            id_to_images_json (str): Path to original id_to_images.json
            train_twin_pairs_json (str): Path to train_twin_id_pairs.json
            aligned_folder (str): Folder to save aligned images
            aligned_json_output (str): Path to save aligned id_to_images.json
            max_workers (int): Number of parallel workers
            overwrite (bool): Whether to overwrite existing aligned images
            
        Returns:
            dict: Alignment statistics
        """
        print("=" * 60)
        print("STAGE 1: TRAINING IMAGE ALIGNMENT")
        print("=" * 60)
        
        # Perform alignment
        stats = align_twin_pairs_images(
            id_to_images_json=id_to_images_json,
            test_twin_pairs_json=train_twin_pairs_json,  # Use train pairs
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
    
    def _setup_model_for_training(self, freeze_backbone=False, learning_rate=1e-4):
        """
        Setup model for training with optional backbone freezing.
        
        Args:
            freeze_backbone (bool): Whether to freeze backbone layers
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            optimizer: Configured optimizer
        """
        if freeze_backbone:
            print("Setting up frozen backbone training...")
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze the final embedding layer(s)
            # For AdaFace, we need to identify the final layers
            # This depends on the specific architecture
            trainable_layers = []
            
            # Find the final linear/fc layers (typically named 'fc' or similar)
            for name, module in self.model.named_modules():
                if 'fc' in name.lower() or 'linear' in name.lower() or 'features' in name.lower():
                    trainable_layers.append(name)
                    for param in module.parameters():
                        param.requires_grad = True
            
            # If no specific layers found, unfreeze last few layers
            if not trainable_layers:
                print("No specific FC layers found, unfreezing last few layers...")
                all_modules = list(self.model.named_children())
                for name, module in all_modules[-2:]:  # Last 2 modules
                    trainable_layers.append(name)
                    for param in module.parameters():
                        param.requires_grad = True
            
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            print(f"Trainable layers: {trainable_layers}")
            print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
            
        else:
            print("Setting up full model training...")
            # Ensure all parameters are trainable
            for param in self.model.parameters():
                param.requires_grad = True
            
            trainable_params = list(self.model.parameters())
            print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Setup optimizer
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        return optimizer
    
    def _compute_batch_features(self, image_tensors):
        """Compute features for a batch of image tensors."""
        if len(image_tensors) == 0:
            return torch.empty(0, 512).to(self.device)
        
        # Stack tensors into a batch
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # Get features from model
        features, _ = self.model(batch_tensor)
        
        return features
    
    def _validate_model(self, val_dataloader, criterion):
        """Validate the model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                anchor_imgs, positive_imgs, negative_imgs, _, _, _ = batch
                
                # Compute features
                anchor_features = self._compute_batch_features(anchor_imgs)
                positive_features = self._compute_batch_features(positive_imgs)
                negative_features = self._compute_batch_features(negative_imgs)
                
                # Compute loss and metrics
                loss, metrics = criterion(anchor_features, positive_features, negative_features, return_metrics=True)
                
                total_loss += loss.item()
                
                # Aggregate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def stage2_train(self, aligned_json_path, train_twin_pairs_json, output_dir, log_dir,
                    train_percentage=100, freeze_backbone=False, margin=0.3,
                    batch_size=16, epochs=20, learning_rate=None, validation_split=0.15):
        """
        Stage 2: Fine-tune the model on twin triplets.
        
        Args:
            aligned_json_path (str): Path to aligned training images JSON
            train_twin_pairs_json (str): Path to train twin pairs JSON
            output_dir (str): Directory to save trained models
            log_dir (str): Directory to save training logs
            train_percentage (int): Percentage of training data to use
            freeze_backbone (bool): Whether to freeze backbone
            margin (float): Triplet loss margin
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate (auto-selected if None)
            validation_split (float): Validation split ratio
            
        Returns:
            dict: Training results and model path
        """
        print("=" * 60)
        print("STAGE 2: MODEL FINE-TUNING")
        print("=" * 60)
        
        # Auto-select learning rate
        if learning_rate is None:
            learning_rate = 1e-3 if freeze_backbone else 1e-4
            print(f"Auto-selected learning rate: {learning_rate}")
        
        # Create datasets
        print("Creating training and validation datasets...")
        train_dataset, val_dataset = create_twin_datasets(
            aligned_json_path=aligned_json_path,
            train_twin_pairs_json=train_twin_pairs_json,
            train_percentage=train_percentage,
            validation_split=validation_split
        )
        
        if len(train_dataset) == 0:
            raise ValueError("No training data available!")
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True if self.device == 'cuda' else False
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True if self.device == 'cuda' else False
        ) if len(val_dataset) > 0 else None
        
        print(f"Training batches: {len(train_dataloader)}")
        print(f"Validation batches: {len(val_dataloader) if val_dataloader else 0}")
        
        # Setup model and optimizer
        optimizer = self._setup_model_for_training(freeze_backbone, learning_rate)
        criterion = TripletLossWithMetrics(margin=margin, distance_metric='cosine')
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training configuration
        config = {
            'model_architecture': self.model_architecture,
            'train_percentage': train_percentage,
            'freeze_backbone': freeze_backbone,
            'margin': margin,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'validation_split': validation_split,
            'device': self.device,
            'num_train_triplets': len(train_dataset),
            'num_val_triplets': len(val_dataset),
            'start_time': datetime.now().isoformat()
        }
        
        # Save config
        config_path = os.path.join(log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Training configuration saved to: {config_path}")
        
        # Training loop
        print("Starting training...")
        best_val_loss = float('inf')
        best_model_path = None
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            total_train_loss = 0.0
            total_train_metrics = {}
            num_train_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                anchor_imgs, positive_imgs, negative_imgs, _, _, _ = batch
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute features
                anchor_features = self._compute_batch_features(anchor_imgs)
                positive_features = self._compute_batch_features(positive_imgs)
                negative_features = self._compute_batch_features(negative_imgs)
                
                # Compute loss
                loss, metrics = criterion(anchor_features, positive_features, negative_features, return_metrics=True)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                total_train_loss += loss.item()
                
                for key, value in metrics.items():
                    if key not in total_train_metrics:
                        total_train_metrics[key] = 0.0
                    total_train_metrics[key] += value
                
                num_train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'active_triplets': f'{metrics["fraction_active_triplets"]:.3f}'
                })
            
            # Average training metrics
            avg_train_loss = total_train_loss / num_train_batches
            avg_train_metrics = {key: value / num_train_batches for key, value in total_train_metrics.items()}
            
            # Validation phase
            avg_val_loss = 0.0
            avg_val_metrics = {}
            
            if val_dataloader:
                avg_val_loss, avg_val_metrics = self._validate_model(val_dataloader, criterion)
            
            # Record history
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_metrics'].append(avg_train_metrics)
            self.training_history['val_metrics'].append(avg_val_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Train Active Triplets: {avg_train_metrics.get('fraction_active_triplets', 0):.3f}")
            
            if val_dataloader:
                print(f"  Val Loss: {avg_val_loss:.4f}")
                print(f"  Val Active Triplets: {avg_val_metrics.get('fraction_active_triplets', 0):.3f}")
            
            # Save best model
            if val_dataloader and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Generate model filename
                mode = 'frozen' if freeze_backbone else 'full'
                model_name = f"twin_tuned_{self.model_architecture}_{mode}_{train_percentage}pct_margin{margin}_epoch{epoch+1}.pth"
                best_model_path = os.path.join(output_dir, model_name)
                
                # Save model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': config,
                    'training_history': self.training_history
                }, best_model_path)
                
                print(f"  Best model saved: {best_model_path}")
        
        # Save final training log
        log_data = {
            'config': config,
            'training_history': self.training_history,
            'best_model_path': best_model_path,
            'best_val_loss': best_val_loss,
            'end_time': datetime.now().isoformat()
        }
        
        log_path = os.path.join(log_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        # Generate training plots
        self._plot_training_curves(log_dir)
        
        print(f"\nTraining completed!")
        print(f"Best model: {best_model_path}")
        print(f"Training log: {log_path}")
        
        return {
            'best_model_path': best_model_path,
            'best_val_loss': best_val_loss,
            'training_history': self.training_history,
            'config': config
        }
    
    def _plot_training_curves(self, log_dir):
        """Generate and save training curve plots."""
        if not self.training_history['epochs']:
            return
        
        epochs = self.training_history['epochs']
        
        # Loss curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        if self.training_history['val_loss'] and any(self.training_history['val_loss']):
            plt.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Active triplets fraction
        plt.subplot(1, 2, 2)
        train_active = [m.get('fraction_active_triplets', 0) for m in self.training_history['train_metrics']]
        plt.plot(epochs, train_active, 'b-', label='Training Active Triplets')
        
        if self.training_history['val_metrics']:
            val_active = [m.get('fraction_active_triplets', 0) for m in self.training_history['val_metrics']]
            if any(val_active):
                plt.plot(epochs, val_active, 'r-', label='Validation Active Triplets')
        
        plt.xlabel('Epoch')
        plt.ylabel('Fraction of Active Triplets')
        plt.title('Active Triplets During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {os.path.join(log_dir, 'training_curves.png')}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune AdaFace on identical twin verification task')
    
    # Input files
    parser.add_argument('--id_to_images', type=str, default='id_to_images.json',
                       help='Path to original id_to_images.json')
    parser.add_argument('--train_pairs', type=str, default='train_twin_id_pairs.json',
                       help='Path to train_twin_id_pairs.json')
    
    # Alignment settings
    parser.add_argument('--aligned_folder', type=str, default='train_aligned_images',
                       help='Folder to save aligned training images')
    parser.add_argument('--aligned_json', type=str, default='id_to_train_aligned_images.json',
                       help='Path to save aligned id_to_images.json')
    parser.add_argument('--skip_alignment', action='store_true',
                       help='Skip stage 1 (alignment) and use existing aligned images')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Number of parallel workers for alignment')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing aligned images')
    
    # Training settings
    parser.add_argument('--train_percentage', type=int, default=100, choices=[1, 2, 10, 25, 50, 75, 100],
                       help='Percentage of training data to use')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone and only train embedding layer')
    parser.add_argument('--margin', type=float, default=0.3,
                       help='Triplet loss margin')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (auto-selected if not provided)')
    parser.add_argument('--validation_split', type=float, default=0.15,
                       help='Validation split ratio')
    
    # Model settings
    parser.add_argument('--model', type=str, default='ir_50',
                       help='Model architecture (ir_50, ir_101, etc.)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--log_dir', type=str, default='training_logs',
                       help='Directory to save training logs')
    
    args = parser.parse_args()
    
    # Create log directory with descriptive name
    mode = 'frozen' if args.freeze_backbone else 'full'
    log_subdir = f"{mode}_{args.train_percentage}pct_margin{args.margin}"
    full_log_dir = os.path.join(args.log_dir, log_subdir)
    
    # Initialize trainer
    trainer = TwinTrainer(
        model_architecture=args.model,
        device=args.device
    )
    
    try:
        # Stage 1: Alignment (if not skipped)
        if not args.skip_alignment:
            alignment_stats = trainer.stage1_align_images(
                id_to_images_json=args.id_to_images,
                train_twin_pairs_json=args.train_pairs,
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
        
        # Stage 2: Training
        results = trainer.stage2_train(
            aligned_json_path=args.aligned_json,
            train_twin_pairs_json=args.train_pairs,
            output_dir=args.output_dir,
            log_dir=full_log_dir,
            train_percentage=args.train_percentage,
            freeze_backbone=args.freeze_backbone,
            margin=args.margin,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split
        )
        
        print("\nTraining completed successfully!")
        print(f"Best model saved at: {results['best_model_path']}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    main()