import os
import json
import random
from itertools import combinations
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TwinTripletDataset(Dataset):
    """
    Dataset for generating triplets from twin face data.
    
    Handles smart triplet sampling:
    - Skip twin pairs where both have only 1 image
    - Generate balanced triplets across twin pairs
    - Support percentage-based subsampling
    """
    
    def __init__(self, aligned_id_to_images, train_twin_pairs, 
                 train_percentage=100, validation_split=0.15, 
                 is_validation=False, seed=42):
        """
        Initialize the twin triplet dataset.
        
        Args:
            aligned_id_to_images (dict): Mapping from ID to list of aligned image paths
            train_twin_pairs (list): List of twin pairs [[id1, id2], ...]
            train_percentage (int): Percentage of maximum triplets to use (10-100)
            validation_split (float): Fraction of data to use for validation
            is_validation (bool): Whether this is validation dataset
            seed (int): Random seed for reproducible sampling
        """
        self.aligned_id_to_images = aligned_id_to_images
        self.train_twin_pairs = train_twin_pairs
        self.train_percentage = train_percentage
        self.validation_split = validation_split
        self.is_validation = is_validation
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate triplets
        self.triplets = self._generate_triplets()
        
        # Split into train/validation
        self._create_train_val_split()
        
        print(f"{'Validation' if is_validation else 'Training'} dataset initialized:")
        print(f"  Total triplets: {len(self.triplets)}")
        print(f"  Train percentage: {train_percentage}%")
        
    def _generate_triplets(self):
        """Generate all possible triplets based on twin pairs."""
        all_triplets = []
        skipped_pairs = []
        
        print("Generating triplets from twin pairs...")
        
        for twin_pair in self.train_twin_pairs:
            id1, id2 = twin_pair
            
            # Check if both IDs exist in aligned data
            if id1 not in self.aligned_id_to_images or id2 not in self.aligned_id_to_images:
                skipped_pairs.append(twin_pair)
                continue
                
            id1_images = self.aligned_id_to_images[id1]
            id2_images = self.aligned_id_to_images[id2]
            
            # Skip if both twins have only 1 image
            if len(id1_images) <= 1 and len(id2_images) <= 1:
                skipped_pairs.append(twin_pair)
                continue
            
            # Generate triplets based on available images
            pair_triplets = []
            
            # Case 1: id1 has multiple images, use id1 as anchor/positive, id2 as negative
            if len(id1_images) >= 2:
                for anchor_img, positive_img in combinations(id1_images, 2):
                    for negative_img in id2_images:
                        pair_triplets.append((anchor_img, positive_img, negative_img, id1, id2))
            
            # Case 2: id2 has multiple images, use id2 as anchor/positive, id1 as negative
            if len(id2_images) >= 2:
                for anchor_img, positive_img in combinations(id2_images, 2):
                    for negative_img in id1_images:
                        pair_triplets.append((anchor_img, positive_img, negative_img, id2, id1))
            
            all_triplets.extend(pair_triplets)
        
        if skipped_pairs:
            print(f"Skipped {len(skipped_pairs)} twin pairs (insufficient images):")
            for pair in skipped_pairs[:5]:  # Show first 5
                id1, id2 = pair
                id1_count = len(self.aligned_id_to_images.get(id1, []))
                id2_count = len(self.aligned_id_to_images.get(id2, []))
                print(f"  {id1} ({id1_count} imgs) - {id2} ({id2_count} imgs)")
            if len(skipped_pairs) > 5:
                print(f"  ... and {len(skipped_pairs) - 5} more")
        
        print(f"Generated {len(all_triplets)} total triplets from {len(self.train_twin_pairs) - len(skipped_pairs)} twin pairs")
        
        # Shuffle triplets for random sampling
        random.shuffle(all_triplets)
        
        return all_triplets
    
    def _create_train_val_split(self):
        """Split triplets into train and validation sets."""
        total_triplets = len(self.triplets)
        val_size = int(total_triplets * self.validation_split)
        
        if self.is_validation:
            # Use last val_size triplets for validation
            self.triplets = self.triplets[-val_size:]
        else:
            # Use first (total - val_size) triplets for training
            self.triplets = self.triplets[:-val_size]
            
            # Apply percentage sampling for training
            if self.train_percentage < 100:
                sample_size = int(len(self.triplets) * self.train_percentage / 100)
                self.triplets = self.triplets[:sample_size]
                print(f"Sampled {sample_size} triplets ({self.train_percentage}% of training data)")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Get a triplet of images.
        
        Returns:
            tuple: (anchor_img, positive_img, negative_img, anchor_id, positive_id, negative_id)
        """
        anchor_path, positive_path, negative_path, anchor_id, negative_id = self.triplets[idx]
        
        try:
            # Load images
            anchor_img = Image.open(anchor_path).convert('RGB')
            positive_img = Image.open(positive_path).convert('RGB') 
            negative_img = Image.open(negative_path).convert('RGB')
            
            return (anchor_img, positive_img, negative_img, anchor_id, anchor_id, negative_id)
            
        except Exception as e:
            print(f"Error loading triplet {idx}: {e}")
            # Return a random valid triplet instead
            return self.__getitem__(random.randint(0, len(self.triplets) - 1))


def create_twin_datasets(aligned_json_path, train_twin_pairs_json, 
                        train_percentage=100, validation_split=0.15, seed=42):
    """
    Create training and validation datasets from twin data.
    
    Args:
        aligned_json_path (str): Path to aligned id_to_images.json
        train_twin_pairs_json (str): Path to train_twin_id_pairs.json
        train_percentage (int): Percentage of data to use for training
        validation_split (float): Fraction for validation
        seed (int): Random seed
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Load data
    with open(aligned_json_path, 'r') as f:
        aligned_id_to_images = json.load(f)
    
    with open(train_twin_pairs_json, 'r') as f:
        train_twin_pairs = json.load(f)
    
    print(f"Loaded {len(aligned_id_to_images)} IDs with aligned images")
    print(f"Loaded {len(train_twin_pairs)} twin pairs")
    
    # Create datasets
    train_dataset = TwinTripletDataset(
        aligned_id_to_images=aligned_id_to_images,
        train_twin_pairs=train_twin_pairs,
        train_percentage=train_percentage,
        validation_split=validation_split,
        is_validation=False,
        seed=seed
    )
    
    val_dataset = TwinTripletDataset(
        aligned_id_to_images=aligned_id_to_images,
        train_twin_pairs=train_twin_pairs,
        train_percentage=100,  # Use all available data for validation
        validation_split=validation_split,
        is_validation=True,
        seed=seed
    )
    
    return train_dataset, val_dataset


def test_dataset():
    """Test function for the dataset implementation."""
    print("Testing TwinTripletDataset...")
    
    # Create dummy data
    aligned_id_to_images = {
        'person1': ['/path/img1_1.png', '/path/img1_2.png', '/path/img1_3.png'],
        'person2': ['/path/img2_1.png', '/path/img2_2.png'],
        'person3': ['/path/img3_1.png'],
        'person4': ['/path/img4_1.png', '/path/img4_2.png'],
    }
    
    train_twin_pairs = [
        ['person1', 'person2'],  # Both have multiple images
        ['person3', 'person4'],  # One has 1 image, other has multiple
    ]
    
    # Test dataset creation
    dataset = TwinTripletDataset(
        aligned_id_to_images=aligned_id_to_images,
        train_twin_pairs=train_twin_pairs,
        train_percentage=50,
        validation_split=0.2,
        is_validation=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    print("Dataset test completed!")


if __name__ == '__main__':
    test_dataset()