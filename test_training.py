#!/usr/bin/env python3
"""
Quick test script to verify the training pipeline components.
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from triplet_loss import TripletLoss, TripletLossWithMetrics
        print("✓ triplet_loss imported successfully")
        
        from triplet_dataset import TwinTripletDataset, create_twin_datasets
        print("✓ triplet_dataset imported successfully")
        
        from train_twins import TwinTrainer
        print("✓ train_twins imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_triplet_loss():
    """Test triplet loss implementation."""
    print("\nTesting triplet loss...")
    
    try:
        from triplet_loss import test_triplet_loss
        test_triplet_loss()
        return True
    except Exception as e:
        print(f"✗ Triplet loss test failed: {e}")
        return False

def test_dataset():
    """Test dataset implementation."""
    print("\nTesting dataset...")
    
    try:
        from triplet_dataset import test_dataset
        test_dataset()
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TESTING TRAINING PIPELINE COMPONENTS")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test triplet loss
    if not test_triplet_loss():
        all_passed = False
    
    # Test dataset
    if not test_dataset():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Training pipeline is ready.")
        print("\nNext steps:")
        print("1. Prepare your data files:")
        print("   - id_to_images.json")
        print("   - train_twin_id_pairs.json")
        print("2. Run training:")
        print("   python train_twins.py --train_percentage 25 --epochs 10")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 50)

if __name__ == '__main__':
    main()