#!/usr/bin/env python3
"""
Simple script to load the MAE ViT-Large pretrained checkpoint.
Inspired by the mae_visualize.ipynb demo.
"""

import sys
import os
import torch
import numpy as np

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_mae
import models_vit


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    """
    Load MAE model from checkpoint.
    
    Args:
        chkpt_dir (str): Path to checkpoint file
        arch (str): Model architecture name
    
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Loading model: {arch}")
    print(f"Checkpoint: {chkpt_dir}")
    
    # Build model
    model = getattr(models_mae, arch)()
    
    # Load checkpoint
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    
    return model


def prepare_vit_model(chkpt_dir, arch='vit_large_patch16'):
    """
    Load ViT model from checkpoint for classification/feature extraction.
    
    Args:
        chkpt_dir (str): Path to checkpoint file
        arch (str): Model architecture name
    
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Loading ViT model: {arch}")
    print(f"Checkpoint: {chkpt_dir}")
    
    # Build model
    model = getattr(models_vit, arch)()
    
    # Load checkpoint
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    
    return model


def test_model(model, model_type='mae'):
    """
    Test the loaded model with dummy input.
    
    Args:
        model: Loaded model
        model_type (str): Type of model ('mae' or 'vit')
    """
    print(f"\nTesting {model_type.upper()} model...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        if model_type == 'mae':
            # MAE model returns loss, prediction, mask
            loss, pred, mask = model(dummy_input, mask_ratio=0.75)
            print(f"MAE Output - Loss: {loss.item():.4f}")
            print(f"Prediction shape: {pred.shape}")
            print(f"Mask shape: {mask.shape}")
        else:
            # ViT model returns features/logits
            output = model(dummy_input)
            print(f"ViT Output shape: {output.shape}")
            print(f"First 10 values: {output[0][:10]}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def main():
    """Main function to load and test the models."""
    
    print("MAE ViT-Large Checkpoint Loader")
    print("=" * 40)
    
    # Path to checkpoint
    checkpoint_path = '/Users/coenvandenelsen/Library/CloudStorage/OneDrive-Kampany/Documenten/AI projects/Daan/improving_mae/demo/mae_visualize_vit_large.pth'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    try:
        # Load MAE model
        print("\n1. Loading MAE model for reconstruction...")
        mae_model = prepare_model(checkpoint_path, 'mae_vit_large_patch16')
        test_model(mae_model, 'mae')
        
        print("\n" + "=" * 40)
        
        # # Load ViT model
        # print("\n2. Loading ViT model for classification/features...")
        # vit_model = prepare_vit_model(checkpoint_path, 'vit_large_patch16')
        # test_model(vit_model, 'vit')
        
        # print("\n" + "=" * 40)
        # print("✓ Both models loaded successfully!")
        # print("✓ MAE model ready for reconstruction tasks")
        # print("✓ ViT model ready for classification/feature extraction")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 