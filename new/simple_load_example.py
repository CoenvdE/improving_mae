#!/usr/bin/env python3
"""
Simple example of loading the MAE ViT-Large checkpoint for different use cases.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_vit
import models_mae
from util.pos_embed import interpolate_pos_embed


def load_for_feature_extraction():
    """Load model for feature extraction (no classification head)."""
    print("Loading model for feature extraction...")
    
    # Load checkpoint
    checkpoint = torch.load("../checkpoints/mae_pretrain_vit_large.pth", map_location='cpu')
    
    # Create model
    model = models_vit.vit_large_patch16()
    
    # Load weights
    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    
    model.eval()
    return model


def load_for_finetuning():
    """Load model for fine-tuning on a classification task."""
    print("Loading model for fine-tuning...")
    
    # Load checkpoint
    checkpoint = torch.load("../checkpoints/mae_pretrain_vit_large.pth", map_location='cpu')
    
    # Create model with custom number of classes
    num_classes = 1000  # Change this for your dataset
    model = models_vit.vit_large_patch16(num_classes=num_classes)
    
    # Load weights (excluding head)
    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(model, checkpoint_model)
    
    # Remove head weights if they exist
    if 'head.weight' in checkpoint_model:
        del checkpoint_model['head.weight']
    if 'head.bias' in checkpoint_model:
        del checkpoint_model['head.bias']
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys (expected for new head): {msg.missing_keys}")
    
    model.train()  # Set to training mode for fine-tuning
    return model


def load_mae_for_reconstruction():
    """Load MAE model for image reconstruction/visualization."""
    print("Loading MAE model for reconstruction...")
    
    # Load checkpoint
    checkpoint = torch.load("../checkpoints/mae_pretrain_vit_large.pth", map_location='cpu')
    
    # Create MAE model
    mae_model = models_mae.mae_vit_large_patch16()
    
    # Load weights
    checkpoint_model = checkpoint['model']
    mae_model.load_state_dict(checkpoint_model, strict=False)
    
    mae_model.eval()
    return mae_model


def example_usage():
    """Example usage of different loading methods."""
    
    print("MAE ViT-Large Checkpoint Loading Examples")
    print("="*50)
    
    # Example 1: Feature extraction
    feature_model = load_for_feature_extraction()
    print("✓ Feature extraction model loaded")
    
    # Example 2: Fine-tuning
    finetune_model = load_for_finetuning()
    print("✓ Fine-tuning model loaded")
    
    # Example 3: MAE reconstruction
    mae_model = load_mae_for_reconstruction()
    print("✓ MAE reconstruction model loaded")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("\nTesting models:")
    
    # Feature extraction
    with torch.no_grad():
        features = feature_model(dummy_input)
        print(f"Feature extraction output shape: {features.shape}")
    
    # Fine-tuning model
    with torch.no_grad():
        logits = finetune_model(dummy_input)
        print(f"Fine-tuning model output shape: {logits.shape}")
    
    # MAE reconstruction
    with torch.no_grad():
        loss, pred, mask = mae_model(dummy_input, mask_ratio=0.75)
        print(f"MAE reconstruction - Loss: {loss.item():.4f}, Pred shape: {pred.shape}, Mask shape: {mask.shape}")


if __name__ == "__main__":
    example_usage() 