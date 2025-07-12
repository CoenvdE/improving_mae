#!/usr/bin/env python3
"""
Simple script to run continual pretraining on Places365
"""

import os
import subprocess
import sys

def run_continual_pretrain(
    model='continual_mae_vit_large_patch16',
    epochs=1,
    batch_size=4,
    freeze_mode='encoder_layernorms_only',
    loss_mode='all_pixels',
    mask_ratio=0,
    lr=1e-4,
    continual_checkpoint='/Users/coenvandenelsen/Library/CloudStorage/OneDrive-Kampany/Documenten/AI projects/Daan/improving_mae/demo/mae_visualize_vit_large.pth',
    # continual_checkpoint='',
    output_dir='./continual_output_places365',
    data_path='/Users/coenvandenelsen/Library/CloudStorage/OneDrive-Kampany/Documenten/AI projects/Daan/improving_mae/data/places365'
):
    """
    Run continual pretraining with specified parameters
    
    Args:
        model: Model architecture (continual_mae_vit_base_patch16, continual_mae_vit_large_patch16, continual_mae_vit_huge_patch14)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        freeze_mode: Freezing strategy (none, layernorm_only, encoder_layernorms_only, encoder_only)
        loss_mode: Loss mode (all_pixels, masked_only)
        mask_ratio: Masking ratio for MAE
        lr: Learning rate
        continual_checkpoint: Path to pretrained checkpoint (empty for from scratch)
        output_dir: Output directory for checkpoints and logs
        data_path: Path to Places365 dataset
    """
    
    # Build command
    cmd = [
        sys.executable, 'continual_pretrain_places365.py',
        '--model', model,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--freeze_mode', freeze_mode,
        '--loss_mode', loss_mode,
        '--mask_ratio', str(mask_ratio),
        '--blr', str(lr),
        '--data_path', data_path,
        '--output_dir', output_dir,
        '--log_dir', output_dir,
        '--device', 'cpu',  # Use CPU device
        '--places_small',  # Use small version by default
        '--places_split', 'val',
        '--warmup_epochs', '10',
        '--weight_decay', '0.05',
        '--num_workers', '4',
        '--pin_mem'
    ]
    
    # Add continual checkpoint if provided
    if continual_checkpoint:
        cmd.extend(['--continual_checkpoint', continual_checkpoint])
    
    print("Running continual pretraining with command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        # Get the directory containing this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir:
            script_dir = os.getcwd()
        
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        return False


def main():
    output_dir = './continual_output_places365'
    print("Continual MAE Pretraining on Places365")
    print("=" * 50)
    try:
        # Run training
        success = run_continual_pretrain()
        
        if success:
            print("\n✓ Training completed successfully!")
            print(f"Check output directory: {output_dir}")
        else:
            print("\n✗ Training failed!")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main() 