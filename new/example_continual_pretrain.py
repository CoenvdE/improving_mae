#!/usr/bin/env python3
"""
Example script showing how to run continual pretraining with different configurations
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print its description"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return False

def main():
    # Base script path
    script_path = os.path.join(os.path.dirname(__file__), "continual_pretrain.py")
    
    # Common arguments (modify these for your setup)
    common_args = [
        "--data_path", "/path/to/your/dataset",  # ⚠️ CHANGE THIS to your dataset path
        "--output_dir", "./continual_output",
        "--log_dir", "./continual_output",
        "--batch_size", "32",  # Adjust based on your GPU memory
        "--num_workers", "4",
        "--device", "cuda" if "cuda" in sys.argv else "cpu",
    ]
    
    print("Continual MAE Pretraining Examples")
    print("=" * 60)
    print("⚠️  Make sure to update --data_path to your actual dataset path!")
    print("⚠️  Adjust --batch_size based on your GPU memory")
    print("")
    
    examples = [
        {
            "name": "Example 1: Basic continual training (50 epochs)",
            "description": "Train all parameters with dense supervision on all pixels",
            "args": [
                "--epochs", "50",
                "--model", "continual_mae_vit_large_patch16",
                "--loss_mode", "all_pixels",
                "--freeze_mode", "none",
                "--blr", "1e-4",
            ]
        },
        {
            "name": "Example 2: Encoder block LayerNorms-only training (30 epochs)", 
            "description": "Freeze everything except encoder transformer block LayerNorms (norm1, norm2) - best for catastrophic forgetting prevention",
            "args": [
                "--epochs", "30",
                "--model", "continual_mae_vit_large_patch16", 
                "--loss_mode", "all_pixels",
                "--freeze_mode", "encoder_layernorms_only",
                "--blr", "5e-4",  # Higher LR since fewer parameters
            ]
        },
        {
            "name": "Example 2b: All LayerNorms training (25 epochs)", 
            "description": "Freeze everything except all LayerNorm layers (encoder + decoder)",
            "args": [
                "--epochs", "25",
                "--model", "continual_mae_vit_large_patch16", 
                "--loss_mode", "all_pixels",
                "--freeze_mode", "layernorm_only",
                "--blr", "3e-4",  # Moderate LR for more parameters
            ]
        },
        {
            "name": "Example 3: Encoder-frozen training (40 epochs)",
            "description": "Freeze encoder, train only decoder - for domain adaptation",
            "args": [
                "--epochs", "40",
                "--model", "continual_mae_vit_large_patch16",
                "--loss_mode", "all_pixels", 
                "--freeze_mode", "encoder_only",
                "--blr", "2e-4",
            ]
        },
        {
            "name": "Example 4: Original MAE loss (20 epochs)",
            "description": "Use original masked-only loss for comparison",
            "args": [
                "--epochs", "20",
                "--model", "continual_mae_vit_large_patch16",
                "--loss_mode", "masked_only", 
                "--freeze_mode", "none",
                "--blr", "1e-4",
            ]
        },
        {
            "name": "Example 5: Quick test (5 epochs)",
            "description": "Quick test run to verify everything works",
            "args": [
                "--epochs", "5",
                "--model", "continual_mae_vit_base_patch16",  # Smaller model
                "--loss_mode", "all_pixels",
                "--freeze_mode", "none", 
                "--blr", "1e-4",
                "--batch_size", "16",  # Smaller batch size
            ]
        }
    ]
    
    print("Available examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   {example['description']}")
    
    print(f"\n0. Exit")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (0-{len(examples)}): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(examples):
                example = examples[choice_idx]
                
                # Construct command
                cmd = [sys.executable, script_path] + common_args + example["args"]
                
                # Ask for confirmation
                print(f"\nSelected: {example['name']}")
                print(f"Description: {example['description']}")
                confirm = input("Proceed? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    success = run_command(cmd, example['name'])
                    if success:
                        print(f"\n✓ Training completed! Check outputs in: ./continual_output")
                else:
                    print("Cancelled.")
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main() 