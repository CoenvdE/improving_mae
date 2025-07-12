#!/usr/bin/env python3
"""
Load a sample image from Places365 dataset and display its label
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_places_sample(root_dir="/Users/coenvandenelsen/Library/CloudStorage/OneDrive-Kampany/Documenten/AI projects/Daan/improving_mae/data/places365", sample_index=0, small=True):
    """
    Load a sample image from Places365 dataset
    
    Args:
        root_dir: Directory where the dataset is stored
        sample_index: Index of the sample to load
        small: If True, use the small version (256x256)
    """
    print(f"Loading Places365 sample from {root_dir}")
    
    try:
        # Load the dataset (without downloading if already exists)
        dataset = Places365(
            root=root_dir, 
            transform=transforms.ToTensor(), 
            small=small,
            download=False,  # Don't download, assume it exists
            split='val'
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {len(dataset.classes)}")
        
        # Load a specific sample
        if sample_index >= len(dataset):
            sample_index = 0
            print(f"⚠ Index too large, using index 0")
        
        image, label = dataset[sample_index]
        class_name = dataset.classes[label]
        
        print(f"\nSample {sample_index}:")
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"Class name: {class_name}")
        
        return image, label, class_name, dataset
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None, None, None, None

def display_sample(image, label, class_name, save_path=None):
    """
    Display the sample image with its label
    
    Args:
        image: Tensor image
        label: Label index
        class_name: Class name string
        save_path: Optional path to save the image
    """
    # Convert tensor to PIL Image for display
    if isinstance(image, torch.Tensor):
        # Convert from CHW to HWC and scale to [0, 255]
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image_np)
    plt.title(f"Label: {label} | Class: {class_name}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Image saved to {save_path}")
    
    plt.show()

def load_multiple_samples(root_dir="/Users/coenvandenelsen/Library/CloudStorage/OneDrive-Kampany/Documenten/AI projects/Daan/improving_mae/data/places365", num_samples=5, small=True):
    """
    Load multiple samples from the dataset
    
    Args:
        root_dir: Directory where the dataset is stored
        num_samples: Number of samples to load
        small: If True, use the small version
    """
    print(f"Loading {num_samples} samples from Places365 dataset")
    
    try:
        # Load the dataset
        dataset = Places365(
            root=root_dir, 
            transform=transforms.ToTensor(), 
            small=small,
            download=False,
            split='val'
        )
        
        print(f"✓ Dataset loaded successfully!")
        
        # Load multiple samples
        samples = []
        for i in range(min(num_samples, len(dataset))):
            image, label = dataset[i]
            class_name = dataset.classes[label]
            samples.append((image, label, class_name))
            print(f"Sample {i}: {class_name} (label: {label})")
        
        return samples, dataset
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None, None

def main():
    print("Places365 Sample Loader")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "/Users/coenvandenelsen/Library/CloudStorage/OneDrive-Kampany/Documenten/AI projects/Daan/improving_mae/data/places365"
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at {dataset_path}")
        print("Please run download_places.py first to download the dataset")
        return
    
    # Load a single sample
    print("\n1. Loading single sample:")
    image, label, class_name, dataset = load_places_sample(sample_index=0)
    
    if image is not None:
        print(f"✓ Loaded sample successfully!")
        
        # Display basic info
        print(f"\nImage details:")
        print(f"  - Shape: {image.shape}")
        print(f"  - Data type: {image.dtype}")
        print(f"  - Min value: {image.min():.4f}")
        print(f"  - Max value: {image.max():.4f}")
        print(f"  - Mean: {image.mean():.4f}")
        
        # Display the image
        display_sample(image, label, class_name)
        
        # Load multiple samples
        print("\n2. Loading multiple samples:")
        samples, _ = load_multiple_samples(num_samples=5)
        
        if samples:
            print(f"✓ Loaded {len(samples)} samples!")
            print("\nSample classes:")
            for i, (img, lbl, cls) in enumerate(samples):
                print(f"  {i}: {cls}")
            
            # Display all samples
            print("\n3. Displaying all samples:")
            for i, (image, label, class_name) in enumerate(samples):
                print(f"\nDisplaying sample {i}: {class_name}")
                display_sample(image, label, class_name)
    else:
        print("✗ Failed to load sample")

if __name__ == "__main__":
    main() 