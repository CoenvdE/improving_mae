import torchvision.transforms as transforms
from torchvision.datasets import Places365

# Download Places365 dataset
dataset = Places365(root='data/places365', transform=transforms.ToTensor(), small=True, download=True, split='val')

# Print dataset information
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class to index mapping: {dataset.class_to_idx}")