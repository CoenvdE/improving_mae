# MAE ViT-Large Checkpoint Loading Scripts

This folder contains scripts to load and use the MAE ViT-Large pretrained checkpoint (`mae_pretrain_vit_large.pth`).

## Files

### 1. `load_checkpoint.py`
**Comprehensive checkpoint loader with detailed demonstration.**

Features:
- Loads the MAE ViT-Large pretrained checkpoint
- Handles position embedding interpolation
- Demonstrates model usage with dummy input
- Shows model statistics and information
- Creates both ViT and MAE models
- Includes error handling and validation

**Usage:**
```bash
cd new
python load_checkpoint.py
```

### 2. `simple_load_example.py`
**Simple examples for different use cases.**

Features:
- Load model for feature extraction
- Load model for fine-tuning (with custom classes)
- Load MAE model for reconstruction
- Test all models with dummy input

**Usage:**
```bash
cd new
python simple_load_example.py
```

## Use Cases

### Feature Extraction
Load the model without classification head for extracting features:
```python
model = load_for_feature_extraction()
features = model(image_tensor)  # Shape: [batch_size, 1024]
```

### Fine-tuning
Load the model and prepare it for fine-tuning on your dataset:
```python
model = load_for_finetuning()  # Creates new classification head
# Train model on your dataset
```

### Image Reconstruction
Load the full MAE model for image reconstruction and visualization:
```python
mae_model = load_mae_for_reconstruction()
loss, pred, mask = mae_model(image_tensor, mask_ratio=0.75)
```

## Requirements

- PyTorch
- The main improving_mae codebase (models_vit.py, models_mae.py, util/)
- The checkpoint file at `../checkpoints/mae_pretrain_vit_large.pth`

## Model Information

- **Architecture**: Vision Transformer Large (ViT-L/16)
- **Parameters**: ~304M parameters
- **Input Size**: 224x224 RGB images
- **Patch Size**: 16x16
- **Embedding Dimension**: 1024
- **Layers**: 24
- **Attention Heads**: 16

## Output Shapes

- **Feature Extraction**: `[batch_size, 1024]`
- **Classification**: `[batch_size, num_classes]`
- **MAE Reconstruction**: 
  - Loss: scalar
  - Prediction: `[batch_size, num_patches, patch_size*patch_size*3]`
  - Mask: `[batch_size, num_patches]`

## Notes

- The scripts automatically handle position embedding interpolation
- Models are loaded in evaluation mode by default
- For fine-tuning, switch to training mode with `model.train()`
- The checkpoint contains the full model state from pretraining
- Missing keys for classification head are expected when loading for fine-tuning 