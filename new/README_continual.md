# Continual MAE Pretraining

This directory contains scripts for continual learning with Masked Autoencoders (MAE).

## 🎯 Key Features

- **Dense Supervision**: Loss computed on ALL pixels, not just masked patches
- **Selective Freezing**: Multiple strategies to prevent catastrophic forgetting
- **Flexible Training**: Support for different loss modes and model sizes
- **Easy Usage**: Example scripts with common configurations

## 📁 Files

- `continual.py` - ContinualMAE class implementation
- `continual_pretrain.py` - Main training script for continual learning
- `example_continual_pretrain.py` - Interactive example runner
- `load_checkpoint.py` - Utilities for loading checkpoints
- `simple_load_example.py` - Simple model loading example

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Run the example script for guided training
python example_continual_pretrain.py

# Or run directly with custom arguments
python continual_pretrain.py \
    --epochs 50 \
    --data_path /path/to/imagenet \
    --model continual_mae_vit_large_patch16 \
    --loss_mode all_pixels \
    --freeze_mode none \
    --batch_size 32
```

### 2. Available Models

- `continual_mae_vit_base_patch16` - ViT-Base (768 dim, 12 layers)
- `continual_mae_vit_large_patch16` - ViT-Large (1024 dim, 24 layers) 
- `continual_mae_vit_huge_patch14` - ViT-Huge (1280 dim, 32 layers)

### 3. Loss Modes

- `all_pixels` - **Recommended**: Dense supervision on all patches
- `masked_only` - Original MAE loss on masked patches only

### 4. Freezing Strategies

- `none` - Train all parameters (default)
- `encoder_layernorms_only` - **Recommended for continual learning**: Freeze everything except encoder transformer block LayerNorms
- `layernorm_only` - Freeze everything except all LayerNorms (encoder + decoder)
- `encoder_only` - Freeze encoder, train decoder only

## 🔧 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--model` | `continual_mae_vit_large_patch16` | Model architecture |
| `--loss_mode` | `all_pixels` | Loss computation mode |
| `--freeze_mode` | `none` | Parameter freezing strategy (none/encoder_layernorms_only/layernorm_only/encoder_only) |
| `--blr` | `1e-4` | Base learning rate |
| `--batch_size` | 64 | Batch size per GPU |
| `--mask_ratio` | 0.75 | Masking ratio for patches |

## 📊 Continual Learning Configurations

### Configuration 1: Encoder Block LayerNorms Only (Recommended)
Best for preventing catastrophic forgetting while allowing minimal adaptation through transformer block normalization:

```bash
python continual_pretrain.py \
    --epochs 30 \
    --loss_mode all_pixels \
    --freeze_mode encoder_layernorms_only \
    --blr 5e-4 \
    --data_path /path/to/new_domain
```

### Configuration 1b: All LayerNorms
More adaptation capability than encoder-only LayerNorms:

```bash
python continual_pretrain.py \
    --epochs 25 \
    --loss_mode all_pixels \
    --freeze_mode layernorm_only \
    --blr 3e-4 \
    --data_path /path/to/new_domain
```

### Configuration 2: Domain Adaptation
Freeze encoder, adapt decoder to new domain:

```bash
python continual_pretrain.py \
    --epochs 40 \
    --loss_mode all_pixels \
    --freeze_mode encoder_only \
    --blr 2e-4 \
    --continual_checkpoint /path/to/pretrained.pth
```

### Configuration 3: Full Fine-tuning
Train all parameters with dense supervision:

```bash
python continual_pretrain.py \
    --epochs 50 \
    --loss_mode all_pixels \
    --freeze_mode none \
    --blr 1e-4
```

## 🎮 Interactive Example Runner

The `example_continual_pretrain.py` script provides an interactive menu:

```bash
python example_continual_pretrain.py

# Available examples:
# 1. Basic continual training (50 epochs)
# 2. Encoder block LayerNorms-only training (30 epochs) 
# 2b. All LayerNorms training (25 epochs)
# 3. Encoder-frozen training (40 epochs)
# 4. Original MAE loss (20 epochs)
# 5. Quick test (5 epochs)
```

## 🧠 Understanding the ContinualMAE Class

The `ContinualMAE` class extends the original MAE with several key improvements:

```python
from continual import continual_mae_vit_large_patch16

# Create model
model = continual_mae_vit_large_patch16()

# Apply freezing strategy
model.freeze_everything_except_layernorm()

# Forward pass with dense supervision
loss, pred, mask = model(images, mask_ratio=0.75, loss_mode='all_pixels')

# Check parameter status
model.print_param_status()
```

### Key Methods

- `freeze_everything_except_layernorm_encoder()` - Freeze all except encoder transformer block LayerNorms (recommended)
- `freeze_everything_except_all_layernorms()` - Freeze all except all LayerNorm layers  
- `unfreeze_all()` - Unfreeze all parameters
- `forward_loss_all_pixels()` - Compute loss on all pixels
- `print_param_status()` - Display trainable/frozen parameter counts

## 📈 Expected Behavior

### Dense Supervision Benefits
- Better gradient flow through all patches
- More stable training for continual learning
- Improved representation learning

### Encoder Block LayerNorms-Only Training
- ~0.3-0.5% of total parameters remain trainable
- Best prevents catastrophic forgetting
- Minimal adaptation through encoder transformer block normalization only (norm1, norm2)

### All LayerNorms Training  
- ~1-2% of total parameters remain trainable
- Good balance between forgetting prevention and adaptation
- Allows adaptation through both encoder and decoder normalization

### Typical Parameter Counts (ViT-Large)
- **Total**: ~307M parameters
- **Trainable (Encoder Block LayerNorms-only)**: ~1.2M parameters (0.4%)
- **Trainable (All LayerNorms)**: ~3M parameters (1%)
- **Trainable (Encoder-frozen)**: ~50M parameters (16%)

## ⚠️ Important Notes

1. **Dataset Path**: Update `--data_path` to your actual dataset location
2. **Memory**: Adjust `--batch_size` based on your GPU memory
3. **Checkpoints**: Use `--continual_checkpoint` to load pretrained weights
4. **Learning Rate**: Lower learning rates (1e-4 to 1e-5) work better for continual learning

## 🔍 Monitoring Training

Training outputs are saved to:
- `./continual_output/` - Checkpoints and logs
- `./continual_output/log.txt` - Training statistics
- TensorBoard logs for visualization

```bash
# View training progress
tensorboard --logdir ./continual_output
```

## 🚨 Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the correct directory:
```bash
cd improving_mae/new
python continual_pretrain.py --help
```

### Memory Issues
Reduce batch size and/or use gradient accumulation:
```bash
python continual_pretrain.py \
    --batch_size 16 \
    --accum_iter 2  # Effective batch size = 16 * 2 = 32
```

### CUDA Issues
Run on CPU for testing:
```bash
python continual_pretrain.py --device cpu --batch_size 8
```

## 📚 References

- Original MAE paper: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- Continual Learning with Vision Transformers
- LayerNorm-based adaptation techniques 