# Continual MAE Pretraining on Places365

This directory contains scripts for continual pretraining of MAE models on the Places365 dataset.

## Files

- `continual.py` - ContinualMAE model class with freezing strategies
- `continual_pretrain_places365.py` - Main training script for Places365
- `run_continual_pretrain_places365.py` - Simple runner script with presets
- `load_places_sample.py` - Script to test Places365 dataset loading

## Quick Start

### 1. Run with presets (Recommended)

```bash
cd improving_mae/new
python run_continual_pretrain_places365.py
```

This will show you 3 configurations:
1. **LayerNorm Only (Recommended)** - Freezes everything except encoder transformer block LayerNorms
2. **All LayerNorms** - Freezes everything except all LayerNorms (encoder + decoder)
3. **Full Training** - Trains all parameters

### 2. Run directly with custom parameters

```bash
python continual_pretrain_places365.py \
    --model continual_mae_vit_large_patch16 \
    --epochs 50 \
    --batch_size 32 \
    --freeze_mode encoder_layernorms_only \
    --loss_mode all_pixels \
    --mask_ratio 0.75 \
    --blr 1e-4 \
    --data_path /path/to/places365 \
    --output_dir ./continual_output_places365
```

## Key Features

### Continual Learning Strategies

1. **Freezing Modes**:
   - `none`: Train all parameters
   - `encoder_layernorms_only`: Freeze everything except encoder transformer block LayerNorms (recommended for efficiency)
   - `layernorm_only`: Freeze everything except all LayerNorms (encoder + decoder)
   - `encoder_only`: Freeze encoder completely, train decoder only

2. **Loss Modes**:
   - `all_pixels`: Compute loss on all pixels (dense supervision, better for continual learning)
   - `masked_only`: Original MAE loss on masked patches only

3. **Checkpoint Loading**:
   - Use `--continual_checkpoint` to load pretrained weights and continue training
   - Supports loading from standard MAE checkpoints

### Model Architectures

- `continual_mae_vit_base_patch16`: ViT-Base (768 dim, 12 layers)
- `continual_mae_vit_large_patch16`: ViT-Large (1024 dim, 24 layers) 
- `continual_mae_vit_huge_patch14`: ViT-Huge (1280 dim, 32 layers)

## Recommended Settings

For continual learning from a pretrained MAE model:

```bash
python continual_pretrain_places365.py \
    --model continual_mae_vit_large_patch16 \
    --epochs 50 \
    --batch_size 32 \
    --freeze_mode encoder_layernorms_only \
    --loss_mode all_pixels \
    --mask_ratio 0.75 \
    --blr 1e-4 \
    --continual_checkpoint /path/to/pretrained/mae/checkpoint.pth \
    --warmup_epochs 10 \
    --weight_decay 0.05
```

## Dataset Requirements

The script expects Places365 dataset to be available at the specified path. By default it uses:
- Small version (256x256 images)
- Train-standard split
- Standard data augmentation (random crops, flips)

## Output

The training will create:
- Checkpoints every 20 epochs in `--output_dir`
- Tensorboard logs in `--log_dir`
- Training log in `log.txt`

## Monitoring

You can monitor training progress with:

```bash
tensorboard --logdir ./continual_output_places365
```

## Tips

1. **Memory Usage**: Use smaller batch sizes for larger models or limited GPU memory
2. **Learning Rate**: Start with lower learning rates (1e-4) for continual learning
3. **Freezing Strategy**: `encoder_layernorms_only` provides good balance between adaptation and efficiency
4. **Loss Mode**: `all_pixels` generally works better for continual learning scenarios 