# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Continual Learning MAE
# A super class of MAE ViT for continual learning scenarios
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from functools import partial
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models_mae import MaskedAutoencoderViT
from util.pos_embed import get_2d_sincos_pos_embed


class ContinualMAE(MaskedAutoencoderViT):
    """
    Continual Learning version of Masked Autoencoder with VisionTransformer backbone
    
    Key differences from base MAE:
    1. Loss computed on ALL pixel values, not just masked patches
    2. Support for freezing everything except LayerNorm layers
    3. Additional continual learning utilities
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 continual_mode=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                        decoder_embed_dim, decoder_depth, decoder_num_heads,
                        mlp_ratio, norm_layer, norm_pix_loss)
        
        self.continual_mode = continual_mode
        print(f"ContinualMAE initialized with continual_mode={continual_mode}")
    
    def freeze_everything_except_layernorm_encoder(self):
        """
        Freeze all parameters except LayerNorm layers in the ENCODER only.
        This is useful for continual learning to prevent catastrophic forgetting
        while allowing adaptation through encoder normalization layers.
        The decoder remains completely frozen.
        
        Specifically targets:
        - blocks.*.norm1.weight/bias (pre-attention LayerNorm)
        - blocks.*.norm2.weight/bias (pre-MLP LayerNorm)  
        - norm.weight/bias (final encoder LayerNorm)
        """
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            # Check if parameter belongs to encoder LayerNorm layers specifically
            is_encoder_blocks_layernorm = (
                # LayerNorm in transformer blocks (norm1, norm2)
                (name.startswith('blocks.') and ('.norm1.' in name or '.norm2.' in name) and ('weight' in name or 'bias' in name))
            )
            
            if is_encoder_blocks_layernorm:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"Keeping trainable (encoder LayerNorm): {name} (shape: {param.shape})")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"\nFreezing summary (encoder LayerNorms only):")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {frozen_params + trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / (frozen_params + trainable_params):.2f}%")
    
    def freeze_everything_except_all_layernorms(self):
        """
        Freeze all parameters except LayerNorm layers in BOTH encoder and decoder.
        This provides more adaptation capability than encoder-only LayerNorms.
        
        Specifically targets:
        - blocks.*.norm1.weight/bias (pre-attention LayerNorm in encoder)
        - blocks.*.norm2.weight/bias (pre-MLP LayerNorm in encoder)  
        - norm.weight/bias (final encoder LayerNorm)
        - decoder_blocks.*.norm1.weight/bias (pre-attention LayerNorm in decoder)
        - decoder_blocks.*.norm2.weight/bias (pre-MLP LayerNorm in decoder)
        - decoder_norm.weight/bias (final decoder LayerNorm)
        """
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            # Check if parameter belongs to any LayerNorm layer (encoder or decoder)
            is_layernorm = (
                # Encoder LayerNorms in transformer blocks
                (name.startswith('blocks.') and ('.norm1.' in name or '.norm2.' in name) and ('weight' in name or 'bias' in name)) or
                # Final encoder LayerNorm
                (name == 'norm.weight' or name == 'norm.bias') or
                # Decoder LayerNorms in transformer blocks  
                (name.startswith('decoder_blocks.') and ('.norm1.' in name or '.norm2.' in name) and ('weight' in name or 'bias' in name)) or
                # Final decoder LayerNorm
                (name == 'decoder_norm.weight' or name == 'decoder_norm.bias')
            )
            
            if is_layernorm:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"Keeping trainable (LayerNorm): {name} (shape: {param.shape})")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"\nFreezing summary (all LayerNorms):")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {frozen_params + trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / (frozen_params + trainable_params):.2f}%")

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")
    
    def forward_loss_all_pixels(self, imgs, pred, mask=None):
        """
        Compute loss on ALL pixel values, not just masked patches.
        This provides a dense supervision signal useful for continual learning.
        
        Args:
            imgs: [N, 3, H, W] - original images
            pred: [N, L, p*p*3] - predicted patches
            mask: [N, L] - mask (optional, ignored in all-pixel mode)
        """
        target = self.patchify(imgs)  # [N, L, p*p*3]
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Compute loss on ALL patches, not just masked ones
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()  # Mean loss across all patches and batch
        
        return loss
    
    def forward(self, imgs, mask_ratio=0.75, loss_mode='all_pixels'):
        """
        Forward pass with different loss modes for continual learning.
        
        Args:
            imgs: input images
            mask_ratio: masking ratio  
            loss_mode: 'all_pixels', or 'masked_only'
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        if loss_mode == 'all_pixels':
            loss = self.forward_loss_all_pixels(imgs, pred, mask)
        else:  # 'masked_only' - original MAE loss
            loss = self.forward_loss(imgs, pred, mask)
            
        return loss, pred, mask
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_frozen_params(self):
        """Get number of frozen parameters"""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    def print_param_status(self):
        """Print detailed parameter status"""
        trainable = self.get_num_trainable_params()
        frozen = self.get_num_frozen_params()
        total = trainable + frozen
        
        print(f"\nParameter Status:")
        print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"Frozen: {frozen:,} ({100*frozen/total:.2f}%)")
        print(f"Total: {total:,}")


def continual_mae_vit_base_patch16(**kwargs):
    """Continual MAE ViT-Base model"""
    model = ContinualMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def continual_mae_vit_large_patch16(**kwargs):
    """Continual MAE ViT-Large model"""
    model = ContinualMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def continual_mae_vit_huge_patch14(**kwargs):
    """Continual MAE ViT-Huge model"""
    model = ContinualMAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Continual MAE Test")
    print("=" * 50)
    
    # Create model
    model = continual_mae_vit_large_patch16()
    
    # Print initial status
    model.print_param_status()
    
    # Test freezing
    print("\nFreezing everything except encoder LayerNorms...")
    model.freeze_everything_except_layernorm_encoder()
    
    # Print status after freezing
    model.print_param_status()
    
    # Test forward pass with different loss modes
    print("\nTesting forward pass...")
    imgs = torch.randn(2, 3, 224, 224)
    
    # Test all-pixel loss
    loss_all, pred, mask = model(imgs, mask_ratio=0.75, loss_mode='all_pixels')
    print(f"All-pixel loss: {loss_all.item():.4f}")

    # Test original MAE loss
    loss_masked, pred, mask = model(imgs, mask_ratio=0.75, loss_mode='masked_only')
    print(f"Masked-only loss: {loss_masked.item():.4f}")
    
    print("\nContinual MAE ready for use!") 