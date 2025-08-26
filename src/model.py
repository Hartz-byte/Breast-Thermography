import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import yaml
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    """Convert 2D image patches to embedding vectors."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 9, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class CNNFeatureExtractor(nn.Module):
    """CNN backbone for local feature extraction."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, num_input_channels: int = 9):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')
        
        # Modify first conv layer for multiple views (9 channels)
        if num_input_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                num_input_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            
            # Initialize new conv layer
            with torch.no_grad():
                # Repeat weights for additional channels
                weight = original_conv.weight.repeat(1, num_input_channels // 3, 1, 1)
                self.backbone.conv1.weight = nn.Parameter(weight)
        
        self.feature_dim = self.backbone.num_features
        
    def forward(self, x):
        features = self.backbone(x)  # (B, feature_dim, H', W')
        return features

class HybridCNNViT(nn.Module):
    """Hybrid CNN-ViT model for breast thermography classification."""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 3,
                 embed_dim: int = 256,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 backbone: str = "resnet18",
                 num_input_channels: int = 9):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # CNN Feature Extractor
        self.cnn_extractor = CNNFeatureExtractor(
            backbone=backbone, 
            pretrained=True,
            num_input_channels=num_input_channels
        )
        
        # Adaptive pooling to reduce CNN feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # 14x14 = 196 patches
        
        # Project CNN features to ViT embedding dimension
        self.cnn_proj = nn.Conv2d(self.cnn_extractor.feature_dim, embed_dim, 1)
        
        # Position embedding
        self.n_patches = 196  # 14x14
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Auxiliary outputs for interpretability
        self.auxiliary_classifier = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """Forward pass through feature extraction."""
        B = x.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn_extractor(x)  # (B, C, H, W)
        cnn_features = self.adaptive_pool(cnn_features)  # (B, C, 14, 14)
        cnn_features = self.cnn_proj(cnn_features)  # (B, embed_dim, 14, 14)
        
        # Convert to sequence
        cnn_features = cnn_features.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, cnn_features), dim=1)  # (B, 197, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embedding
        
        # Apply transformer blocks
        attentions = []
        for transformer_block in self.transformer:
            x = transformer_block(x)
            # Store attention for visualization
            attentions.append(x[:, 1:])  # Exclude cls token
        
        x = self.norm(x)
        
        return x, attentions
    
    def forward(self, x):
        """Forward pass."""
        features, attentions = self.forward_features(x)
        
        # Classification using cls token
        cls_token_output = features[:, 0]
        main_logits = self.classifier(cls_token_output)
        
        # Auxiliary classification (average of all patch tokens)
        patch_features = features[:, 1:]  # Exclude cls token
        avg_patch_features = torch.mean(patch_features, dim=1)
        aux_logits = self.auxiliary_classifier(avg_patch_features)
        
        return {
            'main_logits': main_logits,
            'aux_logits': aux_logits,
            'features': features,
            'attentions': attentions[-1],  # Last layer attention for visualization
            'cls_token': cls_token_output
        }
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization."""
        with torch.no_grad():
            outputs = self.forward(x)
            attention = outputs['attentions']  # (B, 196, embed_dim)
            
            # Convert back to spatial dimensions
            B = attention.shape[0]
            attention_maps = attention.view(B, 14, 14, -1).permute(0, 3, 1, 2)
            attention_maps = F.interpolate(attention_maps, size=(224, 224), mode='bilinear')
            
            return attention_maps

class ModelBuilder:
    """Factory class for building models."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def build_model(self) -> HybridCNNViT:
        """Build the hybrid model."""
        model = HybridCNNViT(
            img_size=self.config['data']['image_size'][0],
            patch_size=self.config['model']['vit_patch_size'],
            num_classes=self.config['model']['num_classes'],
            embed_dim=self.config['model']['vit_embed_dim'],
            depth=self.config['model']['vit_depth'],
            num_heads=self.config['model']['vit_heads'],
            dropout=self.config['model']['dropout'],
            backbone=self.config['model']['backbone'],
            num_input_channels=9  # 3 views × 3 channels each
        )
        
        return model
    
    def count_parameters(self, model: nn.Module) -> dict:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

if __name__ == "__main__":
    # Test model
    builder = ModelBuilder()
    model = builder.build_model()
    
    # Count parameters
    param_count = builder.count_parameters(model)
    print(f"Model parameters: {param_count}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 9, 224, 224)  # Batch of 2, 9 channels (3 views)
    
    with torch.no_grad():
        outputs = model(dummy_input)
        print(f"Main logits shape: {outputs['main_logits'].shape}")
        print(f"Auxiliary logits shape: {outputs['aux_logits'].shape}")
        print(f"Features shape: {outputs['features'].shape}")
        print(f"Attention shape: {outputs['attentions'].shape}")
        
        # Test attention maps
        attention_maps = model.get_attention_maps(dummy_input)
        print(f"Attention maps shape: {attention_maps.shape}")
