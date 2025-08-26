"""Unit tests for model components."""
import torch
import pytest
from src.model import ModelBuilder, HybridCNNViT
from src.data_prep import DataPreprocessor

class TestModel:
    """Test model components and forward pass."""
    
    def test_model_builder_init(self, test_config):
        """Test ModelBuilder initialization."""
        builder = ModelBuilder()
        assert builder is not None
        assert hasattr(builder, 'config')
        assert 'model' in builder.config
        
    def test_build_model(self, test_config, device):
        """Test model building and forward pass."""
        builder = ModelBuilder()
        model = builder.build_model().to(device)
        
        # Check model structure
        assert isinstance(model, HybridCNNViT)
        assert hasattr(model, 'cnn_extractor')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'classifier')
        
        # Test forward pass with dummy input
        batch_size = 2
        channels = 9  # 3 views × 3 channels
        height = test_config['data']['image_size'][0]
        width = test_config['data']['image_size'][1]
        
        dummy_input = torch.randn(batch_size, channels, height, width).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            
        # Check output shapes
        assert 'main_logits' in outputs
        assert 'aux_logits' in outputs
        assert outputs['main_logits'].shape == (batch_size, test_config['model']['num_classes'])
        assert outputs['aux_logits'].shape == (batch_size, test_config['model']['num_classes'])
        
    def test_model_parameters(self, test_config, device):
        """Test model parameter counts and gradient requirements."""
        builder = ModelBuilder()
        model = builder.build_model().to(device)
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable
        
    def test_model_with_attention(self, test_config, device):
        """Test attention mechanism in the model."""
        builder = ModelBuilder()
        model = builder.build_model().to(device)
        
        # Create test input
        batch_size = 2
        channels = 9
        height = test_config['data']['image_size'][0]
        width = test_config['data']['image_size'][1]
        dummy_input = torch.randn(batch_size, channels, height, width).to(device)
        
        # Get model outputs (attention is always returned in the output dict)
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
            
        # Check attention maps in output
        assert 'attentions' in outputs
        assert isinstance(outputs['attentions'], torch.Tensor)
        # Check shape: (batch_size, num_patches, embed_dim)
        assert len(outputs['attentions'].shape) == 3
