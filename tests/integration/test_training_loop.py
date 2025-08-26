"""Integration tests for the training loop."""
import os
import sys
import torch
import pytest
import yaml
import tempfile
from src.train import Trainer, FocalLoss
from src.data_prep import DataPreprocessor

class TestTrainingLoop:
    """Test the training loop and components."""
    
    def test_trainer_init(self, test_config, temp_output_dir):
        """Test Trainer initialization."""
        # Update config with test output directory
        test_config['output']['model_dir'] = os.path.join(temp_output_dir, 'models')
        test_config['output']['plot_dir'] = os.path.join(temp_output_dir, 'plots')
        test_config['output']['result_dir'] = os.path.join(temp_output_dir, 'results')
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Initialize trainer with the test config
            trainer = Trainer(config_path=config_path)
            assert trainer is not None
            assert hasattr(trainer, 'config')
            assert hasattr(trainer, 'device')
        finally:
            # Clean up the temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
        
    def test_build_criterion(self, test_config, temp_output_dir):
        """Test building the loss criterion."""
        # Update config with test output directory
        test_config['output'].update({
            'model_dir': os.path.join(temp_output_dir, 'models'),
            'plot_dir': os.path.join(temp_output_dir, 'plots'),
            'result_dir': os.path.join(temp_output_dir, 'results')
        })
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            trainer = Trainer(config_path=config_path)
            
            # Create a dummy data loader with class weights
            class DummyDataset:
                def __init__(self):
                    self.data_preprocessor = DataPreprocessor(config_path)
                    self.data_preprocessor.class_weights = torch.tensor([1.0, 2.0, 3.0])  # Example weights
            
            class DummyDataLoader:
                def __init__(self):
                    self.dataset = DummyDataset()
                    
            dummy_loader = DummyDataLoader()
            
            # Test criterion creation
            criterion = trainer.build_criterion(dummy_loader)
            assert isinstance(criterion, FocalLoss)
        finally:
            # Clean up the temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
        
    def test_build_optimizer(self, test_config, temp_output_dir):
        """Test optimizer and scheduler creation."""
        from src.model import ModelBuilder
        
        # Update config with test output directory
        test_config['output'].update({
            'model_dir': os.path.join(temp_output_dir, 'models'),
            'plot_dir': os.path.join(temp_output_dir, 'plots'),
            'result_dir': os.path.join(temp_output_dir, 'results')
        })
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            trainer = Trainer(config_path=config_path)
            model_builder = ModelBuilder(config_path)
            model = model_builder.build_model()
            
            # Test optimizer and scheduler creation
            optimizer, scheduler = trainer.build_optimizer(model)
            assert isinstance(optimizer, torch.optim.AdamW)
            assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        finally:
            # Clean up the temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
        
    def test_train_epoch(self, test_config, device, temp_output_dir):
        """Test a single training epoch."""
        from src.model import ModelBuilder
        import torch.optim as optim

        # Update config with test output directory and ensure grad_clip is set
        test_config['output'].update({
            'model_dir': os.path.join(temp_output_dir, 'models'),
            'plot_dir': os.path.join(temp_output_dir, 'plots'),
            'result_dir': os.path.join(temp_output_dir, 'results')
        })
        test_config['training']['grad_clip'] = 1.0  # Add grad_clip to prevent KeyError
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Initialize components
            trainer = Trainer(config_path=config_path)
            model_builder = ModelBuilder(config_path)
            model = model_builder.build_model().to(device)
            
            # Create dummy data loader
            class DummyDataset:
                def __init__(self):
                    self.data_preprocessor = DataPreprocessor(config_path)
                    self.data_preprocessor.class_weights = torch.tensor([1.0, 2.0, 3.0])

                def __len__(self):
                    return 10

                def __getitem__(self, idx):
                    return {
                        'image': torch.randn(9, 224, 224),
                        'label': torch.tensor(1)  # Dummy label
                    }
                    
            dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
            
            # Initialize training components
            criterion = trainer.build_criterion(dummy_loader)
            optimizer = optim.AdamW(model.parameters())
            
            # Use the new torch.amp.GradScaler API
            scaler = torch.amp.GradScaler(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                                        enabled=test_config['training']['mixed_precision'])
            
            # Test training epoch
            loss, acc = trainer.train_epoch(model, dummy_loader, criterion, optimizer, scaler, 0)
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0 <= acc <= 1
        finally:
            # Clean up the temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
        
    def test_validation_epoch(self, test_config, device, temp_output_dir):
        """Test validation loop."""
        from src.model import ModelBuilder

        # Update config with test output directory
        test_config['output'].update({
            'model_dir': os.path.join(temp_output_dir, 'models'),
            'plot_dir': os.path.join(temp_output_dir, 'plots'),
            'result_dir': os.path.join(temp_output_dir, 'results')
        })
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Initialize components
            trainer = Trainer(config_path=config_path)
            model_builder = ModelBuilder(config_path)
            model = model_builder.build_model().to(device)
            
            # Create dummy data loader
            class DummyDataset:
                def __init__(self):
                    self.data_preprocessor = DataPreprocessor(config_path)
                    self.data_preprocessor.class_weights = torch.tensor([1.0, 2.0, 3.0])
                    
                def __len__(self):
                    return 10
                    
                def __getitem__(self, idx):
                    return {
                        'image': torch.randn(9, 224, 224),
                        'label': torch.tensor(1)  # Dummy label
                    }
                    
            dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
            
            # Initialize criterion
            criterion = trainer.build_criterion(dummy_loader)
            
            # Test validation epoch
            loss, acc, metrics = trainer.validate_epoch(model, dummy_loader, criterion, 0)
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0 <= acc <= 1
            assert 'accuracy' in metrics
            assert 'classification_report' in metrics
            assert 'confusion_matrix' in metrics
        finally:
            # Clean up the temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)
