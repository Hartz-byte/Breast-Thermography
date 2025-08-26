"""Unit tests for data loading and preprocessing."""
import os
import pytest
import numpy as np
from src.data_prep import DataPreprocessor, BreastThermographyDataset

class TestDataLoading:
    """Test data loading and preprocessing functionality."""
    
    def test_data_preprocessor_init(self, test_config):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'config')
        assert 'data' in preprocessor.config
        
    def test_load_diagnostics(self, test_config):
        """Test loading of diagnostics data."""
        preprocessor = DataPreprocessor()
        df = preprocessor.load_diagnostics()
        
        # Check required columns exist
        required_columns = ['Image', 'Left', 'Right', 'label']
        for col in required_columns:
            assert col in df.columns
            
        # Check label values are valid
        assert set(df['label'].unique()).issubset({0, 1, 2})
        
    def test_create_transforms(self, test_config):
        """Test creation of data augmentation transforms."""
        preprocessor = DataPreprocessor()
        
        # Test training transforms
        train_transforms = preprocessor.create_transforms(is_train=True)
        assert train_transforms is not None
        
        # Test validation transforms
        val_transforms = preprocessor.create_transforms(is_train=False)
        assert val_transforms is not None
        
    def test_breast_thermography_dataset(self, test_config):
        """Test dataset class functionality."""
        preprocessor = DataPreprocessor()
        df = preprocessor.load_diagnostics()
        
        # Create a small test dataset
        test_df = df.head(4)  # First 4 samples for testing
        dataset = BreastThermographyDataset(
            test_df,
            transform=preprocessor.create_transforms(is_train=False),
            config=test_config,
            data_preprocessor=preprocessor
        )
        
        # Test dataset length
        assert len(dataset) == len(test_df)
        
        # Test __getitem__
        sample = dataset[0]
        assert 'image' in sample
        assert 'label' in sample
        assert 'patient_id' in sample
        
        # Check image shape (672, 3, 224) - 672 = 3 views × 224 height, 3 channels
        expected_shape = (len(test_config['data']['views']) * test_config['data']['image_size'][0], 3, test_config['data']['image_size'][1])
        assert sample['image'].shape == expected_shape, f"Expected shape {expected_shape}, got {sample['image'].shape}"
        
        # Check label type and range
        assert isinstance(sample['label'], (int, np.integer))
        assert 0 <= sample['label'] <= 2  # Should be 0, 1, or 2
