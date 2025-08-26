"""Shared pytest fixtures for testing."""
import sys
import os
import pytest
import torch
import yaml
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

@pytest.fixture(scope="session")
def test_config():
    """Load and return test configuration."""
    config_path = os.path.join("configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for testing
    config['training']['batch_size'] = 2
    config['training']['epochs'] = 1
    config['hardware']['num_workers'] = 0  # Avoid issues with multiprocessing in tests
    
    return config

@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create and return a temporary output directory for tests."""
    return str(tmp_path / "test_outputs")
