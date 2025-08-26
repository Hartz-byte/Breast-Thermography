import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from typing import List, Tuple, Optional, Dict, Any
import yaml

class VisualizationUtils:
    """Utility class for creating various visualizations."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Create output directory
        os.makedirs(self.config['output']['plot_dir'], exist_ok=True)
    
    def plot_sample_images(self, data_loader, num_samples: int = 12, save_path: str = None):
        """Plot sample images from the dataset."""
        batch = next(iter(data_loader))
        images = batch['image'][:num_samples]
        labels = batch['label'][:num_samples]
        patient_ids = batch['patient_id'][:num_samples] if 'patient_id' in batch else None
        
        # Calculate grid dimensions
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            row, col = i // cols, i % cols
            
            # Take first 3 channels (anterior view) for visualization
            img = images[i, :3].numpy().transpose(1, 2, 0)
            
            # Denormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[row, col].imshow(img)
            title = f"Label: {labels[i].item()}"
            if patient_ids:
                title += f"\nID: {patient_ids[i]}"
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Sample Images from Dataset', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_multi_view_comparison(self, data_loader, sample_idx: int = 0, save_path: str = None):
        """Plot all three views for comparison."""
        batch = next(iter(data_loader))
        image = batch['image'][sample_idx]  # Shape: (9, H, W)
        label = batch['label'][sample_idx]
        patient_id = batch['patient_id'][sample_idx] if 'patient_id' in batch else f"Sample {sample_idx}"
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        views = ['Anterior', 'Left Oblique', 'Right Oblique']
        
        for i, view in enumerate(views):
            # Extract channels for each view
            start_ch = i * 3
            end_ch = (i + 1) * 3
            img = image[start_ch:end_ch].numpy().transpose(1, 2, 0)
            
            # Denormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f'{view} View')
            axes[i].axis('off')
        
        plt.suptitle(f'Multi-view Thermography - {patient_id} (Label: {label.item()})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

