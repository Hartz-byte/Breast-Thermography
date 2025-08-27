import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import yaml
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.model import HybridCNNViT, ModelBuilder
from src.data_prep import DataPreprocessor

class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(self.config['output']['plot_dir'], exist_ok=True)
        os.makedirs(self.config['output']['result_dir'], exist_ok=True)
        
        # Initialize data preprocessor for label mappings
        self.data_preprocessor = DataPreprocessor(config_path)
        self.label_mappings = None
        
    def load_model(self, model_path: str) -> HybridCNNViT:
        """Load trained model from checkpoint."""
        model_builder = ModelBuilder()
        model = model_builder.build_model()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return model
    
    def _process_model_outputs(self, outputs):
        """
        Process model outputs to handle different output formats.
        
        Args:
            outputs: Model outputs which could be a tensor, dict, or tuple
            
        Returns:
            tuple: (logits, features) where features could be None
            
        Raises:
            ValueError: If output format is not recognized
        """
        features = None
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # Handle dictionary outputs
            logits = None
            
            # Check common logit keys
            for key in ['logits', 'main_logits', 'out', 'output', 'predictions']:
                if key in outputs and torch.is_tensor(outputs[key]):
                    logits = outputs[key]
                    break
            
            # If no standard logit key found, look for any tensor
            if logits is None:
                for v in outputs.values():
                    if torch.is_tensor(v) and v.dim() >= 2:  # Ensure it's a batch of logits
                        logits = v
                        break
            
            # Get features if available
            for feat_key in ['cls_token', 'features', 'embedding', 'hidden_state']:
                if feat_key in outputs and torch.is_tensor(outputs[feat_key]):
                    features = outputs[feat_key]
                    break
                    
        elif isinstance(outputs, (tuple, list)):
            # Handle tuple/list outputs (common in some architectures)
            if len(outputs) > 1:
                logits, features = outputs[0], outputs[1]
            else:
                logits = outputs[0]
                
        elif torch.is_tensor(outputs):
            # Handle direct tensor output
            logits = outputs
            
        else:
            raise ValueError(f"Unexpected model output type: {type(outputs)}")
        
        # Validate logits
        if logits is None or not torch.is_tensor(logits) or logits.dim() < 2:
            raise ValueError("Could not extract valid logits from model output")
            
        return logits, features
    
    def evaluate_model(self, model: nn.Module, data_loader, 
                      class_names: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with support for different output formats.
        
        Args:
            model: The model to evaluate
            data_loader: DataLoader for evaluation data
            class_names: Optional list of class names for metrics
            
        Returns:
            Dict containing evaluation metrics and results
        """
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_features = []
        batch_losses = []
        
        # Get loss function from config or use default
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                try:
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        images = batch['image'].to(self.device, non_blocking=True)
                        labels = batch['label'].to(self.device, non_blocking=True)
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    else:
                        print(f"Skipping unexpected batch type: {type(batch)}")
                        continue
                    
                    # Forward pass with autocast for mixed precision
                    with torch.cuda.amp.autocast(enabled=self.config.get('training', {}).get('mixed_precision', True)):
                        outputs = model(images)
                        
                        # Process outputs
                        logits, features = self._process_model_outputs(outputs)
                        
                        # Calculate loss
                        loss = loss_fn(logits, labels)
                        batch_losses.append(loss.item())
                        
                        # Get predictions and probabilities
                        probabilities = F.softmax(logits, dim=1)
                        predictions = torch.argmax(probabilities, dim=1)
                        
                        # Store results
                        all_predictions.append(predictions.cpu().numpy())
                        all_probabilities.append(probabilities.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                        
                        if features is not None:
                            all_features.append(features.cpu().numpy())
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    if batch_idx == 0:  # If first batch fails, it's likely a critical error
                        raise
                    continue
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        all_features = np.array(all_features)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities, class_names)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels,
            'features': all_features
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray, class_names: List[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        num_classes = len(np.unique(y_true))
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (for multiclass)
        try:
            if num_classes > 2:
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                auc_scores = []
                for i in range(num_classes):
                    if len(np.unique(y_true_bin[:, i])) > 1:
                        auc_scores.append(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
                    else:
                        auc_scores.append(0.0)
                roc_auc = np.mean(auc_scores)
            else:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        except:
            roc_auc = 0.0
            auc_scores = [0.0] * num_classes
        
        # Per-class metrics
        per_class_metrics = []
        for i in range(num_classes):
            per_class_metrics.append({
                'class': class_names[i],
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i],
                'auc': auc_scores[i] if 'auc_scores' in locals() else 0.0
            })
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        }
