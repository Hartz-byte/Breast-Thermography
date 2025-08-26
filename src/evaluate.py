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
    
    def evaluate_model(self, model: HybridCNNViT, data_loader, 
                      class_names: List[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(images)
                probabilities = F.softmax(outputs['main_logits'], dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_features.extend(outputs['cls_token'].cpu().numpy())
        
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
