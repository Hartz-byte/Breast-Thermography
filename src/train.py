# Standard library imports
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Local application imports
from src.model import ModelBuilder, HybridCNNViT
from src.data_prep import DataPreprocessor

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class Trainer:
    """Training manager for the hybrid CNN-ViT model.
    
    This class handles the entire training pipeline including:
    - Model initialization and setup
    - Data loading and preprocessing
    - Training and validation loops
    - Checkpointing and logging
    - Performance monitoring and early stopping
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        os.makedirs(self.config['output']['plot_dir'], exist_ok=True)
        os.makedirs(self.config['output']['result_dir'], exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model_builder = ModelBuilder(config_path)
        self.data_preprocessor = DataPreprocessor(config_path)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.config['output']['result_dir'], 
                             f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_model(self) -> HybridCNNViT:
        """Build and initialize the model."""
        model = self.model_builder.build_model()
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['training']['gradient_checkpointing']:
            model.gradient_checkpointing = True
            
        # Log model information
        param_count = self.model_builder.count_parameters(model)
        self.logger.info(f"Model parameters: {param_count}")
        
        return model
    
    def build_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Build loss criterion with class weights."""
        # Get class weights from the data loader
        class_weights = train_loader.dataset.data_preprocessor.class_weights.to(self.device)
        
        self.logger.info(f"Using class weights: {class_weights.tolist()}")
        
        # Use focal loss for better handling of imbalanced data
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        return criterion
    
    def build_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
        """Build optimizer and scheduler."""
        # Different learning rates for CNN backbone and ViT parts
        cnn_params = list(model.cnn_extractor.parameters())
        vit_params = [p for n, p in model.named_parameters() if 'cnn_extractor' not in n]
        
        param_groups = [
            {'params': cnn_params, 'lr': self.config['training']['learning_rate'] * 0.1},  # Lower LR for pretrained CNN
            {'params': vit_params, 'lr': self.config['training']['learning_rate']}
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Cosine annealing with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs']
        )
        
        return optimizer, scheduler
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, epoch: int) -> Tuple[float, float, Dict[str, Any]]:
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config['training']['mixed_precision']):
                    outputs = model(images)
                    loss = criterion(outputs['main_logits'], labels)
                    
                    # Add auxiliary loss if available
                    if 'aux_logits' in outputs:
                        aux_loss = criterion(outputs['aux_logits'], labels)
                        loss = loss + 0.5 * aux_loss
                
                running_loss += loss.item()
                predictions = torch.argmax(outputs['main_logits'], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        
        # Detailed metrics
        metrics = {
            'accuracy': epoch_acc,
            'classification_report': classification_report(all_labels, all_predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
        
        return epoch_loss, epoch_acc, metrics
        
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, 
                   scaler: GradScaler, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            with autocast(enabled=self.config['training']['mixed_precision']):
                outputs = model(images)
                loss = criterion(outputs['main_logits'], labels)
                
                # Add auxiliary loss if available
                if 'aux_logits' in outputs:
                    aux_loss = criterion(outputs['aux_logits'], labels)
                    loss = loss + 0.5 * aux_loss
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['grad_clip'])
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item()
            predictions = torch.argmax(outputs['main_logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        
        return epoch_loss, epoch_acc

def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                   epoch: int, best_val_loss: float, is_best: bool = False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'config': self.config
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(self.config['output']['model_dir'], 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(self.config['output']['model_dir'], 'best_model.pth')
        torch.save(checkpoint, best_path)
        self.logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

def plot_training_history(self):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(self.train_losses, label='Train Loss', color='blue')
    axes[0].plot(self.val_losses, label='Validation Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(self.train_accuracies, label='Train Accuracy', color='blue')
    axes[1].plot(self.val_accuracies, label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(self.config['output']['plot_dir'], 'training_history.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

    def train(self) -> HybridCNNViT:
        """Main training function.
        
        Returns:
            HybridCNNViT: The trained model with the best validation performance.
            
        This method orchestrates the entire training process including:
        - Data preparation
        - Model and training components initialization
        - Training and validation loops
        - Checkpointing and early stopping
        """
        self.logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader, df = self.data_preprocessor.create_dataloaders()
        self.logger.info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        
        # Build model and training components
        model = self.build_model()
        criterion = self.build_criterion(train_loader)
        optimizer, scheduler = self.build_optimizer(model)
        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
            
            # Validation phase
            val_loss, val_acc, val_metrics = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # Scheduler step
            scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(model, optimizer, epoch, best_val_loss, is_best)
            
            # Early stopping
            if early_stopping(val_loss, model):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Plot training history
        self.plot_training_history()
        
        self.logger.info("Training completed!")
        return model

def main():
    """Main function to initialize and run the trainer."""
    try:
        trainer = Trainer()
        model = trainer.train()
        return model
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
