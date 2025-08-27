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
from torch.cuda.amp import GradScaler
from torch.amp import autocast
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

def validate_batch_shapes(batch):
    """Validate batch tensor shapes before model forward pass."""
    images = batch['image']
    labels = batch['label']
    
    batch_size = images.size(0)
    expected_image_shape = (batch_size, 9, 224, 224)
    
    if images.shape != expected_image_shape:
        print(f"ERROR: Invalid image shape {images.shape}, expected {expected_image_shape}")
        return False
        
    if len(labels.shape) != 1 or labels.size(0) != batch_size:
        print(f"ERROR: Invalid label shape {labels.shape}")
        return False
        
    print(f"✓ Valid shapes - Images: {images.shape}, Labels: {labels.shape}")
    return True

class FocalLoss(nn.Module):
    """
    Focal Loss variant that supports:
    - Class balancing via alpha parameter
    - Label smoothing
    - Class-balanced focal loss
    - Focal loss with gamma parameter
    
    References:
        [1] https://arxiv.org/abs/1708.02002 (Focal Loss)
        [2] https://arxiv.org/abs/1901.05555 (Class-Balanced Loss)
    """
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0,
                 beta: Optional[float] = None,
                 class_balanced: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.beta = beta
        self.class_balanced = class_balanced
        
        if self.class_balanced and self.beta is None:
            self.beta = 0.999  # Default beta for class-balanced loss
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            log_probs = F.log_softmax(inputs, dim=-1)
            n_classes = inputs.size(-1)
            
            # Convert targets to one-hot and apply smoothing
            one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            smooth_labels = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            
            # Calculate cross entropy
            loss = - (smooth_labels * log_probs).sum(dim=-1)
        else:
            # Standard cross entropy
            loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate p_t (probability of true class)
        pt = torch.exp(-loss)
        
        # Focal loss component
        focal_term = (1 - pt) ** self.gamma
        
        # Class balanced weight if enabled
        if self.class_balanced and self.beta is not None:
            # Effective number of samples: (1 - beta^n) / (1 - beta)
            # where n is the number of samples in each class
            # We use inverse of effective number as weight
            effective_num = (1.0 - self.beta ** self.alpha) / (1.0 - self.beta)
            weights = (1.0 - self.beta) / effective_num
            weights = weights / weights.sum() * len(weights)  # Normalize to have mean 1
            
            if weights.device != targets.device:
                weights = weights.to(targets.device)
            class_weights = weights.gather(0, targets)
            focal_term = focal_term * class_weights
        # Standard class weights
        elif self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_term = focal_term * alpha_t
        
        # Combine terms
        focal_loss = focal_term * loss
        
        # Apply reduction
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
        """Build loss criterion with class weights and other parameters from config."""
        # Get class weights from the data loader
        class_weights = train_loader.dataset.data_preprocessor.class_weights.to(self.device)
        
        # Get loss parameters from config with defaults
        loss_cfg = self.config['training'].get('loss', {})
        gamma = loss_cfg.get('gamma', 2.0)
        label_smoothing = loss_cfg.get('label_smoothing', 0.0)
        use_class_balanced = loss_cfg.get('class_balanced', False)
        beta = loss_cfg.get('beta', 0.999)
        
        self.logger.info(f"Using class weights: {class_weights.tolist()}")
        self.logger.info(f"Loss config - gamma: {gamma}, label_smoothing: {label_smoothing}, "
                        f"class_balanced: {use_class_balanced}, beta: {beta}")
        
        # Initialize focal loss with configurable parameters
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
            class_balanced=use_class_balanced,
            beta=beta if use_class_balanced else None
        )
        
        return criterion
    
    def build_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
        """
        Build optimizer and scheduler with advanced learning rate scheduling.
        
        Supports:
        - Multiple optimizers (AdamW, Adam, SGD)
        - Learning rate warmup
        - Multiple scheduler types (cosine, step, plateau, one_cycle)
        - Gradient accumulation
        """
        
        # Get parameters to optimize (exclude those with requires_grad=False)
        params = [p for p in model.parameters() if p.requires_grad]
        
        # Get optimizer type from config
        optimizer_type = self.config['training'].get('optimizer', 'adamw').lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        # Set up optimizer
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=lr,
                momentum=self.config['training'].get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=self.config['training'].get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Set up learning rate warmup scheduler
        warmup_epochs = self.config['training'].get('warmup_epochs', 5)
        warmup_factor = self.config['training'].get('warmup_factor', 0.1)
        
        def warmup_lr_scheduler(epoch, warmup_epochs=warmup_epochs, warmup_factor=warmup_factor):
            if epoch < warmup_epochs:
                alpha = float(epoch) / warmup_epochs
                warmup_factor = warmup_factor * (1 - alpha) + alpha
                return warmup_factor
            return 1.0
        
        # Set up main learning rate scheduler
        scheduler_type = self.config['training'].get('scheduler', 'cosine').lower()
        min_lr = self.config['training'].get('min_lr', 1e-6)
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['epochs'] - warmup_epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config['training'].get('step_size', 10),
                gamma=self.config['training'].get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=min_lr
            )
        elif scheduler_type == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                steps_per_epoch=len(self.train_loader) // self.config['training'].get('gradient_accumulation_steps', 1),
                epochs=self.config['training']['epochs'] - warmup_epochs,
                pct_start=0.3,
                anneal_strategy='cos',
                final_div_factor=1e4
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Combine warmup and main scheduler
        def lr_lambda(epoch):
            warmup = warmup_lr_scheduler(epoch)
            if epoch < warmup_epochs:
                return warmup
            if scheduler_type == 'one_cycle':
                return 1.0  # Handled by OneCycleLR
            return scheduler.get_last_lr()[0] / lr if hasattr(scheduler, 'get_last_lr') else 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, 
                   scaler: GradScaler, epoch: int):
        """
        Train for one epoch with advanced features:
        - Gradient accumulation
        - Learning rate warmup
        - Mixed precision training
        - Gradient clipping
        - Learning rate scheduling
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation steps
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Progress bar with more detailed information
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        # Reset gradients
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=torch.float16, 
                         enabled=self.config['training']['mixed_precision']):
                outputs = model(images)
                
                # Handle dictionary output from model
                if isinstance(outputs, dict):
                    main_logits = outputs['main_logits']
                    aux_logits = outputs.get('aux_logits')
                    
                    # Calculate main loss
                    main_loss = criterion(main_logits, labels)
                    
                    # Calculate auxiliary loss if available
                    aux_loss = 0.0
                    if aux_logits is not None and self.config['training'].get('use_aux_loss', False):
                        aux_loss = criterion(aux_logits, labels) * self.config['training'].get('aux_loss_weight', 0.5)
                    
                    # Combine losses
                    loss = (main_loss + aux_loss) / accumulation_steps
                    
                    # Use main logits for accuracy calculation
                    outputs = main_logits
                else:
                    # Handle case where model returns raw logits (for backward compatibility)
                    loss = criterion(outputs, labels) / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Calculate accuracy
            with torch.no_grad():
                if isinstance(outputs, dict):
                    outputs = outputs['main_logits']
                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()
                
                # Update running metrics
                total += batch_total
                correct += batch_correct
                running_loss += loss.item() * accumulation_steps  # Scale back loss
            
            # Perform optimization step after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=self.config['training']['grad_clip']
                )
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                avg_acc = 100. * correct / total if total > 0 else 0.0
                current_lr = optimizer.param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%',
                    'lr': f'{current_lr:.2e}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Log metrics
        if hasattr(self, 'logger'):
            self.logger.info(
                f"Epoch {epoch+1} - "
                f"Train Loss: {epoch_loss:.4f}, "
                f"Train Acc: {epoch_acc:.2f}%, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        
        return epoch_loss, epoch_acc
    
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
                
                with autocast(device_type='cuda', enabled=self.config['training']['mixed_precision']):
                    outputs = model(images)
                    
                    # Handle dictionary output from model
                    if isinstance(outputs, dict):
                        main_logits = outputs['main_logits']
                        aux_logits = outputs.get('aux_logits')
                        
                        # Calculate main loss
                        main_loss = criterion(main_logits, labels)
                        
                        # Calculate auxiliary loss if available and enabled
                        aux_loss = 0.0
                        if aux_logits is not None and self.config['training'].get('use_aux_loss', False):
                            aux_loss = criterion(aux_logits, labels) * self.config['training'].get('aux_loss_weight', 0.5)
                        
                        # Combine losses
                        loss = main_loss + aux_loss
                        
                        # Get predictions from main logits
                        predictions = torch.argmax(main_logits, dim=1)
                    else:
                        # Handle case where model returns raw logits (for backward compatibility)
                        loss = criterion(outputs, labels)
                        predictions = torch.argmax(outputs, dim=1)
                
                running_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        
        # Detailed metrics
        metrics = {
            'accuracy': epoch_acc,
            'classification_report': classification_report(all_labels, all_predictions, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
        
        return epoch_loss, epoch_acc, metrics
        
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, best_train_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        # Ensure output directory exists
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_train_acc': best_train_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['output']['model_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Saved checkpoint for epoch {epoch+1}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['output']['model_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 New best model saved! Training Acc: {best_train_acc*100:.2f}% at epoch {epoch+1}")
            
            # Save a copy with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamped_path = os.path.join(
                self.config['output']['model_dir'], 
                f'best_model_acc_{best_train_acc*100:.2f}_epoch_{epoch+1}_{timestamp}.pth'
            )
            torch.save(checkpoint, timestamped_path)
            self.logger.info(f"📌 Saved timestamped best model: {os.path.basename(timestamped_path)}")
    
    def plot_training_history(self):
        """Plot and save training history."""
        if not self.train_losses or not self.val_losses:
            self.logger.warning("No training history to plot")
            return
            
        # Create plot directory if it doesn't exist
        os.makedirs(self.config['output']['plot_dir'], exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save and show the plot
        plt.tight_layout()
        plot_path = os.path.join(self.config['output']['plot_dir'], 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved training history plot to {plot_path}")
        plt.close()  # Close the figure to free memory

    def train(self) -> HybridCNNViT:
        """Main training function.
        
        Returns:
            HybridCNNViT: The trained model with the best training accuracy.
            
        This method orchestrates the entire training process including:
        - Data preparation
        - Model and training components initialization
        - Training and validation loops
        - Checkpointing and early stopping based on training accuracy
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
        
        # Training state
        best_train_acc = 0.0
        best_metrics = {}
        
        # Create figure for live updating plots
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
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
            log_msg = (
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            self.logger.info(log_msg)
            
            # Check for best model based on training accuracy
            is_best = train_acc > best_train_acc
            if is_best:
                best_train_acc = train_acc
                best_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'classification_report': val_metrics.get('classification_report', {})
                }
                self.logger.info(f"🎯 New best training accuracy: {best_train_acc*100:.2f}%")
            
            # Save checkpoint (both latest and best if applicable)
            self.save_checkpoint(model, optimizer, epoch, best_train_acc, is_best)
            
            # Update plots
            self._update_plots(fig, ax1, ax2)
            
            # Early stopping
            if early_stopping(val_loss, model):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Final update of plots
        self._update_plots(fig, ax1, ax2, save=True)
        plt.ioff()
        plt.close()
        
        # Load best model weights
        best_model_path = os.path.join(self.config['output']['model_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("\n✅ Loaded best model weights from checkpoint")
            
            # Log best metrics in a clear format
            self.logger.info("\n" + "="*70)
            self.logger.info("🏆 Best Model Performance (Based on Training Accuracy)")
            self.logger.info("="*70)
            self.logger.info(f"📊 Epoch: {best_metrics['epoch'] + 1}")
            self.logger.info(f"📉 Training Loss: {best_metrics['train_loss']:.4f}")
            self.logger.info(f"📈 Validation Loss: {best_metrics['val_loss']:.4f}")
            self.logger.info(f"🎯 Training Accuracy: {best_train_acc*100:.2f}%")
            self.logger.info(f"🏅 Validation Accuracy: {best_metrics['val_acc']*100:.2f}%")
            
            # Log classification report if available
            if 'classification_report' in best_metrics and best_metrics['classification_report']:
                self.logger.info("\n📋 Classification Report:")
                report = best_metrics['classification_report']
                if 'accuracy' in report:
                    self.logger.info(f"   - Overall Accuracy: {report['accuracy']*100:.2f}%")
                if 'weighted avg' in report:
                    avg = report['weighted avg']
                    self.logger.info(f"   - Weighted Avg - Precision: {avg['precision']:.4f}, "
                                  f"Recall: {avg['recall']:.4f}, F1: {avg['f1-score']:.4f}")
            self.logger.info("="*70 + "\n")
            
        return model
        
    def _update_plots(self, fig, ax1, ax2, save=False):
        """Update training plots in real-time."""
        if not self.train_losses or not self.val_losses:
            return
            
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot loss
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            plot_path = os.path.join(self.config['output']['plot_dir'], 'training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved final training plots to {plot_path}")
        
        plt.pause(0.1)  # Pause to update the plot
        
        # Plot loss
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            plot_path = os.path.join(self.config['output']['plot_dir'], 'training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved final training plots to {plot_path}")
        
        plt.pause(0.1)  # Pause to update the plot

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
