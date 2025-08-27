# 03_model_training_fixed.py
# Imports
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from src.train import Trainer

class TrainingMonitor:
    """Utility class for monitoring and visualizing training progress."""
    
    @staticmethod
    def plot_training_history(trainer, save_path=None):
        """Plot training and validation metrics."""
        if not hasattr(trainer, 'train_losses') or not hasattr(trainer, 'val_losses'):
            print("No training history found to plot.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot losses
        ax1.plot(trainer.train_losses, label='Train Loss')
        ax1.plot(trainer.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies if available
        if hasattr(trainer, 'train_accuracies'):
            ax2.plot(trainer.train_accuracies, label='Train Accuracy')
            ax2.plot(trainer.val_accuracies, label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration
    config_path = project_root / 'configs' / 'config.yaml'
    
    try:
        # Initialize trainer with config
        print("Initializing trainer...")
        trainer = Trainer(config_path=config_path)
        
        # Build model
        print("\nBuilding model...")
        model = trainer.build_model()
        
        # Get data loaders
        print("\nPreparing data...")
        
        # First load the diagnostics to ensure data is properly loaded
        df = trainer.data_preprocessor.load_diagnostics()
        print("\nDataFrame loaded with columns:", df.columns.tolist())
        print("Sample data:")
        print(df[['Image', 'Left', 'Right', 'combined_label']].head())
        
        # Create datasets
        print("\nCreating datasets...")
        train_loader, val_loader, _ = trainer.data_preprocessor.create_datasets()
        
        # Print dataset information
        print(f"\nTraining samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Set batch size
        batch_size = trainer.config['training']['batch_size']
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=trainer.config['hardware'].get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=trainer.config['hardware'].get('num_workers', 4),
            pin_memory=True
        )
        
        # Initialize optimizer and scaler
        print("\nInitializing optimizer and scaler...")
        optimizer, scheduler = trainer.build_optimizer(model)
        
        # Use BCEWithLogitsLoss for binary classification
        # Note: Ensure your model outputs a single value per sample (not 2 values)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        # Update model's final layer for binary classification if needed
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            if model.classifier.out_features != 1:  # If not already set for binary
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, 1)
        
        # Start training
        print("\nStarting training...")
        num_epochs = trainer.config['training']['epochs']
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = trainer.train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch
            )
            
            # Validate
            val_loss, val_acc, _ = trainer.validate_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                epoch=epoch
            )
            
            # Step the scheduler
            scheduler.step()
            
            # Print progress with binary classification metrics
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(trainer.config['output']['model_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': trainer.config
        }, final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Plot and save training history
        plot_path = os.path.join(trainer.config['output']['plot_dir'], 'training_history.png')
        TrainingMonitor.plot_training_history(trainer, save_path=plot_path)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
