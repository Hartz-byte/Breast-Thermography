"""
Visualize training metrics and plots from saved checkpoints.
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    return checkpoint

def plot_training_metrics(checkpoint, output_dir=None):
    """Plot training and validation metrics from checkpoint."""
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    
    if not train_losses or not val_losses:
        print("No training metrics found in checkpoint")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Acc')
    plt.plot(epochs, val_accuracies, 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved training metrics plot to {plot_path}")
    
    plt.show()

def print_best_metrics(checkpoint):
    """Print best training metrics from checkpoint."""
    best_train_acc = checkpoint.get('best_train_acc', 0)
    best_epoch = checkpoint.get('epoch', 0)
    
    print("\n" + "="*70)
    print("🏆 Best Model Performance")
    print("="*70)
    print(f"📊 Best Epoch: {best_epoch + 1}")
    print(f"🎯 Best Training Accuracy: {best_train_acc*100:.2f}%")
    
    if 'val_accuracies' in checkpoint and checkpoint['val_accuracies']:
        best_val_acc = max(checkpoint['val_accuracies'])
        print(f"🏅 Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    if 'classification_report' in checkpoint:
        report = checkpoint['classification_report']
        if isinstance(report, dict):
            print("\n📋 Classification Report:")
            if 'accuracy' in report:
                print(f"   - Overall Accuracy: {report['accuracy']*100:.2f}%")
            if 'weighted avg' in report:
                avg = report['weighted avg']
                print(f"   - Weighted Avg - Precision: {avg['precision']:.4f}, "
                      f"Recall: {avg['recall']:.4f}, F1: {avg['f1-score']:.4f}")
    
    print("="*70 + "\n")

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up paths
    output_dir = config['output']['plot_dir']
    model_dir = config['output']['model_dir']
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Load best checkpoint
    try:
        checkpoint = load_checkpoint(best_model_path)
        
        # Plot metrics
        plot_training_metrics(checkpoint, output_dir)
        
        # Print best metrics
        print_best_metrics(checkpoint)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have run training and that the model checkpoints exist.")
        print(f"Expected best model at: {best_model_path}")

if __name__ == "__main__":
    main()
