# src/evaluate.py
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging
import argparse
from model import create_model
from data_preprocessing import COIL100Preprocessor, create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_comparison(multi_view_metrics, single_view_metrics, save_dir: Path):
    # Plot comparison between single-view and multi-view results
    plt.figure(figsize=(15, 5))
    # Plot training loss comparison
    plt.subplot(1, 3, 1)
    plt.plot(multi_view_metrics['train_loss'], label='Multi-view')
    plt.plot(single_view_metrics['train_loss'], label='Single-view')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot validation loss comparison
    plt.subplot(1, 3, 2)
    plt.plot(multi_view_metrics['val_loss'], label='Multi-view')
    plt.plot(single_view_metrics['val_loss'], label='Single-view')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot validation accuracy comparison
    plt.subplot(1, 3, 3)
    plt.plot(multi_view_metrics['val_acc'], label='Multi-view')
    plt.plot(single_view_metrics['val_acc'], label='Single-view')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png')
    plt.close()
    
    # Saving comparison metrics as JSON
    comparison = {
        'multi_view': {
            'final_train_loss': multi_view_metrics['train_loss'][-1],
            'final_val_loss': multi_view_metrics['val_loss'][-1],
            'final_val_acc': multi_view_metrics['val_acc'][-1],
            'best_val_acc': max(multi_view_metrics['val_acc'])
        },
        'single_view': {
            'final_train_loss': single_view_metrics['train_loss'][-1],
            'final_val_loss': single_view_metrics['val_loss'][-1],
            'final_val_acc': single_view_metrics['val_acc'][-1],
            'best_val_acc': max(single_view_metrics['val_acc'])
        }
    }
    with open(save_dir / 'comparison_metrics.json', 'w') as f:
        json.dump(comparison, f, indent=4)

def plot_metrics(metrics_file: Path, save_dir: Path):
    # Plot training and validation metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    plt.figure(figsize=(10, 5))
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['val_acc'], label='Val Accuracy')
    plt.title('Validation Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png')
    plt.close()

def evaluate_model(model, test_loader, device):
    # Evaluating model performance
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for views, labels in test_loader:
            views, labels = views.to(device), labels.to(device)
            outputs, _ = model(views)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    if args.compare_models:
        # Loading metrics for both models
        multi_view_metrics_file = save_dir / 'metrics.json'
        single_view_metrics_file = save_dir / 'single_view/metrics.json' 
        if multi_view_metrics_file.exists() and single_view_metrics_file.exists():
            with open(multi_view_metrics_file, 'r') as f:
                multi_view_metrics = json.load(f)
            with open(single_view_metrics_file, 'r') as f:
                single_view_metrics = json.load(f) 
            # Creating comparison plots
            plot_comparison(multi_view_metrics, single_view_metrics, save_dir)
            logger.info(f"Comparison plots saved to {save_dir}/model_comparison.png")
        else:
            logger.error("Could not find metrics files for both models")
            return
    else:
        # Original evaluation code
        preprocessor = COIL100Preprocessor(args.data_dir)
        objects_dict = preprocessor.organize_views()
        splits = preprocessor.create_splits(objects_dict)
        _, _, test_loader = create_data_loaders(
            splits,
            batch_size=args.batch_size,
            views_per_sample=args.views_per_sample
        )
        model = create_model(
            num_classes=100,
            num_views=args.views_per_sample,
            feature_dim=args.feature_dim,
            dropout_rate=0,
            fusion_method=args.fusion_method
        )
        checkpoint = torch.load(save_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        plot_metrics(save_dir / 'metrics.json', save_dir)
        predictions, labels = evaluate_model(model, test_loader, device)
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()
        
        report = classification_report(labels, predictions)
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write(report)   
    logger.info(f"Evaluation results saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Late Fusion Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--views_per_sample', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--fusion_method', type=str, default='concat')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare single-view and multi-view results')
    
    args = parser.parse_args()
    main(args)