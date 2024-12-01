# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from model import create_model
from data_preprocessing import COIL100Preprocessor, create_data_loaders

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    # Training model for one epoch
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (views, targets) in enumerate(progress_bar):
        # Moving data to device
        views, targets = views.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forwarding pass
        outputs, _ = model(views)
        # Calculating loss
        loss = criterion(outputs, targets)
        # Backwarding pass and optimiz
        loss.backward()
        optimizer.step()
        # Updating metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # Updating progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    return total_loss / len(train_loader)

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for views, targets in val_loader:
            views, targets = views.to(device), targets.to(device)
            outputs, _ = model(views)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def main(args):
    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    # Creating directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Preparing data
    preprocessor = COIL100Preprocessor(args.data_dir)
    if not (Path(args.data_dir) / 'coil-100').exists():
        preprocessor.download_dataset()
    objects_dict = preprocessor.organize_views()
    splits = preprocessor.create_splits(objects_dict, seed=args.seed)
    # Creating data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        splits,
        batch_size=args.batch_size,
        views_per_sample=args.views_per_sample,
        num_workers=args.num_workers
    )
    # Creating model
    model = create_model(
        num_classes=100, 
        num_views=args.views_per_sample,
        feature_dim=args.feature_dim,
        dropout_rate=args.dropout_rate,
        fusion_method=args.fusion_method
    )
    model = model.to(device)
    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    # Training loop
    best_val_acc = 0
    metrics = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validating
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        # Updating learning rate
        scheduler.step(val_loss)
        # Log metrics
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # Saving metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        # Saving best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_dir / 'best_model.pth')
        # Saving latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, save_dir / 'latest_model.pth')
        # Saving metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Late Fusion Model')
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save models and results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--views_per_sample', type=int, default=3,
                       help='Number of views per object')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=2048,
                       help='Dimension of feature vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--fusion_method', type=str, default='concat',
                       choices=['concat', 'mean', 'max'],
                       help='Method to fuse features from multiple views')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)