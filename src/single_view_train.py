# src/single_view_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import random
from data_preprocessing import COIL100Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleViewCOIL100Dataset(Dataset):
    # Dataset class for single view COIL-100
    def __init__(self, data_dict, transform=None, front_view_only=True):
        self.data_dict = data_dict
        self.objects = list(data_dict.keys())
        self.transform = transform
        self.front_view_only = front_view_only
        self.obj_to_label = {obj: idx for idx, obj in enumerate(self.objects)}
        
    def __len__(self):
        return len(self.objects)
        
    def __getitem__(self, idx):
        obj_id = self.objects[idx]
        views = self.data_dict[obj_id]
        # If front_view_only always use the first view (front view),
        # otherwise randomly select one view
        view_path = views[0] if self.front_view_only else random.choice(views)
        img = Image.open(view_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.obj_to_label[obj_id]
        return img, label

class SingleViewModel(nn.Module):
    def __init__(self, num_classes):
        super(SingleViewModel, self).__init__()
        # Using ResNet50 backbone
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Removing the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Adding new classification head
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

def create_single_view_loaders(splits, batch_size=32, num_workers=4, front_view_only=True):
    # Creating data loaders for single view training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    train_dataset = SingleViewCOIL100Dataset(
        splits['train'], transform=transform, front_view_only=front_view_only
    )
    val_dataset = SingleViewCOIL100Dataset(
        splits['val'], transform=transform, front_view_only=front_view_only
    )
    test_dataset = SingleViewCOIL100Dataset(
        splits['test'], transform=transform, front_view_only=front_view_only
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def train_single_view(model, train_loader, val_loader, device, epochs=8):
    # Training single view model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metrics = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch+1}/{epochs}') 
        # Training
        model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc='Training')):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        # Validating
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_acc'].append(accuracy)
        logger.info(f'Train Loss: {avg_train_loss:.4f}')
        logger.info(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.2f}%')
    return metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    # Creating directories
    save_dir = Path(args.save_dir) / 'single_view'
    save_dir.mkdir(parents=True, exist_ok=True)
    # Preparing data
    preprocessor = COIL100Preprocessor(args.data_dir)
    objects_dict = preprocessor.organize_views()
    splits = preprocessor.create_splits(objects_dict, seed=args.seed)
    # Creating data loaders
    train_loader, val_loader, test_loader = create_single_view_loaders(
        splits,
        batch_size=args.batch_size,
        front_view_only=args.front_view_only
    )
    # Creating model
    model = SingleViewModel(num_classes=100)
    model = model.to(device)
    # Training model
    metrics = train_single_view(
        model, train_loader, val_loader, device, epochs=args.epochs
    )
    # Saving results
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
    }, save_dir / 'model.pth')    
    logger.info(f'Training completed. Results saved to {save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Single View Model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--front_view_only', action='store_true',
                        help='Use only front view (first image) for each object')
    
    args = parser.parse_args()
    main(args)