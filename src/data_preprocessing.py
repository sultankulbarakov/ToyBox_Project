# src/data_preprocessing.py
import os
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import logging
import kagglehub
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import json
import glob

# setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class COIL100Preprocessor:
    # preprocessor for COIL-100 dataset handling multi-view image organization
    def __init__(self, base_dir: str):
        # initializing preprocessor
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'coil-100'
        self.processed_dir = self.base_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # creating splits directory
        self.splits_dir = self.processed_dir / 'splits'
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> None:
        # downloading COIL-100 dataset using kagglehub
        try:
            logger.info("Downloading COIL-100 dataset...")
            dataset_path = kagglehub.dataset_download('jessicali9530/coil100')
            # ensuring data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)
            # finding all PNG files in the downloaded dataset
            png_files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
            if not png_files:
                raise Exception("No PNG files found in the downloaded dataset")
            # copying all PNG files to our data directory
            for png_file in png_files:
                shutil.copy2(png_file, self.data_dir)
            logger.info(f"Copied {len(png_files)} images to {self.data_dir}")
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
            
    def organize_views(self) -> Dict[str, List[str]]:
        # organizing images by object and their views
        objects = {}
        # list all PNG files in the data directory
        png_files = list(self.data_dir.glob('*.png'))
        if not png_files:
            raise Exception(f"No PNG files found in {self.data_dir}")
        logger.info(f"Found {len(png_files)} images")
        # COIL-100 images are named as 'objX__Y.png' where X is object ID and Y is angle
        for img_path in png_files:
            if not img_path.name.startswith('obj'):
                continue   
            obj_id = img_path.stem.split('__')[0] 
            if obj_id not in objects:
                objects[obj_id] = []
            objects[obj_id].append(str(img_path))  
        # sorting images by angle for each object
        for obj_id in objects:
            objects[obj_id].sort()  
        logger.info(f"Organized {len(objects)} objects")
        return objects
        
    def create_splits(self, 
                     objects_dict: Dict[str, List[str]],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     seed: int = 42) -> Dict[str, Dict[str, List[str]]]:
        # create train/validation/test splits.
        random.seed(seed)
        objects = list(objects_dict.keys())
        if not objects:
            raise Exception("No objects found in dataset")
        random.shuffle(objects)
        # calculating split sizes
        n_objects = len(objects)
        n_train = int(n_objects * train_ratio)
        n_val = int(n_objects * val_ratio)
        # splitting objects
        train_objects = objects[:n_train]
        val_objects = objects[n_train:n_train + n_val]
        test_objects = objects[n_train + n_val:]
        # creating split dictionaries
        splits = {
            'train': {obj: objects_dict[obj] for obj in train_objects},
            'val': {obj: objects_dict[obj] for obj in val_objects},
            'test': {obj: objects_dict[obj] for obj in test_objects}
        }
        # saving splits to disk
        split_info = {
            'train': train_objects,
            'val': val_objects,
            'test': test_objects
        }
        with open(self.splits_dir / 'splits.json', 'w') as f:
            json.dump(split_info, f)  
        logger.info(f"Created splits: train={len(train_objects)}, val={len(val_objects)}, test={len(test_objects)} objects")     
        return splits

class COIL100Dataset(Dataset):
    # dataset class for COIL-100 with multi-view support
    def __init__(self, 
                 data_dict: Dict[str, List[str]], 
                 views_per_sample: int = 3,
                 transform=None):
        self.data_dict = data_dict
        self.objects = list(data_dict.keys())
        self.views_per_sample = views_per_sample
        self.transform = transform
        # creating object to label mapping
        self.obj_to_label = {obj: idx for idx, obj in enumerate(self.objects)}
 
    def __len__(self) -> int:
        return len(self.objects)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        obj_id = self.objects[idx]
        views = self.data_dict[obj_id]
        # randomly sample views
        selected_views = random.sample(views, min(self.views_per_sample, len(views)))
        # loading and transforming images
        images = []
        for view_path in selected_views:
            img = Image.open(view_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        label = self.obj_to_label[obj_id]
        return images, label

def create_data_loaders(
    splits: Dict[str, Dict[str, List[str]]],
    batch_size: int = 32,
    views_per_sample: int = 3,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # defining transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    # creating datasets
    train_dataset = COIL100Dataset(
        data_dict=splits['train'],
        views_per_sample=views_per_sample,
        transform=transform
    )
    val_dataset = COIL100Dataset(
        data_dict=splits['val'],
        views_per_sample=views_per_sample,
        transform=transform
    )
    test_dataset = COIL100Dataset(
        data_dict=splits['test'],
        views_per_sample=views_per_sample,
        transform=transform
    )
    # creating data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process COIL-100 dataset')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory for dataset and processed files')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loaders')
    parser.add_argument('--views_per_sample', type=int, default=3,
                       help='Number of views to use per sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    try:
        # initializing preprocessor
        preprocessor = COIL100Preprocessor(args.base_dir)
        # downloading dataset if needed
        preprocessor.download_dataset()
        # organizing views
        logger.info("Organizing views...")
        objects_dict = preprocessor.organize_views()
        # creating splits
        logger.info("Creating data splits...")
        splits = preprocessor.create_splits(objects_dict, seed=args.seed)
        # creating data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            splits,
            batch_size=args.batch_size,
            views_per_sample=args.views_per_sample
        )
        logger.info("Dataset processing complete!")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise