import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class ThyroidDataset(Dataset):
    def __init__(self, root_dir, transform=None, only_benign=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        benign_dir = os.path.join(root_dir, 'benign')
        if os.path.exists(benign_dir):
            for img_name in os.listdir(benign_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(benign_dir, img_name))
                    self.labels.append(0)
        
        if not only_benign:
            malignant_dir = os.path.join(root_dir, 'malignant')
            if os.path.exists(malignant_dir):
                for img_name in os.listdir(malignant_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(malignant_dir, img_name))
                        self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((224, 224)),
            
            # GÜÇLENDİRİLMİŞ AUGMENTASYON (PIL Image üzerinde)
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.2, 0.2),
                scale=(0.85, 1.15)
            ),
            
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
            
            # ToTensor ÖNCE (PIL → Tensor)
            transforms.ToTensor(),
            
            # Normalize
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # RandomErasing SON (Tensor üzerinde)
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir, batch_size=32, split_ratio=0.8, only_benign=False, use_weighted_sampler=False):
    """
    Args:
        data_dir: Veri dizini
        batch_size: Batch boyutu
        split_ratio: Train/val split oranı
        only_benign: Sadece benign veriler (VAE için)
        use_weighted_sampler: Sınıf dengeleme (CNN için)
    
    Returns:
        train_loader, val_loader, train_size, val_size
    """
    full_dataset = ThyroidDataset(data_dir, transform=get_transforms(train=True), only_benign=only_benign)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Empty split: Total={len(full_dataset)}, Train={train_size}, Val={val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_sampler = None
    if use_weighted_sampler and not only_benign:
        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
        
        if len(train_labels) > 0 and len(set(train_labels)) >= 2:
            class_counts = np.bincount(train_labels)
            if not (class_counts == 0).any():
                class_weights = 1.0 / class_counts
                sample_weights = [class_weights[label] for label in train_labels]
                train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
                print(f"WeightedRandomSampler enabled: {class_counts}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"DataLoaders: Train={train_size}, Val={val_size}")
    
    return train_loader, val_loader, train_size, val_size

def get_test_loader(data_dir, batch_size=32):
    test_dataset = ThyroidDataset(data_dir, transform=get_transforms(train=False), only_benign=False)
    
    benign_count = sum(1 for label in test_dataset.labels if label == 0)
    malignant_count = sum(1 for label in test_dataset.labels if label == 1)
    print(f"Test set - Benign: {benign_count}, Malignant: {malignant_count}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_loader
