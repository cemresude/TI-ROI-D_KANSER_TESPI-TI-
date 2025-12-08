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
        
        # root_dir zaten organized klasörünü gösteriyor
        # Benign klasörü
        benign_dir = os.path.join(root_dir, 'benign')
        if os.path.exists(benign_dir):
            for img_name in os.listdir(benign_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                    self.images.append(os.path.join(benign_dir, img_name))
                    self.labels.append(0)  # 0 = benign
        
        # Malignant klasörü (sadece only_benign=False ise)
        if not only_benign:
            malignant_dir = os.path.join(root_dir, 'malignant')
            if os.path.exists(malignant_dir):
                for img_name in os.listdir(malignant_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                        self.images.append(os.path.join(malignant_dir, img_name))
                        self.labels.append(1)  # 1 = malignant
    
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
    """Görüntüyü kare hale getirmek için siyah padding ekle"""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

def get_transforms(train=True):
    """
    ImageNet statistics for transfer learning compatibility
    Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    """
    if train:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(data_dir, batch_size=32, split_ratio=0.8, only_benign=False, use_weighted_sampler=False):
    """
    use_weighted_sampler=True: Sınıf dengesini otomatik düzelt
    """
    full_dataset = ThyroidDataset(
        data_dir, 
        transform=get_transforms(train=True),
        only_benign=only_benign
    )
    
    # Dataset kontrolü
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}. Please check:\n"
                        f"  1. DATA_DIR path is correct\n"
                        f"  2. 'benign' and/or 'malignant' folders exist\n"
                        f"  3. Folders contain image files (.png, .jpg, etc.)")
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    # Sınıf dağılımını kontrol et
    unique_labels = set(full_dataset.labels)
    if only_benign and len(unique_labels) > 1:
        print(f"WARNING: only_benign=True but found {len(unique_labels)} classes")
    
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Split resulted in empty set! Total={len(full_dataset)}, "
                        f"Train={train_size}, Val={val_size}. "
                        f"Try adjusting split_ratio or check dataset size.")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # WeightedRandomSampler for class balance
    train_sampler = None
    if use_weighted_sampler and not only_benign:
        # Train indices'ten labelları al
        train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
        
        # Eğer train_labels boşsa veya tek sınıf varsa sampler kullanma
        if len(train_labels) == 0:
            print("WARNING: No training samples found. Skipping WeightedRandomSampler.")
            use_weighted_sampler = False
        elif len(set(train_labels)) < 2:
            print(f"WARNING: Only one class found in training set: {set(train_labels)}. "
                  f"Skipping WeightedRandomSampler.")
            use_weighted_sampler = False
        else:
            try:
                class_counts = np.bincount(train_labels)
                
                # Sıfır olan sınıfları kontrol et
                if (class_counts == 0).any():
                    print(f"WARNING: Some classes have 0 samples: {class_counts}. "
                          f"Skipping WeightedRandomSampler.")
                    use_weighted_sampler = False
                else:
                    class_weights = 1.0 / class_counts
                    sample_weights = [class_weights[label] for label in train_labels]
                    
                    train_sampler = WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True
                    )
                    print(f"✓ WeightedRandomSampler enabled")
                    print(f"  Class counts: {class_counts}")
                    print(f"  Class weights: {class_weights}")
            except Exception as e:
                print(f"ERROR creating WeightedRandomSampler: {e}")
                print("Falling back to regular sampling.")
                use_weighted_sampler = False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler if use_weighted_sampler else None,
        shuffle=(train_sampler is None),
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)} (total samples: {train_size})")
    print(f"  Val batches: {len(val_loader)} (total samples: {val_size})")
    
    return train_loader, val_loader, train_size, val_size

def get_test_loader(data_dir, batch_size=32):
    """Test için hem benign hem malignant verileri yükle"""
    test_dataset = ThyroidDataset(
        data_dir,
        transform=get_transforms(train=False),
        only_benign=False  # Her iki sınıfı da yükle
    )
    
    # Debug: Kaç benign ve malignant örnek yüklendiğini kontrol et
    benign_count = sum(1 for label in test_dataset.labels if label == 0)
    malignant_count = sum(1 for label in test_dataset.labels if label == 1)
    print(f"Test set - Benign: {benign_count}, Malignant: {malignant_count}")
    
    if malignant_count == 0:
        print(f"WARNING: No malignant samples found in {data_dir}")
        print(f"Please check if malignant folder exists: {os.path.join(data_dir, 'malignant')}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader
