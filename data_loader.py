import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

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
    if train:
        return transforms.Compose([
            SquarePad(),  # Önce kare yap
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),  # 0-1 aralığına normalize eder
        ])
    else:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

def get_dataloaders(data_dir, batch_size=32, split_ratio=0.8, only_benign=False):
    """
    only_benign=True ise sadece benign verileri yükler (VAE eğitimi için)
    """
    full_dataset = ThyroidDataset(
        data_dir, 
        transform=get_transforms(train=True),
        only_benign=only_benign
    )
    
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
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
        num_workers=4
    )
    
    return test_loader
