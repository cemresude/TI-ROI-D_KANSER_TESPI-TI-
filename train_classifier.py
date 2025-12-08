import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np

from classifier import ThyroidResNetClassifier
from data_loader import get_dataloaders
from config import Config
from utils import set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Loss must be scalar for backward
        if not loss.requires_grad:
            continue
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)  # Multiply by batch size
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)  # Divide by total samples
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)  # Multiply by batch size
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)  # Divide by total samples
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    set_seed(Config.SEED)
    
    print(f"Device: {Config.DEVICE}")
    print(f"Training CNN Classifier with {Config.CLASSIFIER_BACKBONE}")
    
    print(f"\nChecking DATA_DIR: {Config.DATA_DIR}")
    if not os.path.exists(Config.DATA_DIR):
        print("ERROR: DATA_DIR does not exist!")
        print("Please run organize_data.py first or update Config.DATA_DIR")
        return
    
    benign_path = os.path.join(Config.DATA_DIR, 'benign')
    malignant_path = os.path.join(Config.DATA_DIR, 'malignant')
    
    print(f"  Benign folder exists: {os.path.exists(benign_path)}")
    print(f"  Malignant folder exists: {os.path.exists(malignant_path)}")
    
    if os.path.exists(benign_path):
        benign_files = [f for f in os.listdir(benign_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        print(f"  Benign images: {len(benign_files)}")
    else:
        print("  Benign folder not found!")
    
    if os.path.exists(malignant_path):
        malignant_files = [f for f in os.listdir(malignant_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        print(f"  Malignant images: {len(malignant_files)}")
    else:
        print("  Malignant folder not found!")
    
    print("\nLoading BALANCED data (BENIGN + MALIGNANT)...")
    
    try:
        train_loader, val_loader, train_size, val_size = get_dataloaders(
            Config.DATA_DIR, 
            Config.BATCH_SIZE, 
            Config.SPLIT_RATIO,
            only_benign=False,
            use_weighted_sampler=True
        )
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        print("\nPossible solutions:")
        print("1. Run organize_data.py first")
        print("2. Update Config.DATA_DIR to point to organized folder")
        print("3. Check if images exist in benign and malignant folders")
        return
    
    print("\nData loaded successfully!")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    model = ThyroidResNetClassifier(
        num_classes=2,
        backbone=Config.CLASSIFIER_BACKBONE,
        pretrained=True
    ).to(Config.DEVICE)
    
    print("\nComputing class weights...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights[0] *= 1.2
    class_weights = class_weights / class_weights.sum()
    
    print(f"Class weights: Benign={class_weights[0]:.3f}, Malignant={class_weights[1]:.3f}")
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(Config.DEVICE))
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.CLASSIFIER_LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    warmup_epochs = Config.WARMUP_EPOCHS
    total_steps = len(train_loader) * Config.CLASSIFIER_EPOCHS
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"\nTraining with Cosine Annealing + Warmup ({warmup_epochs} epochs)")
    print("Using Mixed Precision Training (AMP)\n")
    
    global_step = 0
    
    for epoch in range(Config.CLASSIFIER_EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.CLASSIFIER_EPOCHS}')
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, Config.DEVICE
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        print(f'Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}')
        print(f'Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}')
        
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1
        
        if val_f1 > best_val_f1 + Config.EARLY_STOPPING_DELTA:
            best_val_f1 = val_f1
            patience_counter = 0
            
            checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_classifier.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f'Model saved! Best Val F1: {best_val_f1:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}')
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
    
    print("\nTraining completed!")
    print(f"Best Validation F1 Score: {best_val_f1:.4f}")

if __name__ == '__main__':
    main()
