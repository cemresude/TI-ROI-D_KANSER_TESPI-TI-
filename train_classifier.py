import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

from classifier import ThyroidResNetClassifier
from data_loader import get_dataloaders
from config import Config
from utils import set_seed, save_checkpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    avg_loss = total_loss / len(train_loader)
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
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    set_seed(Config.SEED)
    
    print(f"Device: {Config.DEVICE}")
    print(f"Training CNN Classifier with {Config.CLASSIFIER_BACKBONE}")
    
    # Hem benign hem malignant veriler (sınıf dengeli)
    print("Loading BALANCED data (BENIGN + MALIGNANT)...")
    train_loader, val_loader, train_size, val_size = get_dataloaders(
        Config.DATA_DIR, 
        Config.BATCH_SIZE, 
        Config.SPLIT_RATIO,
        only_benign=False,  # Her iki sınıf
        use_weighted_sampler=True  # Sınıf dengeleme
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Model
    model = ThyroidResNetClassifier(
        num_classes=2,
        backbone=Config.CLASSIFIER_BACKBONE,
        pretrained=True
    ).to(Config.DEVICE)
    
    # Loss (class imbalance için weighted)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.CLASSIFIER_LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Cosine annealing with warmup
    warmup_epochs = Config.WARMUP_EPOCHS
    total_steps = len(train_loader) * Config.CLASSIFIER_EPOCHS
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Checkpoint dir
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"\nTraining with Cosine Annealing + Warmup ({warmup_epochs} epochs)")
    print(f"Using Mixed Precision Training (AMP)\n")
    
    for epoch in range(Config.CLASSIFIER_EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.CLASSIFIER_EPOCHS}')
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, Config.DEVICE
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        print(f'Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | '
              f'Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}')
        print(f'Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | '
              f'Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}')
        
        scheduler.step()
        
        # Save best model (based on F1 score)
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
            print(f'✓ Model saved! Best Val F1: {best_val_f1:.4f}')
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
