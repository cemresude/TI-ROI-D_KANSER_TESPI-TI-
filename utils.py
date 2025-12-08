import torch
import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix, classification_report

def set_seed(seed=42):
    """
    Reproducibility için seed ayarla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, val_loss, save_path, is_vae=False):
    """
    Model checkpoint kaydet
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        val_loss: Validation loss
        save_path: Save directory
        is_vae: VAE mi CNN mi?
    """
    os.makedirs(save_path, exist_ok=True)
    
    if is_vae:
        filename = 'best_model_vae.pth'
    else:
        filename = 'best_model_cnn.pth'
    
    filepath = os.path.join(save_path, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, filepath)
    
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, checkpoint_path, is_vae=False):
    """
    Model checkpoint yükle
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (None olabilir)
        checkpoint_path: Checkpoint file path
        is_vae: VAE mi CNN mi?
    
    Returns:
        epoch, val_loss
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Checkpoint loaded from epoch {epoch} (val_loss: {val_loss:.4f})")
    
    return epoch, val_loss

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'])
    return cm, report

def count_parameters(model):
    """
    Model parametrelerini say
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    
    return total, trainable

def print_model_summary(model, input_size=(1, 3, 128, 128)):
    """
    Model özetini yazdır
    """
    try:
        from torchsummary import summary
        summary(model, input_size[1:])
    except ImportError:
        print("torchsummary not installed. Skipping summary.")
        print("Install with: pip install torchsummary")
