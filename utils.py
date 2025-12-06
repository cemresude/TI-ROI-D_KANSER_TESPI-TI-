import torch
import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix, classification_report

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model, optimizer, epoch, val_loss, save_path, is_vae=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    filename = 'best_model_vae.pth' if is_vae else 'best_model.pth'
    filepath = os.path.join(save_path, filename)
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, checkpoint_path, is_vae=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('val_loss', 0)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'])
    return cm, report
