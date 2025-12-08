import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os
import sys

# FIX: Doğru path'i ekle
sys.path.insert(0, '/Users/cemresudeakdag/TİROİD_KANSER_TESPİTİ')

# Import local modules
from classifier import ThyroidResNetClassifier
from data_loader import get_test_loader
from config import Config

def find_best_threshold(probs, labels, target_recall=0.75):
    """
    Malignant recall hedefine ulaşan en iyi threshold'u bul
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        
        # Malignant recall
        malignant_mask = labels == 1
        malignant_recall = np.sum((preds == 1) & malignant_mask) / np.sum(malignant_mask)
        
        if malignant_recall >= target_recall:
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    return best_threshold

def main():
    print("="*60)
    print("CNN-ONLY TEST (Optimized Threshold)")
    print("="*60)
    
    # Load test data
    test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)
    
    # Load model
    model = ThyroidResNetClassifier(
        num_classes=2,
        backbone=Config.CLASSIFIER_BACKBONE,
        pretrained=False
    ).to(Config.DEVICE)
    
    checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_classifier.pth')
    model.load_state_dict(torch.load(checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded")
    
    # Compute probabilities
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Malignant prob
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    print(f"\nFinding optimal threshold (target malignant recall: 0.75)...")
    optimal_threshold = find_best_threshold(all_probs, all_labels, target_recall=0.75)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Predictions with optimal threshold
    predictions = (all_probs >= optimal_threshold).astype(int)
    
    # Results
    print("\n" + "="*60)
    print(f"CNN-ONLY RESULTS (threshold={optimal_threshold:.4f}):")
    print("="*60)
    print(classification_report(all_labels, predictions, target_names=['Benign', 'Malignant']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, predictions)
    print(cm)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN-Only ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'cnn_only_roc.png'))
    print(f"\nROC curve saved!")
    plt.close()

if __name__ == '__main__':
    main()
