import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import cv2

from classifier import ThyroidResNetClassifier, GradCAM
from data_loader import get_test_loader
from config import Config

def denormalize_image(img_tensor):
    """ImageNet normalizasyonunu geri al"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1)

def visualize_predictions_with_gradcam(model, test_loader, device, num_samples=16, save_path=None):
    """
    Grad-CAM ile kanserli bölgeleri görselleştir
    
    4 satır x 4 sütun = 16 örnek
    - Satır 1: Orijinal görüntü
    - Satır 2: Grad-CAM heatmap
    - Satır 3: Overlay (orijinal + heatmap)
    - Satır 4: Prediction bilgisi
    """
    model.eval()
    gradcam = GradCAM(model)
    
    # Benign ve malignant örnekleri topla
    benign_samples = []
    malignant_samples = []
    
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            if lbl == 0 and len(benign_samples) < num_samples // 2:
                benign_samples.append((img, lbl))
            elif lbl == 1 and len(malignant_samples) < num_samples // 2:
                malignant_samples.append((img, lbl))
            
            if len(benign_samples) >= num_samples // 2 and len(malignant_samples) >= num_samples // 2:
                break
        
        if len(benign_samples) >= num_samples // 2 and len(malignant_samples) >= num_samples // 2:
            break
    
    samples = benign_samples + malignant_samples
    n_samples = len(samples)
    
    if n_samples == 0:
        print("No samples to visualize!")
        return
    
    # 4 satır: Orijinal, Grad-CAM, Overlay, Info
    fig, axes = plt.subplots(4, n_samples, figsize=(3*n_samples, 12))
    
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (img, true_label) in enumerate(samples):
        # Model prediction
        img_input = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_input)
            prob = torch.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
            confidence = prob[0, pred_label].item()
        
        # Grad-CAM
        cam, target_class = gradcam.generate_cam(img_input, target_class=pred_label)
        
        # Denormalize orijinal görüntü
        img_denorm = denormalize_image(img)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Overlay oluştur
        overlay = gradcam.generate_heatmap_overlay(img_denorm, cam, alpha=0.4)
        
        # 1. Satır: Orijinal
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(
            f"True: {'BENIGN' if true_label == 0 else 'MALIGNANT'}",
            color='green' if true_label == 0 else 'red',
            fontweight='bold',
            fontsize=10
        )
        axes[0, i].axis('off')
        
        # 2. Satır: Grad-CAM Heatmap
        axes[1, i].imshow(cam, cmap='jet')
        axes[1, i].axis('off')
        
        # 3. Satır: Overlay
        axes[2, i].imshow(overlay)
        axes[2, i].axis('off')
        
        # 4. Satır: Prediction Info
        pred_text = f"Pred: {'BENIGN' if pred_label == 0 else 'MALIGNANT'}\n"
        pred_text += f"Conf: {confidence:.2%}\n"
        pred_text += f"{'✓ CORRECT' if pred_label == true_label else '✗ WRONG'}"
        
        axes[3, i].text(
            0.5, 0.5, pred_text,
            ha='center', va='center',
            transform=axes[3, i].transAxes,
            fontsize=9,
            fontweight='bold',
            color='green' if pred_label == true_label else 'red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        )
        axes[3, i].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Original\nImage', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Grad-CAM\nHeatmap', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
    axes[2, 0].set_ylabel('Overlay\n(Cancer Region)', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
    axes[3, 0].set_ylabel('Prediction\nInfo', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
    
    plt.suptitle('CNN Cancer Detection with Grad-CAM Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.close()

def plot_roc_curve(labels, probs, save_path=None):
    """ROC Curve çiz"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('CNN Classifier ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved: {save_path}")
    
    plt.close()

def plot_confusion_matrix(cm, save_path=None):
    """Confusion Matrix görselleştir"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Benign', 'Malignant']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    plt.close()

def main():
    print("="*70)
    print("CNN CLASSIFIER - FINAL TEST & VISUALIZATION")
    print("="*70)
    
    print(f"\nUsing Model Path: {Config.MODEL_SAVE_PATH}")
    print(f"Data Path: {Config.DATA_DIR}")
    print(f"Device: {Config.DEVICE}")
    
    # Load test data
    test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    model = ThyroidResNetClassifier(num_classes=2, backbone='resnet18', pretrained=False).to(Config.DEVICE)
    
    checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_classifier.pth')
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(Config.DEVICE))
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Print results
    print("\n" + "="*70)
    print("CNN-ONLY RESULTS (threshold=0.5):")
    print("="*70)
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(f"\nROC-AUC: {roc_auc:.4f}")
    except ValueError:
        print("\nROC-AUC: Could not be calculated")
        roc_auc = None
    
    # Per-class metrics
    print("\n" + "="*70)
    print("PER-CLASS ANALYSIS:")
    print("="*70)
    
    benign_mask = all_labels == 0
    malignant_mask = all_labels == 1
    
    benign_accuracy = np.sum((all_preds == 0) & benign_mask) / np.sum(benign_mask)
    malignant_accuracy = np.sum((all_preds == 1) & malignant_mask) / np.sum(malignant_mask)
    
    print(f"Benign Recall: {benign_accuracy:.2%}")
    print(f"Malignant Recall: {malignant_accuracy:.2%}")
    print(f"Overall Accuracy: {np.sum(all_preds == all_labels) / len(all_labels):.2%}")
    
    # False positives/negatives
    false_positives = np.sum((all_preds == 1) & (all_labels == 0))
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1))
    
    print(f"\nFalse Positives (Benign → Malignant): {false_positives}")
    print(f"False Negatives (Malignant → Benign): {false_negatives} ⚠️")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS...")
    print("="*70)
    
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # 1. Grad-CAM visualization
    print("\n1. Generating Grad-CAM visualizations...")
    visualize_predictions_with_gradcam(
        model, test_loader, Config.DEVICE,
        num_samples=16,
        save_path=os.path.join(Config.MODEL_SAVE_PATH, 'cnn_gradcam_visualization.png')
    )
    
    # 2. ROC curve
    if roc_auc is not None:
        print("\n2. Plotting ROC curve...")
        plot_roc_curve(
            all_labels, all_probs,
            save_path=os.path.join(Config.MODEL_SAVE_PATH, 'cnn_roc_curve.png')
        )
    
    # 3. Confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(
        cm,
        save_path=os.path.join(Config.MODEL_SAVE_PATH, 'cnn_confusion_matrix.png')
    )
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAll results saved in: {Config.MODEL_SAVE_PATH}")
    print("\nGenerated files:")
    print("  - cnn_gradcam_visualization.png (Grad-CAM heatmaps)")
    print("  - cnn_roc_curve.png (ROC curve)")
    print("  - cnn_confusion_matrix.png (Confusion matrix)")

if __name__ == '__main__':
    main()
