import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

from model import ConvVAE
from classifier import ThyroidResNetClassifier, GradCAM
from data_loader import get_test_loader
from config import Config
from test import compute_reconstruction_error, find_optimal_threshold

def denormalize_image(image_tensor):
    """ImageNet normalizasyonunu geri al"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return image_tensor * std + mean

def hybrid_predict(vae_model, cnn_model, image, vae_threshold, device):
    """
    Hybrid prediction: VAE (anomaly) + CNN (classification)
    
    Returns:
        vae_score: Reconstruction error
        cnn_pred: CNN prediction (0=benign, 1=malignant)
        cnn_conf: CNN confidence
        final_pred: Hybrid prediction
    """
    image = image.unsqueeze(0).to(device)
    
    # VAE anomaly score
    with torch.no_grad():
        recon, mu, logvar = vae_model(image)
        vae_score = torch.mean((image - recon) ** 2).item()
    
    # CNN prediction
    with torch.no_grad():
        cnn_output = cnn_model(image)
        cnn_probs = torch.softmax(cnn_output, dim=1)
        cnn_pred = cnn_output.argmax(dim=1).item()
        cnn_conf = cnn_probs[0, cnn_pred].item()
    
    # Hybrid decision
    vae_anomaly = (vae_score > vae_threshold)
    
    # Eğer VAE normal diyor ama CNN malignant diyor -> CNN'e güven
    # Eğer VAE anomaly diyor ve CNN da malignant diyor -> kesinlikle malignant
    if vae_anomaly and cnn_pred == 1:
        final_pred = 1  # High confidence malignant
    elif not vae_anomaly and cnn_pred == 0:
        final_pred = 0  # High confidence benign
    else:
        # Çelişkili durum: CNN'e öncelik ver ama confidence düşük
        final_pred = cnn_pred
    
    return vae_score, cnn_pred, cnn_conf, final_pred

def visualize_hybrid_results(vae_model, cnn_model, test_loader, vae_threshold, device, num_samples=8):
    """
    5 satır görselleştirme:
    1. Orijinal
    2. VAE Reconstruction
    3. VAE Anomaly Map
    4. CNN Grad-CAM
    5. Hybrid Decision
    """
    vae_model.eval()
    cnn_model.eval()
    gradcam = GradCAM(cnn_model)
    
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
    
    fig, axes = plt.subplots(5, len(samples), figsize=(20, 15))
    
    if len(samples) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (img, true_label) in enumerate(samples):
        # Hybrid prediction
        vae_score, cnn_pred, cnn_conf, final_pred = hybrid_predict(
            vae_model, cnn_model, img, vae_threshold, device
        )
        
        # Denormalize
        img_denorm = denormalize_image(img)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # 1. Orijinal
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(
            f"True: {'BENIGN' if true_label == 0 else 'MALIGNANT'}",
            color='green' if true_label == 0 else 'red',
            fontweight='bold'
        )
        axes[0, i].axis('off')
        
        # 2. VAE Reconstruction
        with torch.no_grad():
            recon, _, _ = vae_model(img.unsqueeze(0).to(device))
        recon_denorm = denormalize_image(recon.squeeze(0).cpu())
        axes[1, i].imshow(recon_denorm.permute(1, 2, 0).numpy())
        axes[1, i].text(0.5, -0.1, f'Error: {vae_score:.4f}',
                       ha='center', transform=axes[1, i].transAxes,
                       fontsize=9)
        axes[1, i].axis('off')
        
        # 3. VAE Anomaly Map
        diff = torch.mean((img - recon.squeeze(0).cpu()) ** 2, dim=0).numpy()
        im = axes[2, i].imshow(diff, cmap='hot')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
        axes[2, i].axis('off')
        
        # 4. CNN Grad-CAM
        cam, pred_class = gradcam.generate_cam(
            img.unsqueeze(0).to(device),
            target_class=cnn_pred
        )
        overlay = gradcam.generate_heatmap_overlay(img_denorm, cam)
        axes[3, i].imshow(overlay)
        axes[3, i].text(0.5, -0.1, f'CNN: {cnn_conf:.2f}',
                       ha='center', transform=axes[3, i].transAxes,
                       fontsize=9)
        axes[3, i].axis('off')
        
        # 5. Hybrid Decision
        decision_color = 'green' if final_pred == 0 else 'red'
        correct = (final_pred == true_label)
        axes[4, i].imshow(img_np, alpha=0.3)
        axes[4, i].text(0.5, 0.5,
                       f"{'BENIGN' if final_pred == 0 else 'MALIGNANT'}\n"
                       f"{'✓ CORRECT' if correct else '✗ WRONG'}",
                       ha='center', va='center',
                       transform=axes[4, i].transAxes,
                       fontsize=12, fontweight='bold',
                       color=decision_color,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[4, i].axis('off')
    
    axes[0, 0].set_ylabel('Orijinal\nGörüntü', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('VAE\nÇıktısı', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('VAE\nAnomali', fontsize=12, fontweight='bold')
    axes[3, 0].set_ylabel('CNN\nGrad-CAM', fontsize=12, fontweight='bold')
    axes[4, 0].set_ylabel('Hybrid\nKarar', fontsize=12, fontweight='bold')
    
    plt.suptitle('Hybrid System: VAE + CNN Classifier', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'hybrid_results.png'), dpi=150, bbox_inches='tight')
    print(f"Hybrid visualization saved!")
    plt.close()

def main():
    print("="*60)
    print("HYBRID SYSTEM TEST: VAE + CNN Classifier")
    print("="*60)
    
    # Load test data
    test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)
    
    # Load VAE
    vae_model = ConvVAE(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    vae_checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_vae.pth')
    vae_model.load_state_dict(torch.load(vae_checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    print("✓ VAE model loaded")
    
    # Load CNN Classifier
    cnn_model = ThyroidResNetClassifier(
        num_classes=2,
        backbone=Config.CLASSIFIER_BACKBONE,
        pretrained=False
    ).to(Config.DEVICE)
    cnn_checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_classifier.pth')
    cnn_model.load_state_dict(torch.load(cnn_checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    print("✓ CNN Classifier loaded")
    
    # VAE threshold
    print("\nComputing VAE optimal threshold...")
    vae_errors, labels = compute_reconstruction_error(vae_model, test_loader, Config.DEVICE)
    vae_threshold, fpr, tpr, roc_auc = find_optimal_threshold(vae_errors, labels)
    print(f"VAE Optimal Threshold: {vae_threshold:.6f}")
    print(f"VAE ROC AUC: {roc_auc:.4f}")
    
    # Hybrid predictions
    print("\nGenerating hybrid predictions...")
    all_true = []
    all_hybrid_pred = []
    all_cnn_pred = []
    all_vae_anomaly = []
    
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            vae_score, cnn_pred, cnn_conf, final_pred = hybrid_predict(
                vae_model, cnn_model, img, vae_threshold, Config.DEVICE
            )
            
            all_true.append(lbl.item())
            all_hybrid_pred.append(final_pred)
            all_cnn_pred.append(cnn_pred)
            all_vae_anomaly.append(int(vae_score > vae_threshold))
    
    all_true = np.array(all_true)
    all_hybrid_pred = np.array(all_hybrid_pred)
    all_cnn_pred = np.array(all_cnn_pred)
    all_vae_anomaly = np.array(all_vae_anomaly)
    
    # Results
    print("\n" + "="*60)
    print("VAE ONLY RESULTS:")
    print("="*60)
    print(classification_report(all_true, all_vae_anomaly, target_names=['Benign', 'Malignant']))
    
    print("\n" + "="*60)
    print("CNN ONLY RESULTS:")
    print("="*60)
    print(classification_report(all_true, all_cnn_pred, target_names=['Benign', 'Malignant']))
    
    print("\n" + "="*60)
    print("HYBRID SYSTEM RESULTS:")
    print("="*60)
    print(classification_report(all_true, all_hybrid_pred, target_names=['Benign', 'Malignant']))
    
    # Confusion matrices
    print("\nConfusion Matrix (Hybrid):")
    print(confusion_matrix(all_true, all_hybrid_pred))
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_hybrid_results(vae_model, cnn_model, test_loader, vae_threshold, Config.DEVICE, num_samples=8)
    
    print("\n✓ Hybrid test completed!")

if __name__ == '__main__':
    main()
