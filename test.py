import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import cv2

from model import ConvVAE, vae_loss
from data_loader import get_test_loader
from config import Config
from utils import load_checkpoint

def compute_reconstruction_error(model, data_loader, device):
    """Her görüntü için reconstruction error hesapla"""
    model.eval()
    errors = []
    labels = []
    
    with torch.no_grad():
        for images, label in tqdm(data_loader, desc='Computing reconstruction errors'):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            
            # MSE per sample
            batch_errors = torch.mean((images - recon_images) ** 2, dim=[1, 2, 3])
            errors.extend(batch_errors.cpu().numpy())
            labels.extend(label.numpy())
    
    return np.array(errors), np.array(labels)

def find_optimal_threshold(errors, labels):
    """ROC curve kullanarak optimal threshold bul"""
    fpr, tpr, thresholds = roc_curve(labels, errors)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic ile optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, fpr, tpr, roc_auc

def detect_anomalous_regions(original, reconstructed, threshold_percentile=95, blur_kernel=5):
    """
    Normalize edilmiş MSE ve Gaussian blur kullanarak anomali bölgelerini tespit et
    
    Args:
        original: Orijinal görüntü tensor [C, H, W]
        reconstructed: Reconstruct edilmiş görüntü tensor [C, H, W]
        threshold_percentile: Anomali için percentile threshold (95 = üst %5)
        blur_kernel: Gaussian blur kernel boyutu
    
    Returns:
        anomaly_map: Normalize edilmiş anomali haritası
        binary_mask: İkili maske (anomali bölgeleri)
        overlay: Orijinal görüntü üzerine anomali bölgeleri
    """
    # MSE hesapla (channel bazında ortalama)
    mse_map = torch.mean((original - reconstructed) ** 2, dim=0).numpy()
    
    # Gaussian blur uygula (gürültüyü azalt)
    if blur_kernel > 0:
        mse_map = cv2.GaussianBlur(mse_map, (blur_kernel, blur_kernel), 0)
    
    # Normalize et [0, 1]
    mse_min = mse_map.min()
    mse_max = mse_map.max()
    if mse_max > mse_min:
        anomaly_map = (mse_map - mse_min) / (mse_max - mse_min)
    else:
        anomaly_map = mse_map
    
    # Threshold ile binary mask oluştur
    threshold = np.percentile(anomaly_map, threshold_percentile)
    binary_mask = (anomaly_map > threshold).astype(np.uint8)
    
    # Morfolojik işlemler (küçük gürültüleri temizle)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Orijinal görüntü üzerine overlay oluştur
    original_np = original.permute(1, 2, 0).numpy()
    overlay = original_np.copy()
    
    # Kırmızı renkte işaretle
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = binary_mask * 255  # Kırmızı kanal
    overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
    
    # Konturları çiz
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_with_contours = overlay.copy()
    cv2.drawContours(overlay_with_contours, contours, -1, (1, 0, 0), 2)
    
    return anomaly_map, binary_mask, overlay_with_contours

def visualize_results(model, data_loader, device, num_samples=8):
    """Orijinal, reconstruct edilmiş görüntüleri ve kanserli bölgeleri görselleştir"""
    model.eval()
    
    benign_images = []
    benign_recon = []
    malignant_images = []
    malignant_recon = []
    
    # Her iki sınıftan da örnek topla
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            recon_images, _, _ = model(images)
            
            # Benign örnekleri topla
            benign_mask = labels == 0
            if benign_mask.any() and len(benign_images) < num_samples // 2:
                benign_images.append(images[benign_mask].cpu())
                benign_recon.append(recon_images[benign_mask].cpu())
            
            # Malignant örnekleri topla
            malignant_mask = labels == 1
            if malignant_mask.any() and len(malignant_images) < num_samples // 2:
                malignant_images.append(images[malignant_mask].cpu())
                malignant_recon.append(recon_images[malignant_mask].cpu())
            
            # Yeterli örnek toplandıysa dur
            if len(benign_images) >= num_samples // 2 and len(malignant_images) >= num_samples // 2:
                break
    
    # Örnekleri birleştir
    if benign_images:
        benign_images = torch.cat(benign_images, dim=0)[:num_samples // 2]
        benign_recon = torch.cat(benign_recon, dim=0)[:num_samples // 2]
    else:
        print("WARNING: No benign samples found for visualization!")
        benign_images = torch.zeros(0, 3, 128, 128)
        benign_recon = torch.zeros(0, 3, 128, 128)
    
    if malignant_images:
        malignant_images = torch.cat(malignant_images, dim=0)[:num_samples // 2]
        malignant_recon = torch.cat(malignant_recon, dim=0)[:num_samples // 2]
    else:
        print("WARNING: No malignant samples found for visualization!")
        malignant_images = torch.zeros(0, 3, 128, 128)
        malignant_recon = torch.zeros(0, 3, 128, 128)
    
    # Benign ve malignant'ı yan yana koy
    images_list = torch.cat([benign_images, malignant_images], dim=0)
    recon_list = torch.cat([benign_recon, malignant_recon], dim=0)
    labels_list = torch.cat([
        torch.zeros(len(benign_images), dtype=torch.long),
        torch.ones(len(malignant_images), dtype=torch.long)
    ], dim=0)
    
    actual_samples = len(images_list)
    if actual_samples == 0:
        print("ERROR: No samples to visualize!")
        return
    
    # 3 satır: Orijinal, Reconstruct, Heatmap
    fig, axes = plt.subplots(3, actual_samples, figsize=(20, 8))
    
    # Eğer tek sütun varsa axes'i 2D array'e çevir
    if actual_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(actual_samples):
        # Original
        axes[0, i].imshow(images_list[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"{'BENIGN' if labels_list[i] == 0 else 'MALIGNANT'}", 
                             color='green' if labels_list[i] == 0 else 'red',
                             fontweight='bold')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon_list[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        
        # Difference Heatmap (Kanserli Bölge Tespiti)
        diff = torch.mean((images_list[i] - recon_list[i]) ** 2, dim=0).numpy()
        im = axes[2, i].imshow(diff, cmap='hot', interpolation='bilinear')
        axes[2, i].axis('off')
        
        # Her heatmap için colorbar ekle
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    axes[0, 0].set_ylabel('Orijinal', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Model Çıktısı', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('Tespit Edilen\nKanserli Bölge', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'reconstruction_comparison.png'), dpi=150)
    print(f"Visualization saved to {Config.MODEL_SAVE_PATH}/reconstruction_comparison.png")
    print(f"Visualized {len(benign_images)} benign and {len(malignant_images)} malignant samples")
    plt.close()

def visualize_anomaly_detection(model, data_loader, device, num_samples=8, threshold_percentile=95):
    """
    Gelişmiş anomali tespiti görselleştirmesi
    4 satır: Orijinal, Model Çıktısı, Anomali Haritası, İşaretli Görüntü
    """
    model.eval()
    
    benign_images = []
    benign_recon = []
    malignant_images = []
    malignant_recon = []
    
    # Her iki sınıftan da örnek topla
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            recon_images, _, _ = model(images)
            
            # Benign örnekleri topla
            benign_mask = labels == 0
            if benign_mask.any() and len(benign_images) < num_samples // 2:
                benign_images.append(images[benign_mask].cpu())
                benign_recon.append(recon_images[benign_mask].cpu())
            
            # Malignant örnekleri topla
            malignant_mask = labels == 1
            if malignant_mask.any() and len(malignant_images) < num_samples // 2:
                malignant_images.append(images[malignant_mask].cpu())
                malignant_recon.append(recon_images[malignant_mask].cpu())
            
            if len(benign_images) >= num_samples // 2 and len(malignant_images) >= num_samples // 2:
                break
    
    # Örnekleri birleştir
    if benign_images:
        benign_images = torch.cat(benign_images, dim=0)[:num_samples // 2]
        benign_recon = torch.cat(benign_recon, dim=0)[:num_samples // 2]
    else:
        print("WARNING: No benign samples found!")
        benign_images = torch.zeros(0, 3, 128, 128)
        benign_recon = torch.zeros(0, 3, 128, 128)
    
    if malignant_images:
        malignant_images = torch.cat(malignant_images, dim=0)[:num_samples // 2]
        malignant_recon = torch.cat(malignant_recon, dim=0)[:num_samples // 2]
    else:
        print("WARNING: No malignant samples found!")
        malignant_images = torch.zeros(0, 3, 128, 128)
        malignant_recon = torch.zeros(0, 3, 128, 128)
    
    images_list = torch.cat([benign_images, malignant_images], dim=0)
    recon_list = torch.cat([benign_recon, malignant_recon], dim=0)
    labels_list = torch.cat([
        torch.zeros(len(benign_images), dtype=torch.long),
        torch.ones(len(malignant_images), dtype=torch.long)
    ], dim=0)
    
    actual_samples = len(images_list)
    if actual_samples == 0:
        print("ERROR: No samples to visualize!")
        return
    
    # 4 satır: Orijinal, Reconstruct, Anomaly Map, Overlay
    fig, axes = plt.subplots(4, actual_samples, figsize=(20, 12))
    
    if actual_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(actual_samples):
        # Anomali tespiti
        anomaly_map, binary_mask, overlay = detect_anomalous_regions(
            images_list[i], recon_list[i], threshold_percentile=threshold_percentile
        )
        
        # 1. Satır: Orijinal
        axes[0, i].imshow(images_list[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"{'BENIGN' if labels_list[i] == 0 else 'MALIGNANT'}", 
                             color='green' if labels_list[i] == 0 else 'red',
                             fontweight='bold', fontsize=12)
        axes[0, i].axis('off')
        
        # 2. Satır: Model Çıktısı
        axes[1, i].imshow(recon_list[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        
        # 3. Satır: Normalize Anomaly Map (Gaussian blur uygulanmış)
        im = axes[2, i].imshow(anomaly_map, cmap='jet', interpolation='bilinear')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        # 4. Satır: İşaretli Görüntü (Konturlar ile)
        axes[3, i].imshow(np.clip(overlay, 0, 1))
        axes[3, i].axis('off')
        
        # Anomali oranını hesapla
        anomaly_ratio = (binary_mask.sum() / binary_mask.size) * 100
        axes[3, i].text(0.5, -0.1, f'Anomali: %{anomaly_ratio:.1f}',
                       ha='center', transform=axes[3, i].transAxes,
                       fontsize=9, color='red' if anomaly_ratio > 5 else 'green')
    
    axes[0, 0].set_ylabel('Orijinal\nGörüntü', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Model\nÇıktısı', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Anomali\nHaritası', fontsize=12, fontweight='bold')
    axes[3, 0].set_ylabel('Tespit Edilen\nKanserli Bölge', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Gelişmiş Anomali Tespiti (Threshold: %{threshold_percentile})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'advanced_anomaly_detection.png'), 
                dpi=150, bbox_inches='tight')
    print(f"Advanced visualization saved to {Config.MODEL_SAVE_PATH}/advanced_anomaly_detection.png")
    print(f"Visualized {len(benign_images)} benign and {len(malignant_images)} malignant samples")
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc):
    """ROC curve çiz"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'roc_curve.png'))
    print(f"ROC curve saved to {Config.MODEL_SAVE_PATH}/roc_curve.png")
    plt.close()

def main():
    print(f"Device: {Config.DEVICE}")
    
    # Test verilerini yükle (hem benign hem malignant)
    print("Loading test data (BENIGN + MALIGNANT)...")
    print(f"Data directory: {Config.DATA_DIR}")
    
    # Klasör yapısını kontrol et
    benign_path = os.path.join(Config.DATA_DIR, 'benign')
    malignant_path = os.path.join(Config.DATA_DIR, 'malignant')
    
    print(f"Checking directories:")
    print(f"  Benign path exists: {os.path.exists(benign_path)}")
    print(f"  Malignant path exists: {os.path.exists(malignant_path)}")
    
    if os.path.exists(benign_path):
        benign_files = [f for f in os.listdir(benign_path) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
        print(f"  Benign images found: {len(benign_files)}")
    
    if os.path.exists(malignant_path):
        malignant_files = [f for f in os.listdir(malignant_path) if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
        print(f"  Malignant images found: {len(malignant_files)}")
    
    test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Modeli yükle
    model = ConvVAE(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_vae.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    load_checkpoint(model, None, checkpoint_path, is_vae=True)
    print("Model loaded successfully!")
    
    # Reconstruction error hesapla
    print("\nComputing reconstruction errors...")
    errors, labels = compute_reconstruction_error(model, test_loader, Config.DEVICE)
    
    # İstatistikler
    benign_errors = errors[labels == 0]
    malignant_errors = errors[labels == 1]
    
    print(f"\nBENIGN - Mean Error: {benign_errors.mean():.6f} ± {benign_errors.std():.6f}")
    print(f"MALIGNANT - Mean Error: {malignant_errors.mean():.6f} ± {malignant_errors.std():.6f}")
    
    # Optimal threshold bul
    print("\nFinding optimal threshold...")
    optimal_threshold, fpr, tpr, roc_auc = find_optimal_threshold(errors, labels)
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Predictions
    predictions = (errors > optimal_threshold).astype(int)
    
    # Confusion matrix ve classification report
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions, 
                                target_names=['Benign', 'Malignant']))
    
    # Görselleştirmeler
    print("\nGenerating visualizations...")
    visualize_results(model, test_loader, Config.DEVICE)
    
    # Gelişmiş anomali tespiti görselleştirmesi
    print("\nGenerating advanced anomaly detection visualization...")
    visualize_anomaly_detection(model, test_loader, Config.DEVICE, 
                                num_samples=8, threshold_percentile=95)
    
    plot_roc_curve(fpr, tpr, roc_auc)
    
    # Error distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(benign_errors, bins=50, alpha=0.5, label='Benign', color='green')
    plt.hist(malignant_errors, bins=50, alpha=0.5, label='Malignant', color='red')
    plt.axvline(optimal_threshold, color='black', linestyle='--', 
                label=f'Threshold: {optimal_threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'error_distribution.png'))
    print(f"Error distribution saved to {Config.MODEL_SAVE_PATH}/error_distribution.png")
    plt.close()
    
    print("\nTest completed!")

if __name__ == '__main__':
    main()
