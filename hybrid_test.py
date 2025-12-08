import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
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

def z_score_normalize(scores):
    """Z-score normalizasyon"""
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return scores
    return (scores - mean) / std

def calibrate_cnn_scores(cnn_probs, labels):
    """
    CNN çıktılarını kalibre et (Isotonic Regression)
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(cnn_probs, labels)
    calibrated_probs = calibrator.predict(cnn_probs)
    return calibrated_probs, calibrator

def find_benign_optimized_threshold(vae_errors, cnn_probs, labels, alpha=0.75, target_benign_recall=0.95):
    """
    Benign recall'ı optimize eden hybrid threshold bulma
    """
    # Z-score normalize
    vae_errors_norm = z_score_normalize(vae_errors)
    cnn_probs_norm = z_score_normalize(cnn_probs)
    
    # Hybrid score: alpha * VAE + (1-alpha) * CNN
    hybrid_scores = alpha * vae_errors_norm + (1 - alpha) * cnn_probs_norm
    
    # Threshold aralığı
    thresholds = np.percentile(hybrid_scores, np.linspace(0, 100, 1000))
    
    best_threshold = None
    best_f1 = 0
    best_metrics = None
    
    for threshold in thresholds:
        predictions = (hybrid_scores > threshold).astype(int)
        
        # Benign (0) ve Malignant (1) için metrikleri hesapla
        benign_mask = labels == 0
        malignant_mask = labels == 1
        
        # Benign recall (True Negative Rate)
        benign_correct = np.sum((predictions == 0) & benign_mask)
        benign_recall = benign_correct / np.sum(benign_mask) if np.sum(benign_mask) > 0 else 0
        
        # Benign recall hedefine ulaşıldıysa F1'i hesapla
        if benign_recall >= target_benign_recall:
            # Malignant recall
            malignant_correct = np.sum((predictions == 1) & malignant_mask)
            malignant_recall = malignant_correct / np.sum(malignant_mask) if np.sum(malignant_mask) > 0 else 0
            
            # Precision
            benign_precision = benign_correct / np.sum(predictions == 0) if np.sum(predictions == 0) > 0 else 0
            malignant_precision = malignant_correct / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
            
            # F1
            benign_f1 = 2 * (benign_precision * benign_recall) / (benign_precision + benign_recall) if (benign_precision + benign_recall) > 0 else 0
            malignant_f1 = 2 * (malignant_precision * malignant_recall) / (malignant_precision + malignant_recall) if (malignant_precision + malignant_recall) > 0 else 0
            
            macro_f1 = (benign_f1 + malignant_f1) / 2
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'benign_recall': benign_recall,
                    'benign_precision': benign_precision,
                    'benign_f1': benign_f1,
                    'malignant_recall': malignant_recall,
                    'malignant_precision': malignant_precision,
                    'malignant_f1': malignant_f1,
                    'macro_f1': macro_f1
                }
    
    if best_threshold is None:
        print(f"WARNING: Target benign recall {target_benign_recall} not reached. Using fallback.")
        # Fallback
        predictions = (hybrid_scores > np.median(hybrid_scores)).astype(int)
        best_threshold = np.median(hybrid_scores)
        best_metrics = {'threshold': best_threshold}
    
    return best_threshold, best_metrics, hybrid_scores

def hybrid_predict_v2(vae_model, cnn_model, image, vae_stats, cnn_calibrator, hybrid_threshold, alpha, device):
    """
    İyileştirilmiş hybrid prediction
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
        cnn_prob_raw = cnn_probs[0, 1].cpu().numpy()
    
    # Calibrate CNN
    if cnn_calibrator is not None:
        cnn_prob = cnn_calibrator.predict(np.array([cnn_prob_raw]))[0]
    else:
        cnn_prob = cnn_prob_raw
    
    # Z-score normalize
    vae_score_norm = (vae_score - vae_stats['mean']) / vae_stats['std'] if vae_stats['std'] > 0 else vae_score
    cnn_prob_norm = (cnn_prob - vae_stats['cnn_mean']) / vae_stats['cnn_std'] if vae_stats['cnn_std'] > 0 else cnn_prob
    
    # Hybrid score: alpha * VAE + (1-alpha) * CNN
    hybrid_score = alpha * vae_score_norm + (1 - alpha) * cnn_prob_norm
    
    # Final prediction
    final_pred = 1 if hybrid_score > hybrid_threshold else 0
    
    return vae_score, cnn_prob, hybrid_score, final_pred

def visualize_hybrid_results(vae_model, cnn_model, test_loader, vae_stats, cnn_calibrator, 
                             hybrid_threshold, alpha, device, num_samples=8):
    """
    5 satır görselleştirme
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
        vae_score, cnn_prob, hybrid_score, final_pred = hybrid_predict_v2(
            vae_model, cnn_model, img, vae_stats, cnn_calibrator, hybrid_threshold, alpha, device
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
        axes[1, i].text(0.5, -0.1, f'VAE: {vae_score:.4f}',
                       ha='center', transform=axes[1, i].transAxes, fontsize=9)
        axes[1, i].axis('off')
        
        # 3. VAE Anomaly Map
        diff = torch.mean((img - recon.squeeze(0).cpu()) ** 2, dim=0).numpy()
        im = axes[2, i].imshow(diff, cmap='hot')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
        axes[2, i].axis('off')
        
        # 4. CNN Grad-CAM
        cam, pred_class = gradcam.generate_cam(
            img.unsqueeze(0).to(device),
            target_class=1 if final_pred == 1 else None
        )
        overlay = gradcam.generate_heatmap_overlay(img_denorm, cam)
        axes[3, i].imshow(overlay)
        axes[3, i].text(0.5, -0.1, f'CNN: {cnn_prob:.3f}',
                       ha='center', transform=axes[3, i].transAxes, fontsize=9)
        axes[3, i].axis('off')
        
        # 5. Hybrid Decision
        decision_color = 'green' if final_pred == 0 else 'red'
        correct = (final_pred == true_label)
        axes[4, i].imshow(img_np, alpha=0.3)
        axes[4, i].text(0.5, 0.5,
                       f"{'BENIGN' if final_pred == 0 else 'MALIGNANT'}\n"
                       f"{'CORRECT' if correct else 'WRONG'}\n"
                       f"Score: {hybrid_score:.3f}",
                       ha='center', va='center',
                       transform=axes[4, i].transAxes,
                       fontsize=10, fontweight='bold',
                       color=decision_color,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[4, i].axis('off')
    
    axes[0, 0].set_ylabel('Orijinal', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('VAE Cikti', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('VAE Anomali', fontsize=12, fontweight='bold')
    axes[3, 0].set_ylabel('CNN GradCAM', fontsize=12, fontweight='bold')
    axes[4, 0].set_ylabel('Hybrid Karar', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Hybrid System V2 (alpha={alpha}, target_benign_recall={Config.TARGET_BENIGN_RECALL})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'hybrid_results_v2.png'), dpi=150, bbox_inches='tight')
    print(f"Hybrid visualization saved!")
    plt.close()

def main():
    print("="*60)
    print("HYBRID SYSTEM V2 TEST: VAE + CNN Classifier")
    print("="*60)
    
    # Load test data
    test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)
    
    # Load VAE
    vae_model = ConvVAE(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    vae_checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_vae.pth')
    vae_model.load_state_dict(torch.load(vae_checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    print("VAE model loaded")
    
    # Load CNN Classifier
    cnn_model = ThyroidResNetClassifier(
        num_classes=2,
        backbone=Config.CLASSIFIER_BACKBONE,
        pretrained=False
    ).to(Config.DEVICE)
    cnn_checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_classifier.pth')
    cnn_model.load_state_dict(torch.load(cnn_checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    print("CNN Classifier loaded")
    
    # Compute VAE errors and CNN probs
    print("\nComputing scores...")
    vae_errors = []
    cnn_probs = []
    labels = []
    
    vae_model.eval()
    cnn_model.eval()
    
    with torch.no_grad():
        for images, batch_labels in test_loader:
            images = images.to(Config.DEVICE)
            
            # VAE
            recon, _, _ = vae_model(images)
            batch_errors = torch.mean((images - recon) ** 2, dim=[1, 2, 3])
            vae_errors.extend(batch_errors.cpu().numpy())
            
            # CNN
            outputs = cnn_model(images)
            probs = torch.softmax(outputs, dim=1)
            cnn_probs.extend(probs[:, 1].cpu().numpy())
            
            labels.extend(batch_labels.numpy())
    
    vae_errors = np.array(vae_errors)
    cnn_probs = np.array(cnn_probs)
    labels = np.array(labels)
    
    print(f"VAE errors: mean={vae_errors.mean():.6f}, std={vae_errors.std():.6f}")
    print(f"CNN probs: mean={cnn_probs.mean():.6f}, std={cnn_probs.std():.6f}")
    
    # CNN Calibration
    print("\nCalibrating CNN scores (Isotonic Regression)...")
    cnn_probs_calibrated, cnn_calibrator = calibrate_cnn_scores(cnn_probs, labels)
    print(f"Calibrated CNN probs: mean={cnn_probs_calibrated.mean():.6f}, std={cnn_probs_calibrated.std():.6f}")
    
    # Find benign-optimized threshold
    print(f"\nFinding benign-optimized threshold (alpha={Config.HYBRID_ALPHA}, target_recall={Config.TARGET_BENIGN_RECALL})...")
    hybrid_threshold, metrics, hybrid_scores = find_benign_optimized_threshold(
        vae_errors, cnn_probs_calibrated, labels,
        alpha=Config.HYBRID_ALPHA,
        target_benign_recall=Config.TARGET_BENIGN_RECALL
    )
    
    print(f"\nOptimal Hybrid Threshold: {hybrid_threshold:.6f}")
    if metrics is not None and 'benign_recall' in metrics:
        print(f"Benign Recall: {metrics['benign_recall']:.3f}")
        print(f"Benign Precision: {metrics['benign_precision']:.3f}")
        print(f"Benign F1: {metrics['benign_f1']:.3f}")
        print(f"Malignant Recall: {metrics['malignant_recall']:.3f}")
        print(f"Malignant Precision: {metrics['malignant_precision']:.3f}")
        print(f"Malignant F1: {metrics['malignant_f1']:.3f}")
        print(f"Macro F1: {metrics['macro_f1']:.3f}")
    
    # VAE stats for prediction
    vae_stats = {
        'mean': vae_errors.mean(),
        'std': vae_errors.std(),
        'cnn_mean': cnn_probs_calibrated.mean(),
        'cnn_std': cnn_probs_calibrated.std()
    }
    
    # Hybrid predictions
    print("\nGenerating hybrid predictions...")
    all_hybrid_pred = (hybrid_scores > hybrid_threshold).astype(int)
    
    # VAE only
    vae_threshold_simple, _, _, _ = find_optimal_threshold(vae_errors, labels)
    all_vae_pred = (vae_errors > vae_threshold_simple).astype(int)
    
    # CNN only
    all_cnn_pred = (cnn_probs_calibrated > 0.5).astype(int)
    
    # Results
    print("\n" + "="*60)
    print("VAE ONLY RESULTS:")
    print("="*60)
    print(classification_report(labels, all_vae_pred, target_names=['Benign', 'Malignant']))
    
    print("\n" + "="*60)
    print("CNN ONLY RESULTS (Calibrated):")
    print("="*60)
    print(classification_report(labels, all_cnn_pred, target_names=['Benign', 'Malignant']))
    
    print("\n" + "="*60)
    print(f"HYBRID SYSTEM V2 RESULTS (alpha={Config.HYBRID_ALPHA}):")
    print("="*60)
    print(classification_report(labels, all_hybrid_pred, target_names=['Benign', 'Malignant']))
    
    print("\nConfusion Matrix (Hybrid V2):")
    print(confusion_matrix(labels, all_hybrid_pred))
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_hybrid_results(vae_model, cnn_model, test_loader, vae_stats, cnn_calibrator,
                             hybrid_threshold, Config.HYBRID_ALPHA, Config.DEVICE, num_samples=8)
    
    print("\nHybrid V2 test completed!")

if __name__ == '__main__':
    main()
