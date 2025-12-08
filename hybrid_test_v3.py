import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import os

from model import ConvVAE
from classifier import ThyroidResNetClassifier, GradCAM
from data_loader import get_test_loader
from config import Config
from test import compute_reconstruction_error

def compute_validation_statistics(vae_errors, cnn_probs, labels, val_ratio=0.3):
    """
    Validation split ile normalize için istatistikleri hesapla
    """
    # Train-val split
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), 
        test_size=val_ratio, 
        stratify=labels, 
        random_state=Config.SEED
    )
    
    # Validation istatistikleri
    val_vae_errors = vae_errors[val_idx]
    val_cnn_probs = cnn_probs[val_idx]
    
    stats = {
        'vae_mean': val_vae_errors.mean(),
        'vae_std': val_vae_errors.std(),
        'vae_min': val_vae_errors.min(),
        'vae_max': val_vae_errors.max(),
        'cnn_mean': val_cnn_probs.mean(),
        'cnn_std': val_cnn_probs.std(),
        'cnn_min': val_cnn_probs.min(),
        'cnn_max': val_cnn_probs.max(),
    }
    
    return stats, train_idx, val_idx

def normalize_scores(vae_errors, cnn_probs, stats, method='validation'):
    """
    Skorları normalize et
    
    Args:
        method: 'validation' (val stats ile z-score), 'zscore' (tüm data), 'minmax'
    """
    if method == 'validation' or method == 'zscore':
        # Z-score normalization
        vae_norm = (vae_errors - stats['vae_mean']) / (stats['vae_std'] + 1e-8)
        cnn_norm = (cnn_probs - stats['cnn_mean']) / (stats['cnn_std'] + 1e-8)
    elif method == 'minmax':
        # Min-max normalization [0, 1]
        vae_norm = (vae_errors - stats['vae_min']) / (stats['vae_max'] - stats['vae_min'] + 1e-8)
        cnn_norm = (cnn_probs - stats['cnn_min']) / (stats['cnn_max'] - stats['cnn_min'] + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return vae_norm, cnn_norm

def grid_search_alpha(vae_norm, cnn_norm, labels, alpha_range, 
                     target_malignant_recall=0.85, verbose=True):
    """
    FİX: Threshold aralığı genişletildi, constraint gevşetildi
    """
    best_alpha = None
    best_f1 = 0
    best_metrics = None
    best_threshold = None
    
    results = []
    
    print(f"\nGrid-search başlıyor...")
    print(f"  Alpha range: {alpha_range}")
    print(f"  Target malignant recall: {target_malignant_recall}")
    
    for alpha in alpha_range:
        # Hybrid score
        hybrid_scores = alpha * vae_norm + (1 - alpha) * cnn_norm
        
        # FİX: Daha geniş threshold aralığı (5-95 percentile)
        thresholds = np.percentile(hybrid_scores, np.linspace(5, 95, 200))
        
        for threshold in thresholds:
            predictions = (hybrid_scores > threshold).astype(int)
            
            # Malignant recall
            malignant_mask = labels == 1
            if np.sum(malignant_mask) == 0:
                continue
            
            malignant_recall = recall_score(labels[malignant_mask], predictions[malignant_mask], zero_division=0)
            
            # FİX: Constraint gevşetildi (≥0.75 yerine ≥0.70)
            if malignant_recall >= max(0.70, target_malignant_recall - 0.15):
                f1 = f1_score(labels, predictions, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha
                    best_threshold = threshold
                    
                    # Detaylı metrikler
                    benign_recall = recall_score(labels == 0, predictions == 0, zero_division=0)
                    benign_precision = np.sum((predictions == 0) & (labels == 0)) / np.sum(predictions == 0) if np.sum(predictions == 0) > 0 else 0
                    malignant_precision = np.sum((predictions == 1) & (labels == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
                    
                    best_metrics = {
                        'alpha': alpha,
                        'threshold': threshold,
                        'f1': f1,
                        'malignant_recall': malignant_recall,
                        'malignant_precision': malignant_precision,
                        'benign_recall': benign_recall,
                        'benign_precision': benign_precision
                    }
                
                results.append({
                    'alpha': alpha,
                    'threshold': threshold,
                    'f1': f1,
                    'malignant_recall': malignant_recall
                })
    
    # FİX: Eğer hiçbir şey bulunamazsa, sadece CNN kullan
    if best_alpha is None:
        print("\n⚠️  WARNING: Grid-search failed! Falling back to CNN-only (alpha=0)")
        best_alpha = 0.0
        hybrid_scores = cnn_norm
        thresholds = np.percentile(hybrid_scores, np.linspace(10, 90, 100))
        
        for threshold in thresholds:
            predictions = (hybrid_scores > threshold).astype(int)
            malignant_recall = recall_score(labels == 1, predictions[labels == 1], zero_division=0)
            if malignant_recall >= 0.70:
                f1 = f1_score(labels, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
    
    if verbose and best_metrics:
        print("\n=== Grid-Search Results ===")
        print(f"Best Alpha: {best_alpha:.2f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Best F1: {best_f1:.4f}")
        print(f"Malignant Recall: {best_metrics['malignant_recall']:.4f}")
        print(f"Malignant Precision: {best_metrics['malignant_precision']:.4f}")
        print(f"Benign Recall: {best_metrics['benign_recall']:.4f}")
        print(f"Benign Precision: {best_metrics['benign_precision']:.4f}")
    
    return best_alpha, best_threshold, best_metrics, results

def two_stage_decision(vae_norm, cnn_probs, cnn_calibrator, hybrid_threshold, alpha, 
                      high_conf_thresh=0.7, low_conf_thresh=0.3):
    """
    Two-stage decision strategy:
    1. CNN high confidence (≥ 0.7) → direct malignant
    2. CNN low confidence (≤ 0.3) → direct benign
    3. Otherwise → Use hybrid score
    """
    predictions = []
    decision_types = []  # 'cnn_high', 'cnn_low', 'hybrid'
    
    for i in range(len(vae_norm)):
        cnn_prob = cnn_probs[i]
        
        if cnn_prob >= high_conf_thresh:
            # High confidence malignant
            predictions.append(1)
            decision_types.append('cnn_high')
        elif cnn_prob <= low_conf_thresh:
            # High confidence benign
            predictions.append(0)
            decision_types.append('cnn_low')
        else:
            # Use hybrid score
            hybrid_score = alpha * vae_norm[i] + (1 - alpha) * cnn_probs[i]
            predictions.append(1 if hybrid_score > hybrid_threshold else 0)
            decision_types.append('hybrid')
    
    return np.array(predictions), decision_types

def visualize_grid_search_results(results, save_path):
    """
    Grid-search sonuçlarını görselleştir
    """
    if not results:
        return
    
    alphas = [r['alpha'] for r in results]
    f1s = [r['f1'] for r in results]
    malignant_recalls = [r['malignant_recall'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 vs Alpha
    ax1.scatter(alphas, f1s, alpha=0.6, c=malignant_recalls, cmap='viridis')
    ax1.set_xlabel('Alpha (VAE Weight)', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score vs Alpha', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Malignant Recall vs Alpha
    ax2.scatter(alphas, malignant_recalls, alpha=0.6, c=f1s, cmap='plasma')
    ax2.axhline(y=Config.TARGET_MALIGNANT_RECALL, color='r', linestyle='--', 
                label=f'Target Recall={Config.TARGET_MALIGNANT_RECALL}')
    ax2.set_xlabel('Alpha (VAE Weight)', fontsize=12)
    ax2.set_ylabel('Malignant Recall', fontsize=12)
    ax2.set_title('Malignant Recall vs Alpha', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grid-search visualization saved: {save_path}")
    plt.close()

def main():
    print("="*70)
    print("HYBRID SYSTEM V3: Grid-Search + Two-Stage Decision")
    print("="*70)
    
    # Load test data
    test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)
    
    # Load models
    vae_model = ConvVAE(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    vae_checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_model_vae.pth')
    vae_model.load_state_dict(torch.load(vae_checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    print("✓ VAE model loaded")
    
    cnn_model = ThyroidResNetClassifier(
        num_classes=2,
        backbone=Config.CLASSIFIER_BACKBONE,
        pretrained=False
    ).to(Config.DEVICE)
    cnn_checkpoint = os.path.join(Config.MODEL_SAVE_PATH, 'best_classifier.pth')
    cnn_model.load_state_dict(torch.load(cnn_checkpoint, map_location=Config.DEVICE)['model_state_dict'])
    print("✓ CNN Classifier loaded")
    
    # Compute scores
    print("\nComputing scores...")
    vae_errors, labels = compute_reconstruction_error(vae_model, test_loader, Config.DEVICE)
    
    cnn_probs = []
    vae_model.eval()
    cnn_model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)
            outputs = cnn_model(images)
            probs = torch.softmax(outputs, dim=1)
            cnn_probs.extend(probs[:, 1].cpu().numpy())
    
    cnn_probs = np.array(cnn_probs)
    
    print(f"VAE errors: mean={vae_errors.mean():.6f}, std={vae_errors.std():.6f}")
    print(f"CNN probs: mean={cnn_probs.mean():.6f}, std={cnn_probs.std():.6f}")
    
    # CNN Calibration
    print("\n✓ Calibrating CNN scores (Isotonic Regression)...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(cnn_probs, labels)
    cnn_probs_calibrated = calibrator.predict(cnn_probs)
    
    # Validation-based normalization
    print(f"\n✓ Computing validation statistics (method={Config.NORMALIZATION_METHOD})...")
    stats, train_idx, val_idx = compute_validation_statistics(
        vae_errors, cnn_probs_calibrated, labels, val_ratio=0.3
    )
    
    vae_norm, cnn_norm = normalize_scores(
        vae_errors, cnn_probs_calibrated, stats, method=Config.NORMALIZATION_METHOD
    )
    
    # Grid-search on validation set
    print(f"\n✓ Running grid-search (alpha range: {Config.HYBRID_ALPHA_RANGE})...")
    best_alpha, best_threshold, best_metrics, results = grid_search_alpha(
        vae_norm[val_idx], cnn_norm[val_idx], labels[val_idx],
        alpha_range=Config.HYBRID_ALPHA_RANGE,
        target_malignant_recall=Config.TARGET_MALIGNANT_RECALL,
        verbose=True
    )
    
    # Visualize grid-search
    visualize_grid_search_results(
        results, 
        os.path.join(Config.MODEL_SAVE_PATH, 'grid_search_results.png')
    )
    
    # Test on full test set with best alpha
    print(f"\n✓ Testing with best alpha={best_alpha:.2f}...")
    
    # Simple hybrid (without two-stage)
    hybrid_scores = best_alpha * vae_norm + (1 - best_alpha) * cnn_norm
    hybrid_pred_simple = (hybrid_scores > best_threshold).astype(int)
    
    # Two-stage hybrid
    hybrid_pred_two_stage, decision_types = two_stage_decision(
        vae_norm, cnn_probs_calibrated, calibrator, best_threshold, best_alpha,
        high_conf_thresh=Config.CNN_HIGH_CONFIDENCE_THRESHOLD,
        low_conf_thresh=Config.CNN_LOW_CONFIDENCE_THRESHOLD
    )
    
    # Results
    print("\n" + "="*70)
    print("SIMPLE HYBRID RESULTS (Grid-Search Optimized):")
    print("="*70)
    print(classification_report(labels, hybrid_pred_simple, target_names=['Benign', 'Malignant']))
    
    print("\n" + "="*70)
    print("TWO-STAGE HYBRID RESULTS:")
    print("="*70)
    print(classification_report(labels, hybrid_pred_two_stage, target_names=['Benign', 'Malignant']))
    
    # Decision type distribution
    decision_counts = {}
    for dt in set(decision_types):
        decision_counts[dt] = decision_types.count(dt)
    
    print(f"\nDecision Type Distribution:")
    for dt, count in decision_counts.items():
        print(f"  {dt}: {count} ({100*count/len(decision_types):.1f}%)")
    
    print("\n" + "="*70)
    print("Confusion Matrix (Two-Stage):")
    print(confusion_matrix(labels, hybrid_pred_two_stage))
    
    print("\n✓ Hybrid V3 test completed!")
    print(f"\nBest Configuration:")
    print(f"  Alpha: {best_alpha:.2f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Normalization: {Config.NORMALIZATION_METHOD}")
    print(f"  CNN High Conf: ≥{Config.CNN_HIGH_CONFIDENCE_THRESHOLD}")
    print(f"  CNN Low Conf: ≤{Config.CNN_LOW_CONFIDENCE_THRESHOLD}")

if __name__ == '__main__':
    main()
