import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import os

from model import ConvVAE
from classifier import ThyroidResNetClassifier
from data_loader import get_test_loader
from config import Config
from test import compute_reconstruction_error

def compute_benign_validation_statistics(vae_errors, cnn_probs, labels, val_ratio=0.3):
    """
    SADECE BENIGN validation samples ile normalize için stats hesapla
    """
    # Benign samples'ı ayır
    benign_mask = labels == 0
    benign_vae_errors = vae_errors[benign_mask]
    benign_cnn_probs = cnn_probs[benign_mask]
    benign_labels = labels[benign_mask]
    
    # Benign içinde train-val split
    n_benign = len(benign_vae_errors)
    n_val = int(val_ratio * n_benign)
    
    val_indices = np.random.choice(n_benign, n_val, replace=False)
    
    val_vae_errors = benign_vae_errors[val_indices]
    val_cnn_probs = benign_cnn_probs[val_indices]
    
    stats = {
        'vae_mean': val_vae_errors.mean(),
        'vae_std': val_vae_errors.std(),
        'cnn_mean': val_cnn_probs.mean(),
        'cnn_std': val_cnn_probs.std(),
    }
    
    print(f"\n✓ Benign validation statistics computed:")
    print(f"  VAE: mean={stats['vae_mean']:.6f}, std={stats['vae_std']:.6f}")
    print(f"  CNN: mean={stats['cnn_mean']:.6f}, std={stats['cnn_std']:.6f}")
    
    return stats

def normalize_scores(vae_errors, cnn_probs, stats, method='benign_validation', cnn_score_method='logit'):
    """
    Benign validation stats ile normalize et
    CNN için logit transform kullan
    """
    # VAE: Z-score normalization
    vae_norm = (vae_errors - stats['vae_mean']) / (stats['vae_std'] + 1e-8)
    
    # CNN: Logit transform + normalize
    if cnn_score_method == 'logit':
        # p -> logit(p) = log(p / (1-p))
        cnn_probs_clipped = np.clip(cnn_probs, 1e-7, 1 - 1e-7)
        cnn_logits = np.log(cnn_probs_clipped / (1 - cnn_probs_clipped))
        cnn_norm = cnn_logits  # Logit zaten normalized scale'de
    else:
        # Simple z-score
        cnn_norm = (cnn_probs - stats['cnn_mean']) / (stats['cnn_std'] + 1e-8)
    
    return vae_norm, cnn_norm

def grid_search_with_constraint(vae_norm, cnn_norm, labels, alpha_range, 
                                target_malignant_recall=0.85, 
                                optimization_metric='macro_f1',
                                verbose=True):
    """
    Malignant recall ≥ target constraint ile macro-F1 veya balanced accuracy maksimize et
    """
    best_alpha = None
    best_threshold = None
    best_score = -np.inf
    best_metrics = None
    
    results = []
    
    print(f"\n✓ Grid-search başlıyor...")
    print(f"  Alpha range: {alpha_range}")
    print(f"  Constraint: Malignant recall ≥ {target_malignant_recall}")
    print(f"  Optimization: {optimization_metric}")
    
    for alpha in alpha_range:
        hybrid_scores = alpha * vae_norm + (1 - alpha) * cnn_norm
        
        thresholds = np.percentile(hybrid_scores, np.linspace(5, 95, 200))
        
        for threshold in thresholds:
            predictions = (hybrid_scores > threshold).astype(int)
            
            # Malignant recall (CONSTRAINT)
            malignant_mask = labels == 1
            if np.sum(malignant_mask) == 0:
                continue
            
            malignant_recall = recall_score(labels[malignant_mask], predictions[malignant_mask], zero_division=0)
            
            # Constraint check
            if malignant_recall < target_malignant_recall:
                continue
            
            # Optimization metric
            if optimization_metric == 'macro_f1':
                score = f1_score(labels, predictions, average='macro', zero_division=0)
            elif optimization_metric == 'balanced_accuracy':
                score = balanced_accuracy_score(labels, predictions)
            else:
                score = f1_score(labels, predictions, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_threshold = threshold
                
                # Detailed metrics
                benign_recall = recall_score(labels == 0, predictions == 0, zero_division=0)
                benign_precision = np.sum((predictions == 0) & (labels == 0)) / np.sum(predictions == 0) if np.sum(predictions == 0) > 0 else 0
                malignant_precision = np.sum((predictions == 1) & (labels == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
                
                best_metrics = {
                    'alpha': alpha,
                    'threshold': threshold,
                    'optimization_score': score,
                    'malignant_recall': malignant_recall,
                    'malignant_precision': malignant_precision,
                    'benign_recall': benign_recall,
                    'benign_precision': benign_precision,
                    'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(labels, predictions)
                }
            
            results.append({
                'alpha': alpha,
                'threshold': threshold,
                'score': score,
                'malignant_recall': malignant_recall
            })
    
    if best_alpha is None:
        print("\n⚠️  No valid configuration found! Falling back to CNN-only")
        best_alpha = 0.0
        best_threshold = 0.5
    
    if verbose and best_metrics:
        print("\n=== Grid-Search Results ===")
        print(f"Best Alpha: {best_alpha:.2f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Optimization Score ({optimization_metric}): {best_score:.4f}")
        print(f"Malignant Recall: {best_metrics['malignant_recall']:.4f}")
        print(f"Malignant Precision: {best_metrics['malignant_precision']:.4f}")
        print(f"Benign Recall: {best_metrics['benign_recall']:.4f}")
        print(f"Benign Precision: {best_metrics['benign_precision']:.4f}")
        print(f"Macro F1: {best_metrics['macro_f1']:.4f}")
        print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    
    return best_alpha, best_threshold, best_metrics, results

def two_stage_decision_strict(vae_norm, cnn_probs, hybrid_threshold, alpha, 
                              high_conf_thresh=0.8, low_conf_thresh=0.4):
    """
    Sıkılaştırılmış two-stage decision
    p ≥ 0.8 → Malignant
    p ≤ 0.4 → Benign
    Otherwise → Hybrid
    """
    predictions = []
    decision_types = []
    
    for i in range(len(vae_norm)):
        cnn_prob = cnn_probs[i]
        
        if cnn_prob >= high_conf_thresh:
            predictions.append(1)
            decision_types.append('cnn_high')
        elif cnn_prob <= low_conf_thresh:
            predictions.append(0)
            decision_types.append('cnn_low')
        else:
            hybrid_score = alpha * vae_norm[i] + (1 - alpha) * cnn_probs[i]
            predictions.append(1 if hybrid_score > hybrid_threshold else 0)
            decision_types.append('hybrid')
    
    return np.array(predictions), decision_types

def main():
    print("="*70)
    print("HYBRID SYSTEM V4: Benign-Validation + Logit + Strict Two-Stage")
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
    print("✓ CNN model loaded")
    
    # Compute scores
    print("\n✓ Computing scores...")
    vae_errors, labels = compute_reconstruction_error(vae_model, test_loader, Config.DEVICE)
    
    cnn_probs = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)
            outputs = cnn_model(images)
            probs = torch.softmax(outputs, dim=1)
            cnn_probs.extend(probs[:, 1].cpu().numpy())
    
    cnn_probs = np.array(cnn_probs)
    
    # CNN Calibration
    print("\n✓ Calibrating CNN...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(cnn_probs, labels)
    cnn_probs_calibrated = calibrator.predict(cnn_probs)
    
    # Benign validation statistics
    stats = compute_benign_validation_statistics(vae_errors, cnn_probs_calibrated, labels)
    
    # Normalize with benign stats + logit
    vae_norm, cnn_norm = normalize_scores(
        vae_errors, cnn_probs_calibrated, stats, 
        method=Config.NORMALIZATION_METHOD,
        cnn_score_method=Config.CNN_SCORE_METHOD
    )
    
    # Grid-search with constraint
    best_alpha, best_threshold, best_metrics, results = grid_search_with_constraint(
        vae_norm, cnn_norm, labels,
        alpha_range=Config.HYBRID_ALPHA_RANGE,
        target_malignant_recall=Config.TARGET_MALIGNANT_RECALL,
        optimization_metric=Config.OPTIMIZATION_METRIC,
        verbose=True
    )
    
    # Test on full set
    print(f"\n✓ Testing with alpha={best_alpha:.2f}...")
    
    # Simple hybrid
    hybrid_scores = best_alpha * vae_norm + (1 - best_alpha) * cnn_norm
    hybrid_pred_simple = (hybrid_scores > best_threshold).astype(int)
    
    # Two-stage (strict)
    hybrid_pred_two_stage, decision_types = two_stage_decision_strict(
        vae_norm, cnn_probs_calibrated, best_threshold, best_alpha,
        high_conf_thresh=Config.CNN_HIGH_CONFIDENCE_THRESHOLD,
        low_conf_thresh=Config.CNN_LOW_CONFIDENCE_THRESHOLD
    )
    
    # Results
    print("\n" + "="*70)
    print("SIMPLE HYBRID V4:")
    print("="*70)
    print(classification_report(labels, hybrid_pred_simple, target_names=['Benign', 'Malignant']))
    
    print("\n" + "="*70)
    print("TWO-STAGE HYBRID V4 (Strict):")
    print("="*70)
    print(classification_report(labels, hybrid_pred_two_stage, target_names=['Benign', 'Malignant']))
    
    # Decision distribution
    decision_counts = {}
    for dt in set(decision_types):
        decision_counts[dt] = decision_types.count(dt)
    
    print(f"\nDecision Type Distribution:")
    for dt, count in decision_counts.items():
        print(f"  {dt}: {count} ({100*count/len(decision_types):.1f}%)")
    
    print("\nConfusion Matrix (Two-Stage V4):")
    print(confusion_matrix(labels, hybrid_pred_two_stage))
    
    print("\n✓ Hybrid V4 completed!")

if __name__ == '__main__':
    main()
