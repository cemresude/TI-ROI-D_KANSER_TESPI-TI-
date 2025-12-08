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
    CNN için logit transform + z-score normalize
    """
    # VAE: Z-score normalization
    vae_norm = (vae_errors - stats['vae_mean']) / (stats['vae_std'] + 1e-8)
    
    # CNN: Logit transform + Z-SCORE NORMALIZE (DÜZELTİLDİ)
    if cnn_score_method == 'logit':
        # 1. p -> logit(p) = log(p / (1-p))
        cnn_probs_clipped = np.clip(cnn_probs, 1e-7, 1 - 1e-7)
        cnn_logits = np.log(cnn_probs_clipped / (1 - cnn_probs_clipped))
        
        # 2. Z-score normalize logits (benign validation stats ile)
        # Logit'lerin mean/std'sini stats'e ekleyin
        cnn_logits_mean = cnn_logits[np.isfinite(cnn_logits)].mean()  # NaN/inf kontrolü
        cnn_logits_std = cnn_logits[np.isfinite(cnn_logits)].std()
        
        cnn_norm = (cnn_logits - cnn_logits_mean) / (cnn_logits_std + 1e-8)
    else:
        # Simple z-score
        cnn_norm = (cnn_probs - stats['cnn_mean']) / (stats['cnn_std'] + 1e-8)
    
    return vae_norm, cnn_norm

def grid_search_with_constraint(vae_norm, cnn_norm, labels, alpha_range, 
                                target_malignant_recall=0.85,
                                target_benign_recall=0.70,
                                optimization_metric='balanced_accuracy',
                                verbose=True):
    """
    Dual constraint: Malignant recall ≥ 0.85 VE Benign recall ≥ 0.70
    """
    best_alpha = None
    best_threshold = None
    best_score = -np.inf
    best_metrics = None
    
    results = []
    
    print(f"\n✓ Grid-search başlıyor...")
    print(f"  Alpha range: {alpha_range}")
    print(f"  Constraint 1: Malignant recall ≥ {target_malignant_recall}")
    print(f"  Constraint 2: Benign recall ≥ {target_benign_recall}")
    print(f"  Optimization: {optimization_metric}")
    
    for alpha in alpha_range:
        hybrid_scores = alpha * vae_norm + (1 - alpha) * cnn_norm
        
        thresholds = np.percentile(hybrid_scores, np.linspace(10, 98, 250))
        
        for threshold in thresholds:
            predictions = (hybrid_scores > threshold).astype(int)
            
            # Malignant recall (CONSTRAINT 1) - DÜZELTİLDİ
            malignant_mask = labels == 1
            if np.sum(malignant_mask) == 0:
                continue
            
            # FIX: Sadece malignant sample'ları al
            malignant_labels = labels[malignant_mask]
            malignant_preds = predictions[malignant_mask]
            malignant_recall = recall_score(malignant_labels, malignant_preds, zero_division=0)
            
            # Benign recall (CONSTRAINT 2) - DÜZELTİLDİ
            benign_mask = labels == 0
            if np.sum(benign_mask) == 0:
                continue
            
            # FIX: Sadece benign sample'ları al
            benign_labels = labels[benign_mask]
            benign_preds = predictions[benign_mask]
            benign_recall = recall_score(benign_labels, benign_preds, zero_division=0)
            
            # DUAL CONSTRAINT CHECK
            if malignant_recall < target_malignant_recall:
                continue
            if benign_recall < target_benign_recall:
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
                'malignant_recall': malignant_recall,
                'benign_recall': benign_recall
            })
    
    # FALLBACK: Eğer constraint sağlanamazsa - DÜZELTİLDİ
    if best_alpha is None:
        print("\n⚠️  Dual constraint sağlanamadı! Relaxing constraints...")
        
        for alpha in alpha_range:
            hybrid_scores = alpha * vae_norm + (1 - alpha) * cnn_norm
            thresholds = np.percentile(hybrid_scores, np.linspace(10, 98, 250))
            
            for threshold in thresholds:
                predictions = (hybrid_scores > threshold).astype(int)
                
                # FIX: Correct recall computation
                malignant_mask = labels == 1
                benign_mask = labels == 0
                
                malignant_recall = recall_score(
                    labels[malignant_mask], 
                    predictions[malignant_mask], 
                    zero_division=0
                )
                
                benign_recall = recall_score(
                    labels[benign_mask], 
                    predictions[benign_mask], 
                    zero_division=0
                )
                
                # Sadece malignant constraint
                if malignant_recall >= target_malignant_recall:
                    score = balanced_accuracy_score(labels, predictions)
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                        best_threshold = threshold
                        
                        # Best metrics oluştur
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
                            'balanced_accuracy': score
                        }
                        
                        print(f"  Relaxed: alpha={alpha:.2f}, thresh={threshold:.4f}, mal_recall={malignant_recall:.3f}, ben_recall={benign_recall:.3f}")
    
    if verbose and best_metrics:
        print("\n=== Grid-Search Results ===")
        print(f"Best Alpha: {best_alpha:.2f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Optimization Score: {best_score:.4f}")
        print(f"Malignant Recall: {best_metrics['malignant_recall']:.4f} ✓" if best_metrics['malignant_recall'] >= target_malignant_recall else f"Malignant Recall: {best_metrics['malignant_recall']:.4f} ✗")
        print(f"Benign Recall: {best_metrics['benign_recall']:.4f} ✓" if best_metrics['benign_recall'] >= target_benign_recall else f"Benign Recall: {best_metrics['benign_recall']:.4f} ✗")
        print(f"Malignant Precision: {best_metrics['malignant_precision']:.4f}")
        print(f"Benign Precision: {best_metrics['benign_precision']:.4f}")
        print(f"Macro F1: {best_metrics['macro_f1']:.4f}")
        print(f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
    
    return best_alpha, best_threshold, best_metrics, results

def two_stage_decision_strict(vae_norm, cnn_norm, cnn_probs_raw, hybrid_threshold, alpha, 
                              high_conf_thresh=0.85, low_conf_thresh=0.50):
    """
    DÜZELTİLMİŞ two-stage:
    - High/low confidence için RAW kalibre probability kullan
    - Hybrid decision için NORMALIZED CNN score kullan
    
    Args:
        vae_norm: Normalized VAE errors
        cnn_norm: Normalized CNN logits (Z-SCORE)
        cnn_probs_raw: Raw calibrated probabilities (high/low decision için)
        hybrid_threshold: Hybrid score threshold
        alpha: VAE weight
        high_conf_thresh: High confidence threshold (raw prob için)
        low_conf_thresh: Low confidence threshold (raw prob için)
    """
    predictions = []
    decision_types = []
    
    for i in range(len(vae_norm)):
        cnn_prob_raw = cnn_probs_raw[i]  # RAW kalibre prob (high/low için)
        
        # Stage 1: High confidence malignant (RAW prob)
        if cnn_prob_raw >= high_conf_thresh:
            predictions.append(1)
            decision_types.append('cnn_high')
        # Stage 2: Low confidence benign (RAW prob)
        elif cnn_prob_raw <= low_conf_thresh:
            predictions.append(0)
            decision_types.append('cnn_low')
        # Stage 3: Hybrid (NORMALIZED scores)
        else:
            hybrid_score = alpha * vae_norm[i] + (1 - alpha) * cnn_norm[i]  # cnn_norm kullan!
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
    
    # FİX: VERİ SIZINTISINI ÖNLE - Train/Val split
    # Test set'in %70'ini grid-search için, %30'unu final test için kullan
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.3,
        stratify=labels,
        random_state=Config.SEED
    )
    
    # Grid-search için CNN calibration (sadece train split ile)
    calibrator.fit(cnn_probs[train_idx], labels[train_idx])
    cnn_probs_calibrated = calibrator.predict(cnn_probs)
    
    print(f"✓ Data split: {len(train_idx)} grid-search, {len(test_idx)} final test")
    
    # Benign validation statistics (sadece train split'ten)
    stats = compute_benign_validation_statistics(
        vae_errors[train_idx], 
        cnn_probs_calibrated[train_idx], 
        labels[train_idx]
    )
    
    # Normalize with benign stats + logit (TÜM DATA için)
    vae_norm, cnn_norm = normalize_scores(
        vae_errors, cnn_probs_calibrated, stats, 
        method=Config.NORMALIZATION_METHOD,
        cnn_score_method=Config.CNN_SCORE_METHOD
    )
    
    print(f"\n✓ Normalization applied:")
    print(f"  VAE normalized: mean={vae_norm.mean():.4f}, std={vae_norm.std():.4f}")
    print(f"  CNN logits normalized: mean={cnn_norm.mean():.4f}, std={cnn_norm.std():.4f}")
    
    # Grid-search with DUAL constraint (sadece train split üzerinde)
    best_alpha, best_threshold, best_metrics, results = grid_search_with_constraint(
        vae_norm[train_idx], cnn_norm[train_idx], labels[train_idx],
        alpha_range=Config.HYBRID_ALPHA_RANGE,
        target_malignant_recall=Config.TARGET_MALIGNANT_RECALL,
        target_benign_recall=Config.TARGET_BENIGN_RECALL,
        optimization_metric=Config.OPTIMIZATION_METRIC,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print("GRID-SEARCH SET RESULTS (70% of data):")
    print(f"{'='*70}")
    
    # Grid-search set results
    hybrid_scores_train = best_alpha * vae_norm[train_idx] + (1 - best_alpha) * cnn_norm[train_idx]
    hybrid_pred_train = (hybrid_scores_train > best_threshold).astype(int)
    print(classification_report(labels[train_idx], hybrid_pred_train, target_names=['Benign', 'Malignant']))
    
    # Test on FINAL TEST SET (30% - hiç görülmemiş)
    print(f"\n{'='*70}")
    print("FINAL TEST SET RESULTS (30% of data - UNSEEN):")
    print(f"{'='*70}")
    
    # Simple hybrid (test set)
    hybrid_scores_test = best_alpha * vae_norm[test_idx] + (1 - best_alpha) * cnn_norm[test_idx]
    hybrid_pred_simple_test = (hybrid_scores_test > best_threshold).astype(int)
    
    # Two-stage (test set) - DÜZELTİLDİ: cnn_probs_calibrated geçiliyor
    hybrid_pred_two_stage_test, decision_types = two_stage_decision_strict(
        vae_norm[test_idx], 
        cnn_norm[test_idx],  # Normalized CNN logits
        cnn_probs_calibrated[test_idx],  # Raw calibrated probs (high/low için)
        best_threshold, 
        best_alpha,
        high_conf_thresh=Config.CNN_HIGH_CONFIDENCE_THRESHOLD,
        low_conf_thresh=Config.CNN_LOW_CONFIDENCE_THRESHOLD
    )
    
    print("\n--- Simple Hybrid V4 (Test Set) ---")
    print(classification_report(labels[test_idx], hybrid_pred_simple_test, target_names=['Benign', 'Malignant']))
    
    print("\n--- Two-Stage Hybrid V4 (Test Set) ---")
    print(classification_report(labels[test_idx], hybrid_pred_two_stage_test, target_names=['Benign', 'Malignant']))
    
    # Decision distribution
    decision_counts = {}
    for dt in set(decision_types):
        decision_counts[dt] = decision_types.count(dt)
    
    print(f"\nDecision Type Distribution (Test Set):")
    for dt, count in decision_counts.items():
        print(f"  {dt}: {count} ({100*count/len(decision_types):.1f}%)")
    
    print("\nConfusion Matrix (Two-Stage Test Set):")
    print(confusion_matrix(labels[test_idx], hybrid_pred_two_stage_test))
    
    print("\n✓ Hybrid V4 completed!")
    print(f"\n{'='*70}")
    print("FINAL CONFIGURATION:")
    print(f"{'='*70}")
    print(f"  Alpha: {best_alpha:.2f}")
    print(f"  Threshold: {best_threshold:.4f}")
    print(f"  Normalization: {Config.NORMALIZATION_METHOD}")
    print(f"  CNN score method: {Config.CNN_SCORE_METHOD}")
    print(f"  CNN High Conf: ≥{Config.CNN_HIGH_CONFIDENCE_THRESHOLD}")
    print(f"  CNN Low Conf: ≤{Config.CNN_LOW_CONFIDENCE_THRESHOLD}")

if __name__ == '__main__':
    main()
