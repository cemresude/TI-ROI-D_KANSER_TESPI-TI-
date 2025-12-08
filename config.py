import torch

class Config:
    # Veri yolları
    DATA_DIR = '/Users/cemresudeakdag/Downloads/Thyroid Dataset/DDTI dataset/DDTI/organized'
    MODEL_SAVE_PATH = '/Users/cemresudeakdag/TİROİD_KANSER_TESPİTİ/model_checkpoints'
    
    # Model parametreleri (VAE) - İYİLEŞTİRİLDİ
    IMAGE_SIZE = 224  # 128'den 224'e artırıldı
    LATENT_DIM = 256  # 512'den 256'ya düşürüldü (daha az overfitting)
    
    # Eğitim parametreleri (VAE) - İYİLEŞTİRİLDİ
    BATCH_SIZE = 16  # 32'den 16'ya düşürüldü (224x224 için)
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    
    # Beta annealing parametreleri - İYİLEŞTİRİLDİ
    BETA_START = 0.0
    BETA_END = 0.001  # 0.005'ten 0.001'e düşürüldü (daha az regularization)
    BETA_WARMUP_EPOCHS = 30  # 15'ten 30'a artırıldı (daha yavaş warmup)
    
    # SSIM loss - İYİLEŞTİRİLDİ (MAE + SSIM)
    USE_SSIM = True
    USE_MAE = True  # YENİ: MAE ekle
    SSIM_WEIGHT = 0.4  # 0.3'ten 0.4'e artırıldı
    MAE_WEIGHT = 0.3   # YENİ: MAE ağırlığı
    MSE_WEIGHT = 0.3   # Geri kalan MSE
    
    # CNN Classifier parametreleri
    CLASSIFIER_BACKBONE = 'resnet18'
    CLASSIFIER_EPOCHS = 75
    CLASSIFIER_LR = 0.0005
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    
    # Hybrid System V4 parametreleri - TAM YENİLENDİ
    HYBRID_ALPHA_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    TARGET_MALIGNANT_RECALL = 0.85  # Kesin kısıt
    OPTIMIZATION_METRIC = 'macro_f1'  # 'macro_f1' veya 'balanced_accuracy'
    
    # Two-stage decision - SIKLAŞTIRILDI
    CNN_HIGH_CONFIDENCE_THRESHOLD = 0.8  # 0.7'den 0.8'e artırıldı
    CNN_LOW_CONFIDENCE_THRESHOLD = 0.4   # 0.3'ten 0.4'e artırıldı
    
    # Normalization strategy - DEĞİŞTİRİLDİ
    NORMALIZATION_METHOD = 'benign_validation'  # Sadece benign validation stats
    CNN_SCORE_METHOD = 'logit'  # 'probability' veya 'logit'
    
    # CNN training
    BENIGN_CLASS_WEIGHT_MULTIPLIER = 1.0
    
    # Genel parametreler
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPLIT_RATIO = 0.8
    EARLY_STOPPING_PATIENCE = 20  # 15'ten 20'ye artırıldı
    EARLY_STOPPING_DELTA = 0.0005  # 0.001'den düşürüldü
    ANOMALY_THRESHOLD = None
