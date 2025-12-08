import torch

class Config:
    # Veri yolları - BURASI ÇOK ÖNEMLİ! organize_data.py çıktısına göre güncelleyin
    DATA_DIR = '/Users/cemresudeakdag/Downloads/Thyroid Dataset/DDTI dataset/DDTI/organized'
    MODEL_SAVE_PATH = '/Users/cemresudeakdag/TİROİD_KANSER_TESPİTİ/model_checkpoints'
    
    # Model parametreleri (VAE)
    IMAGE_SIZE = 128
    LATENT_DIM = 512
    
    # Eğitim parametreleri (VAE)
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    BETA = 0.001
    
    # Beta annealing parametreleri
    BETA_START = 0.0
    BETA_END = 0.005  # 0.001'den 0.005'e artırıldı
    BETA_WARMUP_EPOCHS = 15  # 20'den 15'e düşürüldü (daha hızlı)
    
    # SSIM loss
    USE_SSIM = True
    SSIM_WEIGHT = 0.3  # 0.5'ten 0.3'e düşürüldü (MSE daha dominant)
    
    # CNN Classifier parametreleri
    CLASSIFIER_BACKBONE = 'resnet18'  # 'resnet18' veya 'resnet50'
    CLASSIFIER_EPOCHS = 50
    CLASSIFIER_LR = 0.001
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    
    # Hybrid System V2 parametreleri
    HYBRID_ALPHA = 0.75  # VAE ağırlığı (0.7-0.8 önerilir)
    TARGET_BENIGN_RECALL = 0.95  # Hedef benign recall
    USE_CNN_CALIBRATION = True  # CNN probability calibration
    
    # Hybrid System V3 parametreleri (Grid-search optimized)
    HYBRID_ALPHA_RANGE = [0.2, 0.3, 0.4, 0.5, 0.6]  # Grid-search için alpha aralığı
    HYBRID_ALPHA = 0.4  # Default (grid-search sonrası güncellenecek)
    TARGET_MALIGNANT_RECALL = 0.85  # Malignant recall hedefi
    TARGET_BENIGN_RECALL = 0.95  # Benign recall hedefi (opsiyonel)
    
    # Two-stage decision
    CNN_HIGH_CONFIDENCE_THRESHOLD = 0.7  # CNN p_cls ≥ 0.7 ise doğrudan malignant
    CNN_LOW_CONFIDENCE_THRESHOLD = 0.3   # CNN p_cls ≤ 0.3 ise doğrudan benign
    
    # Normalization strategy
    NORMALIZATION_METHOD = 'validation'  # 'validation', 'zscore', 'minmax'
    
    # CNN training (benign recall artırma)
    BENIGN_CLASS_WEIGHT_MULTIPLIER = 1.5  # 1.2'den 1.5'e artırıldı
    
    # Genel parametreler
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPLIT_RATIO = 0.8
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_DELTA = 0.001
    ANOMALY_THRESHOLD = None
