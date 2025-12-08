import torch

class Config:
    # Veri yolları
    DATA_DIR = '/Users/cemresudeakdag/Downloads/Thyroid Dataset/DDTI dataset/DDTI/organized'
    MODEL_SAVE_PATH = '/Users/cemresudeakdag/TİROİD_KANSER_TESPİTİ/model_checkpoints'
    
    # Model parametreleri (VAE) - İYİLEŞTİRİLDİ
    IMAGE_SIZE = 224
    LATENT_DIM = 256
    
    # Eğitim parametreleri (VAE)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    
    # Beta annealing - DÜŞÜK TUTULDU
    BETA_START = 0.0
    BETA_END = 0.001  # Düşük tut (aggressive reconstruction)
    BETA_WARMUP_EPOCHS = 40  # 30'dan 40'a uzatıldı (daha yavaş)
    
    # SSIM loss - DENGELİ
    USE_SSIM = True
    USE_MAE = True
    SSIM_WEIGHT = 0.35  # 0.4'ten 0.35'e
    MAE_WEIGHT = 0.35   # 0.3'ten 0.35'e (daha balanced)
    MSE_WEIGHT = 0.30   # 0.3'te kaldı
    
    # CNN Classifier - GÜÇLENDİRİLDİ
    CLASSIFIER_BACKBONE = 'resnet18'
    CLASSIFIER_EPOCHS = 75  # 75'ten 100'e artırıldı
    CLASSIFIER_LR = 0.0003  # Daha düşük LR (0.0005'ten)
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    
    # CNN training - BENIGN LEHİNE AĞIRLIK ARTIR
    BENIGN_CLASS_WEIGHT_MULTIPLIER = 2.0  # 1.3'ten 2.0'a çık!
    USE_TEMPERATURE_SCALING = True  # YENİ: Temperature scaling ekle
    TEMPERATURE = 1.5  # YENİ: Softmax temperature
    
    # Hybrid System V4 - BENIGN RECALL OPTIMIZE
    HYBRID_ALPHA_RANGE = [0.25, 0.30, 0.35, 0.40, 0.45]  # Daraltıldı ve artırıldı
    TARGET_MALIGNANT_RECALL = 0.85  # Sabit constraint
    TARGET_BENIGN_RECALL = 0.70  # YENİ: Benign recall hedefi
    OPTIMIZATION_METRIC = 'balanced_accuracy'  # macro_f1'den değiştirildi
    
    # Two-stage decision - DAHA SIKLAŞTIRILDI
    CNN_HIGH_CONFIDENCE_THRESHOLD = 0.85  # 0.8'den 0.85'e artırıldı
    CNN_LOW_CONFIDENCE_THRESHOLD = 0.50   # 0.4'ten 0.5'e artırıldı (daha geniş low-conf aralığı)
    
    # Normalization
    NORMALIZATION_METHOD = 'benign_validation'
    CNN_SCORE_METHOD = 'logit'
    
    # Genel parametreler
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPLIT_RATIO = 0.8
    EARLY_STOPPING_PATIENCE = 25  # 20'den 25'e artırıldı
    EARLY_STOPPING_DELTA = 0.0003  # 0.0005'ten düşürüldü
    ANOMALY_THRESHOLD = None
