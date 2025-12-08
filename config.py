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
    BETA_END = 0.001
    BETA_WARMUP_EPOCHS = 20
    
    # SSIM loss
    USE_SSIM = True
    SSIM_WEIGHT = 0.5
    
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
    
    # Genel parametreler
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPLIT_RATIO = 0.8
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_DELTA = 0.001
    ANOMALY_THRESHOLD = None
