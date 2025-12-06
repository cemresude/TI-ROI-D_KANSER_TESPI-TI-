import torch

class Config:
    # Veri yolları
    # Organize edilmiş veri klasörünü kullan (organize_data.py çalıştırdıktan sonra)
    DATA_DIR = '/content/TI-ROI-D_KANSER_TESPI-TI-/organized'
    MODEL_SAVE_PATH = '/content/TI-ROI-D_KANSER_TESPI-TI-/organized/model_checkpoints'
    
    # Model parametreleri
    IMAGE_SIZE = 128  # 128x128
    LATENT_DIM = 512
    
    # Eğitim parametreleri
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    BETA = 0.001  # KL divergence weight
    
    # Diğer
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SPLIT_RATIO = 0.8
    EARLY_STOPPING_PATIENCE = 10  # Erken durdurma için sabır sayısı
    EARLY_STOPPING_DELTA = 0.001  # İyileşme için minimum delta
    # Anomaly detection threshold (test sonrası ayarlanacak)
    ANOMALY_THRESHOLD = None
