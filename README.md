# Tiroid Kanseri Tespiti - ConvVAE Anomaly Detection

DDTI veri setini kullanarak Convolutional Variational Autoencoder (ConvVAE) ile tiroid kanseri anomali tespiti.

## Yaklaşım

Model sadece **BENIGN** (normal) verilerle eğitilir. Test aşamasında, MALIGNANT (kanserli) veriler yüksek reconstruction error göstererek anomali olarak tespit edilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Veri Hazırlama

**ÖNEMLİ:** Eğitim öncesi veri setini organize etmelisiniz!

```bash
# DDTI veri setini category.csv'ye göre organize et
python organize_data.py
```

Bu script:
- `DDTI/image/` klasöründeki tüm görüntüleri okur
- `category.csv` dosyasındaki kategorilere göre ayırır
- `DDTI/organized/benign/` ve `DDTI/organized/malignant/` klasörlerine kaydeder

## Kullanım

### 1. Model Eğitimi (Sadece BENIGN verilerle)
```bash
python train.py
```

### 2. Test (BENIGN + MALIGNANT verilerle)
```bash
python test.py
```

## Model Özellikleri

### Preprocessing
- Görüntüler 128x128 boyutuna resize edilir
- Dikdörtgen görüntüler için siyah padding eklenir (aspect ratio korunur)
- Piksel değerleri 0-1 aralığında normalize edilir

### ConvVAE Mimarisi
**Encoder:**
- Conv2D → BatchNorm → LeakyReLU (5 katman)
- Latent space: mean (μ) ve log-variance (log σ²)

**Reparameterization Trick:**
```
z = μ + σ * ε, ε ~ N(0, 1)
```

**Decoder:**
- ConvTranspose2D → BatchNorm → LeakyReLU (5 katman)
- Son katman: Sigmoid (0-1 çıktı)

### Loss Fonksiyonu
```
Total Loss = MSE(x, x') + β * KL(q(z|x) || p(z))
```
- **MSE**: Reconstruction loss
- **KL Divergence**: Latent space regularization
- **β**: KL weight (default: 1.0)

## Test Metrikleri

- Reconstruction Error distribution
- ROC Curve ve AUC
- Confusion Matrix
- Classification Report
- Görsel karşılaştırmalar

## Dosya Yapısı
```
TİROİD_KANSER_TESPİTİ/
├── organize_data.py  # Veri organizasyon scripti
├── model.py          # ConvVAE modeli
├── data_loader.py    # Veri yükleme (padding, normalize)
├── train.py          # Eğitim (sadece BENIGN)
├── test.py           # Test (anomaly detection)
├── config.py         # Konfigürasyon
├── utils.py          # Yardımcı fonksiyonlar
├── requirements.txt  # Gereksinimler
└── checkpoints/      # Model kayıtları
    ├── best_model_vae.pth
    ├── reconstruction_comparison.png
    ├── roc_curve.png
    └── error_distribution.png
```

## DDTI Veri Seti Yapısı

**Orijinal:**
```
DDTI/
├── image/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── category.csv
```

**Organize edilmiş (organize_data.py sonrası):**
```
DDTI/
├── organized/
│   ├── benign/
│   │   ├── img001.jpg
│   │   └── ...
│   └── malignant/
│       ├── img050.jpg
│       └── ...
```
