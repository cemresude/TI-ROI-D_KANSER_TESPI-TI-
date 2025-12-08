# Tiroid Kanseri Tespiti - Deep Learning Projesi

Tiroid ultrasound görüntülerinden kanser tespiti için Hybrid Deep Learning yaklaşımı.

## 🎯 Proje Özeti

- **VAE (Variational Autoencoder)**: Anomaly detection
- **ResNet Classifier**: Binary classification (benign vs malignant)
- **Hybrid System**: VAE + CNN fusion

## 📊 Dataset

DDTI (Digital Database of Thyroid Images)
- Benign: ~X görüntü
- Malignant: ~Y görüntü

## 🚀 Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/cemresude/tiroid-kanser-tespiti.git
cd tiroid-kanser-tespiti

# Virtual environment oluştur
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Gereksinimleri yükle
pip install -r requirements.txt
```

## 📁 Kullanım

### 1. Veriyi Organize Et
```bash
python organize_data.py
```

### 2. VAE Eğit
```bash
python train.py
```

### 3. CNN Sınıflandırıcı Eğit
```bash
python train_classifier.py
```

### 4. Test
```bash
# VAE testi
python test.py

# Hybrid sistem testi
python hybrid_test.py
```

### 5. Hyperparameter Optimization (Opsiyonel)
```bash
python optimize.py
```

## 📈 Sonuçlar

### VAE Anomaly Detection
- ROC-AUC: X.XX
- Benign Recall: X.XX

### CNN Classifier
- Accuracy: X.XX
- F1 Score: X.XX

### Hybrid System V2
- Accuracy: X.XX
- Benign Recall: X.XX (target: 0.95)
- Macro F1: X.XX

## 🔬 Metodoloji

### İyileştirmeler
✅ ImageNet normalizasyon  
✅ Agresif augmentasyon (GaussianBlur, scale, ColorJitter)  
✅ Beta annealing (0.0 → 0.001)  
✅ SSIM + MSE hybrid loss  
✅ WeightedRandomSampler  
✅ Class weights (benign×1.2)  
✅ Cosine annealing + warmup  
✅ Mixed precision training (AMP)  
✅ CNN calibration (Isotonic Regression)  
✅ Hybrid score: α=0.75 (VAE lehine)  
✅ Benign-optimized threshold

## 📚 Detaylı Dokümantasyon

Detaylı dokümantasyon için [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) dosyasına bakın.

## 👥 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır!

## 📄 Lisans

MIT License

## 📧 İletişim

Cemre Sude Akdağ - [GitHub](https://github.com/cemresude)
