# Tiroid Kanseri Tespiti Projesi - Detaylı Dokümantasyon

## 📋 Proje Özeti

Bu proje, **Hybrid Deep Learning** yaklaşımı kullanarak tiroid ultrasound görüntülerinden kanser tespiti yapmaktadır:

1. **VAE (Variational Autoencoder)**: Anomaly detection - sadece benign verilerle eğitilir
2. **ResNet Classifier**: Binary classification - benign vs malignant
3. **Hybrid System**: VAE + CNN birleşimi ile daha yüksek doğruluk

---

## 🗂️ Proje Yapısı ve Dosyalar

### 📂 Veri Hazırlama

#### `organize_data.py`
**Amaç**: DDTI dataset'ini organize eder (benign/malignant klasörlerine ayırır)

**Ne Yapar**:
- `category.csv` dosyasını okur
- Görüntüleri `benign/` ve `malignant/` klasörlerine kopyalar
- Veri dağılımını raporlar

**Kullanım**:
```bash
python organize_data.py
```

**Çıktı**:
