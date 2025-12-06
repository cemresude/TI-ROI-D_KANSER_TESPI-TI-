import os
import shutil
import pandas as pd
from pathlib import Path

def organize_thyroid_data():
    """
    DDTI veri setini category.csv'ye göre benign ve malignant klasörlerine ayırır
    """
    # Yollar
    
    ddti_root = Path('/Users/cemresudeakdag/Downloads/Thyroid Dataset/DDTI dataset/DDTI')
    base_dir = ddti_root / '1_or_data'
    
    # CSV dosyasının olası konumlarını kontrol et
    possible_csv_paths = [
        base_dir / 'category.csv',  # 1_or_data içinde
        ddti_root / 'category.csv',  # DDTI ana klasöründe
        base_dir / 'Category.csv',   # Büyük harfle
        ddti_root / 'Category.csv'
    ]
    
    csv_path = None
    for path in possible_csv_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print(f"HATA: CSV dosyası bulunamadı!")
        print(f"Kontrol edilen konumlar:")
        for path in possible_csv_paths:
            print(f"  - {path}")
        return

    image_dir = base_dir / 'image'
    
    # Hedef klasörler
    output_dir = ddti_root / 'organized'
    benign_dir = output_dir / 'benign'
    malignant_dir = output_dir / 'malignant'
    
    # Klasörleri oluştur
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DDTI Veri Seti Organizasyonu")
    print("=" * 60)
    
    if not image_dir.exists():
        print(f"HATA: Image klasörü bulunamadı: {image_dir}")
        return
    
    # CSV dosyasını oku
    print(f"\nCSV dosyası okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Toplam kayıt sayısı: {len(df)}")
    print(f"\nKolon isimleri: {list(df.columns)}")
    print(f"\nİlk birkaç satır:")
    print(df.head())
    
    # Kategori dağılımını göster
    if 'CATE' in df.columns:
        print(f"\nKategori dağılımı:")
        print(df['CATE'].value_counts())
        print(f"\nBenzersiz kategori değerleri:")
        for cat in df['CATE'].unique():
            print(f"  '{cat}' (tip: {type(cat)})")
    else:
        print(f"\nUYARI: 'CATE' kolonu bulunamadı!")
        print(f"Mevcut kolonlar: {list(df.columns)}")
        return
    
    # Görüntü dosya adı kolonunu kontrol et
    if 'ID' not in df.columns:
        print(f"\nUYARI: 'ID' kolonu bulunamadı!")
        print(f"Mevcut kolonlar: {list(df.columns)}")
        return
    
    filename_col = 'ID'
    print(f"\nDosya adı kolonu: '{filename_col}'")
    
    # Görüntüleri organize et
    benign_count = 0
    malignant_count = 0
    skipped_count = 0
    unknown_categories = {}  # Bilinmeyen kategorileri say
    
    print(f"\nGörüntüler organize ediliyor...")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        filename = str(row[filename_col])
        category = str(row['CATE']).strip()  # strip() ekle
        
        # Dosya uzantısını kontrol et
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF')):
            # Eğer uzantı yoksa, olası uzantıları dene
            possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            found = False
            for ext in possible_extensions:
                test_path = image_dir / (filename + ext)
                if test_path.exists():
                    filename = filename + ext
                    found = True
                    break
            
            if not found:
                print(f"  Atlandı (dosya bulunamadı): {filename}")
                skipped_count += 1
                continue
        
        src_path = image_dir / filename
        
        if not src_path.exists():
            print(f"  Atlandı (dosya yok): {filename}")
            skipped_count += 1
            continue
        
        # Kategoriyi belirle - CATE değerlerine göre (1=benign, 2=malignant veya benzeri)
        # Önce tam eşleşme dene
        if category in ['0', '0.0', 'benign', 'Benign', 'BENIGN']:
            dest_path = benign_dir / filename
            benign_count += 1
        elif category in ['1', '1.0', 'malignant', 'Malignant', 'MALIGNANT']:
            dest_path = malignant_dir / filename
            malignant_count += 1
        # Sonra substring eşleşme dene
        elif 'benign' in category.lower():
            dest_path = benign_dir / filename
            benign_count += 1
        elif 'malignant' in category.lower():
            dest_path = malignant_dir / filename
            malignant_count += 1
        else:
            # Bilinmeyen kategorileri kaydet
            unknown_categories[category] = unknown_categories.get(category, 0) + 1
            if skipped_count < 10:
                print(f"  Atlandı (bilinmeyen kategori '{category}'): {filename}")
            skipped_count += 1
            continue
        
        # Dosyayı kopyala
        try:
            shutil.copy2(src_path, dest_path)
            if (benign_count + malignant_count) % 100 == 0:
                print(f"  İşlenen: {benign_count + malignant_count} dosya...")
        except Exception as e:
            print(f"  HATA ({filename}): {e}")
            skipped_count += 1
    
    print("-" * 60)
    print("\n" + "=" * 60)
    print("Organizasyon Tamamlandı!")
    print("=" * 60)
    print(f"\nBenign görüntüler    : {benign_count} dosya")
    print(f"  Konum: {benign_dir}")
    print(f"\nMalignant görüntüler : {malignant_count} dosya")
    print(f"  Konum: {malignant_dir}")
    print(f"\nAtlanan dosyalar    : {skipped_count} dosya")
    
    if unknown_categories:
        print(f"\n⚠️  Bilinmeyen kategori değerleri:")
        for cat, count in sorted(unknown_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{cat}': {count} dosya")
    
    print(f"\nToplam              : {benign_count + malignant_count} dosya kopyalandı")
    print("=" * 60)
    
    # Config.py'yi güncelleme önerisi
    print(f"\n💡 ÖNERİ:")
    print(f"config.py dosyasındaki DATA_DIR yolunu şu şekilde güncelleyin:")
    print(f"DATA_DIR = '{output_dir}'")
    print()

if __name__ == '__main__':
    try:
        organize_thyroid_data()
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
