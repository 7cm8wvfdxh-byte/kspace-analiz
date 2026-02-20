# 🧠 K-Space DICOM Analiz Platformu

MR (Manyetik Rezonans) görüntülerinin ham frekans verilerini (K-Space) analiz ederek, gözle görülmesi zor patolojileri ve anomalileri tespit eden yapay zeka destekli analiz aracı.

![K-Space Analysis](https://via.placeholder.com/800x400?text=K-Space+Analysis+Dashboard "Ekran Görüntüsü Temsilidir")

## 🚀 Özellikler

### 1. 🔍 Otomatik Anomali Tespiti
Sistem, DICOM görüntülerini K-Space'e dönüştürür ve ardışık kesitler arasındaki frekans değişimlerini (Differantial K-Space) analiz eder. Normal doku geçişine uymayan ani sıçramaları tespit eder.

### 2. 📝 Yapay Zeka Raporu (Türkçe)
Her analiz sonunda radyomik verilere dayalı detaylı bir rapor sunulur:
*   Anomali var mı?
*   Doku homojen mi?
*   Hangi kesitler riskli?

### 3. ⚖️ Çoklu Seri Karşılaştırma (Comparison)
Farklı çekimleri (Örn: T1 vs T1+C) yan yana koyarak K-Space fark haritasını çıkarabilirsiniz. Kontrast tutulumunu frekans boyutunda görebilirsiniz.

### 4. 🧊 3D Volumetric Görselleştirme
Tespit edilen anomalileri 3 boyutlu uzayda inceleyin.
*   **Glow Effect:** Parlayan noktalar ile anomali yoğunluğunu görün.
*   **İnteraktif Kontroller:** Eşik (Threshold) ve boyut ayarları ile gürültüyü filtreleyin.

---

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
*   Python 3.10+
*   Gerekli kütüphaneler: `requirements.txt`

### Yerel Çalıştırma
```bash
# 1. Kütüphaneleri yükleyin
pip install -r requirements.txt

# 2. Uygulamayı başlatın
uvicorn web.app:app --reload
```
Tarayıcıda `http://localhost:8000` adresine gidin.

## 📄 Kullanım Kılavuzu
Detaylı kullanım rehberi için [USER_GUIDE.md](USER_GUIDE.md) dosyasına bakabilirsiniz.

## ☁️ Yayına Alma (Deployment)
Bu projeyi Render.com veya benzeri platformlarda yayınlamak için [DEPLOYMENT.md](DEPLOYMENT.md) dosyasındaki adımları takip edin.

---
*Geliştirildi: 2026, K-Space Research Lab*
