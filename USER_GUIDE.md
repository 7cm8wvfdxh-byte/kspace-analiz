# 🏥 K-Space Analiz Platformu - Kullanım Kılavuzu

Bu uygulama, MR (Manyetik Rezonans) görüntülerinin ham frekans verilerini (K-Space) analiz ederek, gözle görülmesi zor olan ince detayları ve anormallikleri tespit etmenize yardımcı olur.

İşte bu platform ile yapabilecekleriniz:

## 1. 📂 Otomatik Anomali Tespiti
MR kesitlerinizi (DICOM klasörü) sisteme yüklediğinizde, yapay zeka şunları yapar:
-   **K-Space Dönüşümü:** Görüntüleri frekans uzayına çevirir.
-   **Fark Analizi (Differential Analysis):** Ardışık kesitler arasındaki frekans değişimlerini ölçer.
-   **Anomali Yakalama:** Normal doku geçişine uymayan ani frekans sıçramalarını (dK skoru) tespit eder.

**Nasıl Kullanılır:**
1.  **Upload** sekmesinden dosyanızı seçin ve "Upload & Analyze" butonuna basın.
2.  İşlem bitince otomatik olarak **Dashboard** açılır.
3.  **Transitions** tablosunda `dK Score` değeri kırmızı olan satırlar, şüpheli kesitleri gösterir.

## 2. 📝 Yapay Zeka Raporu (Türkçe)
Her analizin sonucunda, radyomik verilere dayalı bir özet rapor sunulur.
-   **Anomali:** Hangi kesitlerde sorun var?
-   **Doku:** Doku homojen mi (Faz Uyumu), yoksa karmaşık mı (Entropi)?
-   Rapor, Dashboard'un en üstünde "Analiz Raporu" kutusunda yer alır.

## 3. ⚖️ Çoklu Seri Karşılaştırma (Multi-Series Comparison)
İki farklı çekimi (Örn: İlaçsız T1 vs İlaçlı T1+C) kıyaslayabilirsiniz.
-   Bu modül, iki seri arasındaki **K-Space fark haritasını** çıkarır.
-   Kontrast tutulumunun veya doku değişiminin frekans boyutundaki yansımasını gösterir.

**Nasıl Kullanılır:**
1.  **Compare** sekmesine gidin.
2.  Soldan "Baseline" (Referans), sağdan "Comparison" (Kıyaslanacak) çalışmayı seçin.
3.  **Run Comparison** butonuna basın.

## 4. 🧊 3D Volumetric Görselleştirme
Anomalilerin beyin (veya incelenen organ) içindeki yerleşimini 3 boyutlu olarak görebilirsiniz.
-   K-Space anomalileri (yüksek frekanslı sapmalar) 3D uzayda noktalar halinde işaretlenir.
-   Noktaların yoğunlaştığı bölge, patolojinin (tümör vb.) merkezini işaret edebilir.

**Nasıl Kullanılır:**
1.  Dashboard'da **"3D View"** butonuna basın.
2.  Açılan pencerede mouse ile modeli çevirebilir, zoom yapabilirsiniz.
