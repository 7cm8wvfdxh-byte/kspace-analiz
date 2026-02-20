# 🚀 Global Deployment Guide

Uygulamanızı tüm dünyaya açmak için en kolay ve ücretsiz yöntemlerden biri **Render.com** kullanmaktır. İşte adım adım nasıl yapacağınız:

## 1. Hazırlık
Projenize şu dosyaları ekledim:
-   `requirements.txt`: Gerekli kütüphaneler listesi.
-   `Procfile`: Sunucunun nasıl çalışacağını belirten dosya.
-   `runtime.txt`: Python sürümü.

## 2. GitHub'a Yükleme
Öncelikle projenizi GitHub'a yüklemeniz gerekiyor:
1.  [GitHub.com](https://github.com) üzerinde yeni bir "Repository" oluşturun (Örn: `dicom-kspace-analyser`).
2.  Bu klasörde terminali açıp şu komutları girin:
    ```bash
    git init
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin https://github.com/USERNAME/REPO_NAME.git
    git push -u origin main
    ```

## 3. Render.com Kurulumu
1.  [Render.com](https://render.com) adresine gidip hesap oluşturun (GitHub ile giriş yapın).
2.  Dashboard'dan **"New +"** butonuna basıp **"Web Service"** seçin.
3.  **"Build and deploy from a Git repository"** seçeneği ile ilerleyin.
4.  GitHub'daki projenizi bağlayın (`Connect`).
5.  Aşağıdaki ayarları kontrol edin:
    -   **Name:** `dicom-analyser` (veya istediğiniz isim)
    -   **Region:** `Frankfurt` (Türkiye'ye en yakın)
    -   **Branch:** `main`
    -   **Runtime:** `Python 3`
    -   **Build Command:** `pip install -r requirements.txt`
    -   **Start Command:** `uvicorn web.app:app --host 0.0.0.0 --port $PORT`
    -   **Plan:** `Free`

6.  **"Create Web Service"** butonuna basın.

## 4. Sonuç
Render projenizi derleyip sunucuya kuracaktır. İşlem bitince size `https://dicom-analyser.onrender.com` gibi global bir adres verecektir.

Artık bu linki dilediğiniz kişiyle paylaşabilirsiniz! 🎉

⚠️ **Not:** Ücretsiz planda sunucu kullanılmadığında uyku moduna geçer, ilk açılış 30-50 saniye sürebilir.
