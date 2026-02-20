@echo off
cd /d "%~dp0"
echo Dosyalar hazirlaniyor...
git add .
git commit -m "Proje dokumantasyonu eklendi (README)"
echo GitHub'a yukleniyor...
git push -u origin main
echo.
echo Islem tamamlandi! Pencereyi kapatabilirsiniz.
pause
