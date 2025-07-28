# change-detection
Uydu görüntülerinde değişim tespiti projesi
# Uydu Görüntülerinde Değişim Tespit Sistemi

## 📌 Proje Amacı
İki farklı zamanda elde edilmiş ve aynı bölgeyi kapsayan uydu görüntülerinde;
- Yapılaşma değişiklikleri
- Doğal afet etkileri
- Arazi kullanım değişimleri
gibi nedenlerle değişimler gözlemlenebilmektedir. Python ile geliştirilmiş bu projede iki farklı zaman aralığında çekilmiş uydu görüntüleri arasındaki farkların tespit edilmesi amaçlanmıştır.

## 🌟 Temel Özellikler
| Özellik                | Açıklama                                      | Teknoloji          |
|------------------------|-----------------------------------------------|--------------------|
| Çoklu Analiz Yöntemi   | SSIM, Fark Alma ve Derin Öğrenme              | OpenCV, PyTorch    |
| Kullanıcı Dostu Arayüz | Kolay görüntü yükleme ve sonuç görselleştirme | Tkinter            |
| Model Eğitimi          | Özel veri setiyle model geliştirme imkanı     | U-Net Architecture |
| Çapraz Platform        | Windows/Linux/macOS uyumlu                    | Python 3.8+        |

git clone https://github.com/safiye-cagla/satellite-change-detection.git

📌 Eğitim ve test için kullanılan veri seti aşağıdaki bağlantıdan elde edilmiştir;
"https://www.kaggle.com/code/sakifalam/change-detection-epoch-100/input"

