# 📈 Time Series Forecasting Benchmark: Traditional vs. Deep Learning vs. Foundation Models

Bu proje; zaman serisi tahminlemesinde geleneksel istatistiksel modellerin, modern derin öğrenme mimarilerinin ve sıfır-eğitim (zero-shot) yeteneğine sahip yeni nesil Foundation Modellerin kapsamlı bir performans karşılaştırmasını sunmaktadır.

## 🚀 Proje Özeti
Zaman serisi problemleri veri yapısına göre (seyrek, gürültülü, trend ağırlıklı) farklı zorluklar barındırır. Bu çalışmada **3 farklı zorluk seviyesindeki** veri seti kullanılarak, **6 farklı algoritma** CPU ve GPU (NVIDIA GTX 1660 Ti) donanım limitleri optimize edilerek test edilmiştir.

### 📊 Kullanılan Veri Setleri
1. **AirPassengers (Kolay):** Net trend ve mevsimsellik barındıran aylık veri.
2. **ETTh1 (Orta):** Elektrik trafosu yük/sıcaklık değişimlerini içeren gürültülü saatlik veri.
3. **M5 Hobbies (Zor):** Walmart perakende satışlarına ait, yoğun olarak "sıfır" (sparse) değerler içeren günlük veri.

### 🧠 Karşılaştırılan Modeller
* **İstatistiksel (CPU Paralel İşleme):** AutoARIMA, AutoETS
* **Derin Öğrenme (GPU Hızlandırmalı):** LSTM, GRU, N-BEATS
* **Foundation Model (Zero-Shot):** Amazon Chronos (t5-small)

---

## 🛠️ Kurulum ve Gereksinimler

Proje donanım kaynaklarını (CPU/GPU) otomatik algılayacak şekilde tasarlanmıştır.

```bash
# Repoyu klonlayın
git clone [https://github.com/KULLANICI_ADIN/Time_Series_Benchmark.git](https://github.com/KULLANICI_ADIN/Time_Series_Benchmark.git)
cd Time_Series_Benchmark

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

(Gereksinimler: pandas, numpy, torch, statsforecast, neuralforecast, chronos-forecasting, matplotlib, scikit-learn)
⚙️ Benchmark Pipeline (Kullanım)
```

Proje modüler bir mimariye sahiptir. İşlemleri sırasıyla çalıştırabilirsiniz:

1. Veri Hazırlığı: Veri setlerini indirir, yapılandırır ve Train/Test olarak böler.
Bash

python data_preparation.py

2. Geleneksel ve DL Modellerinin Eğitimi: AutoARIMA/ETS için CPU çekirdeklerini asenkron kullanır; N-BEATS, LSTM ve GRU için GPU'yu devreye sokar.
Bash

python model_training.py

3. Foundation Model Çıkarımı (Chronos): Model eğitilmez (Zero-Shot). OOM (Out of Memory) hatalarını engellemek için Adaptive Batching (Esnek Gruplama) algoritması kullanılarak donanım limitleri güvenli şekilde sonuna kadar kullanılır.
Bash

python chronos_benchmark.py

4. Raporlama ve Görselleştirme: MAE, RMSE, MAPE, sMAPE, WAPE ve R2 metriklerini hesaplar, birleştirir ve grafikleri oluşturur. M5 gibi seyrek veri setlerinde MAPE metriği yanılttığı için özellikle WAPE baz alınmıştır.
Bash

python final_report.py

🏆 Temel Bulgular ve Sonuçlar

    N-BEATS'in Hakimiyeti: ETTh1 ve AirPassengers gibi standart/gürültülü zaman serilerinde N-BEATS, varyansı açıklamada (R2 Score: ~0.93) en başarılı model olmuştur.

    M5 Sparse Data Zorluğu: M5 veri setinin aşırı seyrek yapısında klasik modeller (ARIMA) düz çizgiler çizerken; LSTM, GRU ve Chronos sıfır noktalarını ve anlık yükselişleri daha iyi kavramış ve en düşük WAPE skorlarını vermiştir.

    Sıfır Eğitim Gücü (Chronos): Amazon Chronos, veriyi hiç görmemesine ve eğitilmemesine rağmen, saatlerce eğitilen RNN mimarileriyle (LSTM/GRU) kafa kafaya yarışmış, Foundation Modellerin zaman serilerindeki potansiyelini kanıtlamıştır.

Tüm metrik detaylarına ve grafiklere reports/ klasöründen ulaşabilirsiniz.

Geliştirici: Taha Sencer Şenel