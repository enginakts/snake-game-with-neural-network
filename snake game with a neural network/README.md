# Yapay Sinir Ağı ile Yılan Oyunu

Bu proje, klasik yılan oyununu oynayan bir yapay zeka sistemi içerir. Deep Q-Learning kullanarak yılanın kendi kendine oynamayı öğrenmesini sağlar.

## Özellikler

- Klasik yılan oyunu implementasyonu
- Derin Q-Öğrenme tabanlı yapay zeka
- Gerçek zamanlı eğitim görselleştirmesi
- Önceden eğitilmiş modeli kullanabilme
- Eğitim istatistikleri ve grafikleri

## Gereksinimler

```bash
pip install pygame torch numpy matplotlib IPython
```

- Python 3.7+
- PyTorch
- Pygame
- Numpy
- Matplotlib
- IPython

## Dosya Yapısı

- `snake_game.py`: Yılan oyununun temel mantığı ve görsel arayüzü
- `model.py`: Yapay sinir ağı modeli ve eğitim sınıfları
- `agent.py`: Yapay zeka ajanı ve karar verme mekanizması
- `train.py`: Eğitim döngüsü ve görselleştirme
- `play.py`: Eğitilmiş modeli test etme

## Kullanım

### Eğitim

```bash
python train.py
```

İki seçenek sunulur:
1. Yeni model eğitimi başlatma
2. Var olan modeli kullanarak eğitime devam etme

### Test

```bash
python play.py
```

Eğitilmiş modeli yükleyerek yılanın oynamasını izleyebilirsiniz.

## Yapay Sinir Ağı Yapısı

### Giriş Katmanı (11 nöron):
- Tehlike durumları (3 boyut)
  - Önde tehlike var mı?
  - Sağda tehlike var mı?
  - Solda tehlike var mı?
- Mevcut yön (4 boyut)
  - Sağa mı gidiyor?
  - Sola mı gidiyor?
  - Yukarı mı gidiyor?
  - Aşağı mı gidiyor?
- Yemek konumu (4 boyut)
  - Yemek sağda mı?
  - Yemek solda mı?
  - Yemek yukarıda mı?
  - Yemek aşağıda mı?

### Gizli Katman (256 nöron):
- ReLU aktivasyon fonksiyonu
- Karmaşık örüntüleri öğrenme

### Çıkış Katmanı (3 nöron):
- Düz git
- Sağa dön
- Sola dön

## Eğitim Parametreleri

- Maksimum bellek: 100,000 hamle
- Batch boyutu: 1,000
- Öğrenme oranı: 0.001
- Gamma (indirim faktörü): 0.9
- Epsilon: Başlangıçta 80, her oyunda 1 azalır

## Ödül Sistemi

- Yemek yeme: +10 puan
- Çarpışma/ölüm: -10 puan
- Normal hareket: 0 puan

## Performans

Mevcut eğitilmiş model:
- 400+ oyun deneyimi
- En yüksek skor: 75
- Ortalama skor: ~30

## Öğrenme Süreci

1. Başlangıç (1-50 oyun):
   - Çoğunlukla rastgele hareketler
   - Düşük skorlar

2. Temel Öğrenme (50-200 oyun):
   - Duvarlardan kaçınmayı öğrenme
   - Yemeğe yönelmeyi öğrenme
   - Orta düzey skorlar

3. İleri Seviye (200+ oyun):
   - Kendi kuyruğundan kaçınma
   - Karmaşık stratejiler
   - Yüksek skorlar

## Geliştirme Fikirleri

1. Model İyileştirmeleri:
   - Daha derin ağ yapısı
   - Farklı ödül sistemleri
   - Yeni giriş özellikleri

2. Oyun Özellikleri:
   - Engeller ekleme
   - Farklı yemek türleri
   - Hız ayarları

3. Görselleştirme:
   - Detaylı istatistikler
   - Eğitim sürecini kaydetme
   - Oyun tekrarları

## Lisans

MIT License