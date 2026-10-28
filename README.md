# FLO Customer Segmentation

FLO müşterilerini RFM (Recency, Frequency, Monetary) metrikleriyle
segmentlere ayıran ve iki kampanya için hedef müşteri listeleri üreten Python
projesi.

## Neler yapar?

- Müşteri verisini doğrular ve çevrim içi/çevrim dışı siparişleri birleştirir.
- Recency, frequency ve monetary metriklerini hesaplar ve 1-5 arasında puanlar.
- RF puanlarını açıklanabilir müşteri segmentlerine dönüştürür.
- Segment bazında metrik özetleri üretir.
- Kadın ayakkabı markası için sadık müşterileri (müşteri harcaması 250 TL'den
  yüksek ve kadın kategorisiyle ilgili) belirler.
- Erkek/çocuk indirimi için `cant_loose`, `hibernating` ve `new_customers`
  segmentlerinden uygun müşterileri belirler.

## Proje yapısı

```text
FLOCustomerSegmentation/
├── .github/workflows/tests.yml
├── data/
│   ├── outputs/       # Üretilen müşteri listeleri (Git'e eklenmez)
│   └── raw/           # Girdi veri seti (Git'e eklenmez)
├── src/flo_customer_segmentation/
│   ├── analysis.py    # Veri hazırlama, RFM ve kampanya hedefleri
│   └── cli.py         # Komut satırı arayüzü
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Gereksinimler

- Python 3.10 veya üzeri
- `pandas` (kurulum sırasında yüklenir)
- FLO veri seti: `flo_data_20k.csv`

Veri setini `data/raw/flo_data_20k.csv` konumuna yerleştirin. Veri seti ve
üretilen müşteri listeleri kişisel/müşteri verisi içerebileceğinden depoya
eklenmez. Veri setinin dağıtım ve kullanım izinlerini ayrıca doğrulayın.

## Kurulum ve çalıştırma

Proje kök dizininde:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m flo_customer_segmentation.cli
```

Özel girdi, çıktı dizini veya analiz tarihi vermek için:

```powershell
python -m flo_customer_segmentation.cli `
  --input data/raw/flo_data_20k.csv `
  --output-dir data/outputs `
  --analysis-date 2021-06-01
```

Analiz tarihi verilmezse veri setindeki en son sipariş tarihinden iki gün sonrası
kullanılır. Varsayılan girdi yolu `data/raw/flo_data_20k.csv`, varsayılan çıktı
yolu `data/outputs/` dizinidir.

Üretilen dosyalar:

- `new_brand_target_customer_ids.csv`
- `discount_target_customer_ids.csv`
- `segment_summary.csv`

## Beklenen veri alanları

Kaynak CSV şu sütunları içermelidir:

`master_id`, `last_order_date`, `order_num_total_ever_online`,
`order_num_total_ever_offline`, `customer_value_total_ever_online`,
`customer_value_total_ever_offline`, `interested_in_categories_12`.

Müşteri kimlikleri boş veya yinelenen olamaz; sipariş ve harcama değerleri sayısal
ve negatif olmayan değerler olmalıdır. Tarihler okunabilir tarih biçiminde
olmalıdır.

## Testler

```powershell
python -m unittest discover -s tests -v
```

Testler gerçek müşteri verisi kullanmaz.
