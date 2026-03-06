# 🎣 Marmara Av Radarı PRO v6.0

Marmara Denizi için gerçek zamanlı çok-kaynaklı av radar uygulaması.

## 🌐 Canlı Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🚀 GitHub → Streamlit Cloud Deploy

### Adım 1: Repo oluştur
```bash
git init
git add app.py requirements.txt
git commit -m "Marmara Av Radarı PRO v6.0"
git remote add origin https://github.com/KULLANICI_ADI/av-radari-pro.git
git push -u origin main
```

### Adım 2: Streamlit Cloud
1. [share.streamlit.io](https://share.streamlit.io) → GitHub ile giriş
2. **New app** → Repository seç → `app.py` → Deploy
3. **Advanced settings → Secrets**:
```toml
STORMGLASS_KEY = "senin-anahtarın"
GFW_TOKEN = ""
```

## 📡 Veri Kaynakları

| Platform | Veri | Maliyet |
|----------|------|---------|
| **Open-Meteo** | Rüzgar, basınç (ICON-D2/ECMWF/GFS) | ✅ Ücretsiz |
| **Open-Meteo Marine** | Dalga, swell, periyot | ✅ Ücretsiz |
| **MET Norway** | SST, akıntı, dalga doğrulama | ✅ Ücretsiz |
| **NASA ERDDAP VIIRS** | Klorofil-a 4km | ✅ Ücretsiz |
| **Stormglass.io** | Su sıcaklığı, tuzluluk | 🔑 10/gün ücretsiz |
| **Overpass (OSM)** | Kıyı izohipsleri | ✅ Ücretsiz |
| **OpenSeaMap** | Deniz işaretleri (WMS) | ✅ Ücretsiz |
| **Global Fishing Watch** | Balıkçı aktivitesi | 🔑 Ücretsiz kayıt |

## 📊 Özellikler
- 35+ Marmara mera noktası
- 4 model doğrulama (ICON-D2/EU, ECMWF, GFS)
- Kıyıya dik rüzgar analizi
- Yeşil bulut klorofil overlay (scipy + matplotlib)
- Animasyonlu SVG rüzgar okları
- 2hPa izobar kontur haritası
- MET Norway bağımsız doğrulama
- 6 saatlik trend göstergeleri
- 7 günlük trend grafikleri
- OpenSeaMap deniz işaretleri

## 📁 Dosyalar
- `app.py` — Ana Streamlit uygulaması
- `requirements.txt` — Python bağımlılıklar
- `secrets_template.toml` — API key şablonu (repo'ya ekleme!)
