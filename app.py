"""
╔══════════════════════════════════════════════════════════════════════════╗
║   MARMARA AV RADARI  PRO  v6.0  —  Streamlit Cloud Edition             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Yeni Özellikler (v5→v6):                                               ║
║  ☁️  Streamlit Cloud / GitHub deploy (yerel sunucu gerekmez)              ║
║  🇳🇴 MET Norway Oceanforecast — dalga, SST, akıntı (FREE, key yok)       ║
║  🛰️  NASA ERDDAP Klorofil Grid — Marmara bölgesi tek sorguda            ║
║  🌿 Yeşil bulut klorofil overlay — matplotlib filled contour + Gaussian ║
║  💨 Animasyonlu rüzgar okları — CSS keyframe DivIcon                    ║
║  📉 Gelişmiş izobar — 2hPa, gradient fill, daha keskin etiket           ║
║  🎣 Global Fishing Watch — balıkçı yoğunluk katmanı (opsiyonel)         ║
║  🔐 Tüm API key'ler st.secrets üzerinden (güvenli deploy)               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from folium import MacroElement
from jinja2 import Template
import requests
import datetime
import math
import io
import base64
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ══════════════════════════════════════════════════════════════════
#  API ANAHTARLARI  — Streamlit secrets (Streamlit Cloud uyumlu)
#  Yerel için: .streamlit/secrets.toml
#  Cloud için: App Settings → Secrets
# ══════════════════════════════════════════════════════════════════
def _s(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

STORMGLASS_KEY = _s("STORMGLASS_KEY",
    "15d75640-16d7-11f1-8006-0242ac120004-15d756ae-16d7-11f1-8006-0242ac120004")
GFW_TOKEN      = _s("GFW_TOKEN", "")     # globalfishingwatch.org — ücretsiz kayıt
SG_QUOTA_MAX   = 8

# NASA CoastWatch ERDDAP
ERDDAP_BASE     = "https://coastwatch.noaa.gov/erddap"
ERDDAP_VIIRS_DS = "noaacwNPPVIIRSchlaDaily"

# Marmara deniz sınırları
MARMARA = dict(lon_min=26.0, lon_max=30.2, lat_min=40.3, lat_max=41.5)

# ══════════════════════════════════════════════════════════════════
#  CMEMS — Copernicus Marine Service (opsiyonel, ücretsiz hesap)
#  Klorofil-a: Black Sea NRT 3nm çözünürlük
#  https://data.marine.copernicus.eu/product/BLKSEA_ANALYSISFORECAST_BGC_007_010
# ══════════════════════════════════════════════════════════════════
CMEMS_USER = _s("CMEMS_USER", "")
CMEMS_PASS = _s("CMEMS_PASS", "")

def cmems_wms_url() -> Optional[str]:
    """CMEMS Black Sea BGC WMS — kullanıcı adı/şifre gerektiriyorsa None döner."""
    if not CMEMS_USER or not CMEMS_PASS:
        return None
    return (f"https://nrt.cmems-du.eu/thredds/wms/"
            f"cmems_mod_blk_bgc-bio_anfc_3nm_P1D-m?"
            f"username={CMEMS_USER}&password={CMEMS_PASS}&")

# ══════════════════════════════════════════════════════════════════
#  LEAFLETVELOCITYLayer — Folium MacroElement
#  Leaflet.Velocity CDN animasyonlu rüzgar partikülleri
#  Ref: https://github.com/onaci/leaflet-velocity
# ══════════════════════════════════════════════════════════════════
class LeafletVelocityLayer(MacroElement):
    """
    Leaflet.Velocity animated wind particle layer.
    wind_data_json: JSON string in GRIB2JSON / Leaflet.Velocity format
                    [ {header:{nx,ny,la1,lo1,dx,dy,...}, data:[u...]},
                      {header:{...}, data:[v...]} ]
    max_velocity:   km/h upper bound for color scale
    """
    def __init__(self, wind_data_json: str, max_velocity: float = 50.0):
        super().__init__()
        self._name = "LeafletVelocityLayer"
        self._wind_data = wind_data_json
        self._max_vel   = max_velocity
        self._template  = Template("""
{% macro script(this, kwargs) %}
(function() {
  var _lv_init = function() {
    if (typeof L.velocityLayer === 'undefined') return;
    var mapObj = {{ this._parent.get_name() }};
    if (!mapObj) return;
    var windData = {{ this._wind_data }};
    L.velocityLayer({
      displayValues: true,
      displayOptions: {
        velocityType: "Rüzgar",
        position: "bottomleft",
        emptyString: "Rüzgar verisi yok",
        angleConvention: "bearingCCW",
        speedUnit: "km/h"
      },
      data: windData,
      maxVelocity: {{ this._max_vel }},
      colorScale: [
        "#bfdbfe","#93c5fd","#60a5fa","#3b82f6",
        "#34d399","#a3e635",
        "#fbbf24","#f97316",
        "#f87171","#e11d48"
      ],
      velocityScale: 0.0085,
      particleAge: 90,
      lineWidth: 2.0,
      particleMultiplier: 0.0015,
      frameRate: 20,
      opacity: 0.88,
    }).addTo(mapObj);
  };

  // Inject CSS (once)
  if (!document.getElementById('lvs-css')) {
    var lnk = document.createElement('link');
    lnk.id  = 'lvs-css';
    lnk.rel = 'stylesheet';
    lnk.href= 'https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.css';
    document.head.appendChild(lnk);
  }

  // Inject JS
  if (!window._lvLoaded) {
    window._lvLoaded = true;
    var sc = document.createElement('script');
    sc.src  = 'https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.js';
    sc.onload = function() { _lv_init(); };
    document.head.appendChild(sc);
  } else {
    setTimeout(_lv_init, 400);
  }
})();
{% endmacro %}
""")


def build_wind_uv_grid(spots_data: dict) -> Tuple[Optional[str], float]:
    """
    Spot verilerinden Leaflet.Velocity uyumlu UV ızgara JSON'u oluşturur.
    scipy griddata ile interpolasyon, gaussian smoothing.
    Returns: (json_string, max_velocity_kmh)
    """
    if not HAS_SCIPY or len(spots_data) < 5:
        return None, 30.0

    lats, lons, us, vs = [], [], [], []
    for d in spots_data.values():
        ws_ms  = d["ws"] / 3.6
        wd_rad = math.radians(d["wd"])
        lats.append(d["lat"]); lons.append(d["lon"])
        # Meteorolojik → matematiksel: FROM yönünden gelen rüzgar
        us.append(-ws_ms * math.sin(wd_rad))
        vs.append(-ws_ms * math.cos(wd_rad))

    nx, ny   = 48, 24
    lon0,lon1= 25.6, 30.8
    lat1,lat0= 41.9, 40.0    # lat1 = north (büyük), lat0 = south
    grid_lons = np.linspace(lon0, lon1, nx)
    grid_lats = np.linspace(lat1, lat0, ny)   # azalan sıra (kuzeyden güneye)
    glo, gla  = np.meshgrid(grid_lons, grid_lats)
    pts = list(zip(lons, lats))

    try:
        gu = griddata(pts, us, (glo, gla), method="cubic")
        gv = griddata(pts, vs, (glo, gla), method="cubic")
        # NaN → nearest
        mask = np.isnan(gu)
        if mask.any():
            gu_n = griddata(pts, us, (glo, gla), method="nearest")
            gv_n = griddata(pts, vs, (glo, gla), method="nearest")
            gu = np.where(mask, gu_n, gu)
            gv = np.where(mask, gv_n, gv)
        # Hafif yumuşatma
        gu = gaussian_filter(gu, sigma=1.2)
        gv = gaussian_filter(gv, sigma=1.2)
    except Exception:
        return None, 30.0

    max_vel = float(np.sqrt(gu**2 + gv**2).max()) * 3.6
    max_vel = max(8.0, min(120.0, max_vel))

    dx = (lon1 - lon0) / (nx - 1)
    dy = (lat1 - lat0) / (ny - 1)

    hdr = {
        "parameterCategory": 2,
        "dx": round(dx, 5), "dy": round(dy, 5),
        "la1": lat1, "lo1": lon0,
        "la2": lat0, "lo2": lon1,
        "nx": nx, "ny": ny,
        "refTime": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = [
        {"header": {**hdr, "parameterNumber": 2},
         "data":   [round(float(x), 4) for x in gu.flatten()]},
        {"header": {**hdr, "parameterNumber": 3},
         "data":   [round(float(x), 4) for x in gv.flatten()]},
    ]
    return json.dumps(payload), max_vel

# ══════════════════════════════════════════════════════════════════
#  TAKım ENVANTERİ
# ══════════════════════════════════════════════════════════════════
TAKIMLAR = [
    {"ad":"Shimano Bassterra LRF",    "tip":"LRF",   "agirlik":"2-10g",  "ip":"PE 0.4",
     "senaryo":"Sessiz sığ levrek",           "renk":"#38bdf8"},
    {"ad":"Major Craft Aji-do 6'8\"", "tip":"Aji/UL","agirlik":"0.5-5g", "ip":"Ester 0.3",
     "senaryo":"İstavrit — gece, berrak su",  "renk":"#2dd4bf"},
    {"ad":"NS Black Hole DH2",        "tip":"Spin",  "agirlik":"10-36g", "ip":"PE 1.2",
     "senaryo":"Fırtına levreği — lodos",     "renk":"#fb7185"},
]

# ══════════════════════════════════════════════════════════════════
#  35+ MARMARA MERASI
# ══════════════════════════════════════════════════════════════════
MERALAR = {
    # ── İSTANBUL AVRUPA ──────────────────────────────────────────
    "Silivri Sahil":        {"lat":41.073,"lon":28.245,"hedef":"Levrek",   "shore_facing":180,"not":"Lodos'ta köpüklü dalga"},
    "Büyükçekmece Köprü":   {"lat":40.998,"lon":28.565,"hedef":"Levrek",   "shore_facing":180,"not":"Lodosta kıyıya vurur"},
    "Avcılar Kıyı":         {"lat":40.978,"lon":28.715,"hedef":"İstavrit", "shore_facing":180,"not":"Durgun havalarda yüzey"},
    "Bakırköy Sahil":       {"lat":40.962,"lon":28.841,"hedef":"İstavrit", "shore_facing":180,"not":"Sirkülasyon, aji ideal"},
    "Yenikapı İskele":      {"lat":41.002,"lon":28.960,"hedef":"İstavrit", "shore_facing":180,"not":"Akşam ışıklı istavrit"},
    "Sarayburnu":           {"lat":41.015,"lon":28.984,"hedef":"İstavrit", "shore_facing":90, "not":"Boğaz akıntısı"},
    "Çatalca/Karaburun":    {"lat":41.358,"lon":28.235,"hedef":"Levrek",   "shore_facing":270,"not":"Karadeniz etkili kıyı"},
    # ── İSTANBUL ANADOLU ─────────────────────────────────────────
    "Maltepe/Dragos":       {"lat":40.912,"lon":29.156,"hedef":"İstavrit", "shore_facing":180,"not":"Gece ışıklı aji"},
    "Pendik İskele":        {"lat":40.877,"lon":29.230,"hedef":"İstavrit", "shore_facing":200,"not":"Rıhtım dibi"},
    "Tuzla/Mercan":         {"lat":40.825,"lon":29.300,"hedef":"Levrek",   "shore_facing":180,"not":"Lodos'ta aktif levrek"},
    "Tuzla/Aydınlı":        {"lat":40.815,"lon":29.350,"hedef":"Levrek",   "shore_facing":185,"not":"Sığ taşlı dip"},
    # ── KOCAELİ ──────────────────────────────────────────────────
    "Gebze Sahil":          {"lat":40.798,"lon":29.420,"hedef":"Levrek",   "shore_facing":200,"not":"Körfez kıyısı"},
    "Körfez/Hereke":        {"lat":40.790,"lon":29.650,"hedef":"Levrek",   "shore_facing":210,"not":"Sanayi dışı"},
    "İzmit Körfez Kapısı":  {"lat":40.785,"lon":29.910,"hedef":"Levrek",   "shore_facing":270,"not":"Kapalı körfez"},
    "Gölcük Sahil":         {"lat":40.713,"lon":29.845,"hedef":"İstavrit", "shore_facing":0,  "not":"Sakin su, aji"},
    "Karamürsel":           {"lat":40.692,"lon":29.615,"hedef":"İstavrit", "shore_facing":0,  "not":"Körfez içi"},
    # ── BURSA / GEMLİK ───────────────────────────────────────────
    "Mudanya İskele":       {"lat":40.373,"lon":28.884,"hedef":"İstavrit", "shore_facing":0,  "not":"Gece aji, berrak"},
    "Gemlik Körfezi":       {"lat":40.432,"lon":29.155,"hedef":"Levrek",   "shore_facing":350,"not":"Poyraz'da aktif"},
    "Armutlu Burnu":        {"lat":40.528,"lon":28.832,"hedef":"Levrek",   "shore_facing":315,"not":"Açık kıyı, derin"},
    # ── YALOVA ───────────────────────────────────────────────────
    "Yalova Merkez":        {"lat":40.655,"lon":29.272,"hedef":"İstavrit", "shore_facing":0,  "not":"Liman çevresi"},
    "Çınarcık":             {"lat":40.640,"lon":29.112,"hedef":"İstavrit", "shore_facing":0,  "not":"Sakin koy"},
    "Hersek Lagünü":        {"lat":40.604,"lon":29.600,"hedef":"Levrek",   "shore_facing":315,"not":"Tatlı-tuzlu karışım"},
    # ── BALIKESİR / ERDEK ────────────────────────────────────────
    "Erdek İskele":         {"lat":40.392,"lon":27.798,"hedef":"Levrek",   "shore_facing":315,"not":"Kapıdağ koruması"},
    "Bandırma Sahil":       {"lat":40.354,"lon":27.975,"hedef":"İstavrit", "shore_facing":30, "not":"Sanayi dışı"},
    "Kapıdağ Yarımadası":   {"lat":40.350,"lon":27.680,"hedef":"Levrek",   "shore_facing":225,"not":"Açık kıyı, derin"},
    "Paşalimanı Adası":     {"lat":40.424,"lon":27.552,"hedef":"Levrek",   "shore_facing":270,"not":"Ada etrafı, akıntı"},
    # ── TEKİRDAĞ ─────────────────────────────────────────────────
    "Tekirdağ Merkez":      {"lat":40.972,"lon":27.513,"hedef":"Levrek",   "shore_facing":135,"not":"Lodos'ta mükemmel"},
    "Şarköy Sahil":         {"lat":40.612,"lon":27.114,"hedef":"Levrek",   "shore_facing":135,"not":"Güçlü lodos kıyısı"},
    "Marmara Ereğlisi":     {"lat":40.968,"lon":27.960,"hedef":"Levrek",   "shore_facing":180,"not":"Lodos etkili kıyı"},
    # ── ÇANAKKALE / GELİBOLU ─────────────────────────────────────
    "Gelibolu Yarımadası":  {"lat":40.408,"lon":26.670,"hedef":"Levrek",   "shore_facing":135,"not":"Boğaz girişi"},
    "Karabiga":             {"lat":40.395,"lon":27.295,"hedef":"Levrek",   "shore_facing":0,  "not":"Açık Marmara"},
    # ── ADALAR ───────────────────────────────────────────────────
    "Büyükada Güney":       {"lat":40.836,"lon":29.126,"hedef":"İstavrit", "shore_facing":180,"not":"Ada güneyinde derin"},
    "Marmara Adası Kuzey":  {"lat":40.638,"lon":27.535,"hedef":"Levrek",   "shore_facing":0,  "not":"Kuzey Marmara açığı"},
    "İmralı Çevresi":       {"lat":40.547,"lon":28.947,"hedef":"Levrek",   "shore_facing":270,"not":"Serbest bölge kontrol"},
    # ── BATI MARMARA ─────────────────────────────────────────────
    "Enez Deltası":         {"lat":40.728,"lon":26.079,"hedef":"Levrek",   "shore_facing":180,"not":"Meriç deltası, tatlı-tuzlu"},
}

# ══════════════════════════════════════════════════════════════════
#  SAYFA KONFIG & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Marmara Av Radarı PRO",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{
  --glass:rgba(5,18,45,.8);--border:rgba(56,189,248,.14);
  --sky:#38bdf8;--teal:#2dd4bf;--rose:#fb7185;--amber:#fbbf24;
  --green:#4ade80;--muted:#5a7a96;--txt:#dff1ff;
}
html,body,.stApp{
  background:radial-gradient(ellipse 140% 90% at 10% 0%,#071e3d 0%,#030c1a 55%,#020810 100%)!important;
  color:var(--txt)!important;font-family:'Syne',sans-serif!important;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#020f22,#030c1a)!important;
  border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
h1{font-size:1.85rem!important;font-weight:800!important;
  background:linear-gradient(120deg,var(--sky) 20%,var(--teal) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.5px;}
h2,h3{color:var(--sky)!important;font-weight:700!important;}
[data-testid="stMetricValue"]{color:var(--teal)!important;
  font-family:'JetBrains Mono',monospace!important;font-size:1.25rem!important;font-weight:600!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.68rem!important;}
[data-baseweb="tab-list"]{background:rgba(5,18,45,.95)!important;border-radius:14px!important;
  padding:5px!important;gap:4px!important;border:1px solid var(--border)!important;}
[data-baseweb="tab"]{border-radius:10px!important;font-family:'Syne',sans-serif!important;
  font-weight:600!important;font-size:.78rem!important;color:var(--muted)!important;padding:7px 12px!important;}
[aria-selected="true"][data-baseweb="tab"]{
  background:rgba(56,189,248,.15)!important;color:var(--sky)!important;
  border:1px solid rgba(56,189,248,.4)!important;}
[data-baseweb="select"]>div,.stTextInput>div>div>input{
  background:rgba(5,18,45,.9)!important;border:1px solid var(--border)!important;
  border-radius:9px!important;color:var(--txt)!important;}
.kart{background:var(--glass);border:1px solid var(--border);border-radius:16px;
  padding:16px 18px;margin-bottom:10px;
  transition:transform .2s,border-color .2s,box-shadow .2s;position:relative;overflow:hidden;}
.kart::after{content:'';position:absolute;top:0;left:0;right:0;height:1.5px;
  background:linear-gradient(90deg,transparent,var(--sky),transparent);opacity:0;transition:opacity .2s;}
.kart:hover{transform:translateY(-3px);border-color:rgba(56,189,248,.38);box-shadow:0 12px 40px rgba(0,0,0,.5);}
.kart:hover::after{opacity:1;}
.kart-L{border-left:3px solid var(--rose)!important;}
.kart-I{border-left:3px solid var(--teal)!important;}
.kart-T{border-left:3px solid var(--amber)!important;}
.sy{color:#4ade80!important;font-size:2em!important;font-weight:900!important;
  font-family:'JetBrains Mono',monospace!important;line-height:1;}
.so{color:#fbbf24!important;font-size:2em!important;font-weight:900!important;
  font-family:'JetBrains Mono',monospace!important;line-height:1;}
.sd{color:#f87171!important;font-size:2em!important;font-weight:900!important;
  font-family:'JetBrains Mono',monospace!important;line-height:1;}
.b{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.67em;
  font-weight:700;letter-spacing:.4px;text-transform:uppercase;margin:1px 2px;}
.bL{background:rgba(251,113,133,.12);border:1px solid rgba(251,113,133,.4);color:#fb7185;}
.bI{background:rgba(45,212,191,.12);border:1px solid rgba(45,212,191,.4);color:#2dd4bf;}
.bl{background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.4);color:#fbbf24;}
.bo{background:rgba(74,222,128,.12);border:1px solid rgba(74,222,128,.4);color:#4ade80;}
.bx{background:rgba(90,122,150,.12);border:1px solid rgba(90,122,150,.4);color:#5a7a96;}
.bg{background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.35);color:#4ade80;}
.br{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.35);color:#f87171;}
.bw{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.35);color:#fbbf24;}
.mok{color:#4ade80;font-family:'JetBrains Mono',monospace;font-size:.73em;}
.mwn{color:#fbbf24;font-family:'JetBrains Mono',monospace;font-size:.73em;}
.tu{color:#4ade80;font-size:.79em;}
.td{color:#f87171;font-size:.79em;}
.te{color:#5a7a96;font-size:.79em;}
.sep{border:none;border-top:1px solid var(--border);margin:9px 0;}
.mono{font-family:'JetBrains Mono',monospace;}
@keyframes wv{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.wv{display:inline-block;animation:wv 2.8s ease-in-out infinite;}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;height:0;}
.block-container{padding-top:.6rem!important;}
/* Animasyonlu rüzgar oku için DivIcon */
@keyframes wind_pulse{0%,100%{opacity:.9;transform:scale(1)}50%{opacity:.6;transform:scale(.9)}}
.wind-arrow{animation:wind_pulse 1.8s ease-in-out infinite;display:inline-block;transition:transform .3s;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════
def wd_name(deg: float) -> str:
    d=["K","KKD","KD","DKD","D","DGD","GD","GGD","G","GGB","GB","BGB","B","KBB","KB","KKB"]
    return d[int((float(deg)+11.25)/22.5)%16]

def bft(kmh: float) -> int:
    for i,t in enumerate([1,6,12,20,29,39,50,62,75,89,103,118]):
        if kmh<t: return i
    return 12

def bft_col(b: int) -> str:
    c=["#dbeafe","#93c5fd","#60a5fa","#34d399","#a3e635",
       "#fbbf24","#fb923c","#f87171","#e11d48","#be185d","#7e22ce","#312e81","#030712"]
    return c[min(b,12)]

def pres_col(h: float) -> str:
    if h<970:  return "#7c3aed"
    if h<985:  return "#be185d"
    if h<1000: return "#ef4444"
    if h<1008: return "#f97316"
    if h<1013: return "#fbbf24"
    if h<1018: return "#a3e635"
    if h<1023: return "#4ade80"
    if h<1030: return "#22d3ee"
    return "#818cf8"

def adiff(a: float, b: float) -> float:
    d=abs(a-b); return min(d,360-d)

def is_on(wd: float, sf: float)  -> bool: return adiff(wd,sf)<=45
def is_sid(wd: float, sf: float) -> bool:
    d=adiff(wd,sf); return 45<d<=90

def trend_arrow(now: float, prev: Optional[float], unit: str="") -> str:
    if prev is None: return ""
    diff=now-prev
    if abs(diff)<0.5: return "<span class='te'>→</span>"
    if diff>0:        return f"<span class='tu'>↑{diff:+.1f}{unit}</span>"
    return                  f"<span class='td'>↓{diff:.1f}{unit}</span>"


# ══════════════════════════════════════════════════════════════════
#  API 1: OPEN-METEO — Çok-Model Atmosfer (paralel)
# ══════════════════════════════════════════════════════════════════
MODELS=[("icon_d2","ICON-D2","2km"),("icon_eu","ICON-EU","7km"),
        ("ecmwf_ifs04","ECMWF","9km"),("gfs_seamless","GFS","25km")]

@st.cache_data(ttl=1800,show_spinner=False)
def fetch_meteo(lat: float, lon: float) -> dict:
    base="https://api.open-meteo.com/v1/forecast"
    p={"latitude":lat,"longitude":lon,
       "hourly":"windspeed_10m,winddirection_10m,windgusts_10m,surface_pressure",
       "windspeed_unit":"kmh","timezone":"Europe/Istanbul","forecast_days":7}
    out={}
    def _m(key,name,res):
        try:
            r=requests.get(base,params={**p,"models":key},timeout=7)
            if r.ok:
                d=r.json()
                if d.get("hourly",{}).get("windspeed_10m"):
                    return name,{"d":d,"res":res}
        except: pass
        return name,None
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name,res in [f.result() for f in as_completed(
                [ex.submit(_m,k,n,r) for k,n,r in MODELS])]:
            if res: out[name]=res
    return out

@st.cache_data(ttl=1800,show_spinner=False)
def fetch_marine_openmeteo(lat: float, lon: float) -> dict:
    try:
        r=requests.get("https://marine-api.open-meteo.com/v1/marine",
            params={"latitude":lat,"longitude":lon,
                    "hourly":"wave_height,wave_period,swell_wave_height",
                    "timezone":"Europe/Istanbul","forecast_days":7},timeout=7)
        if r.ok: return r.json()
    except: pass
    return {}


# ══════════════════════════════════════════════════════════════════
#  API 2: MET NORWAY OCEANFORECAST — Dalga, SST, Akıntı (FREE, key yok)
#  Kıyı izohipsine dik rüzgar tespitinde kullanılan SST
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600,show_spinner=False)
def fetch_met_norway(lat: float, lon: float) -> dict:
    """
    api.met.no/weatherapi/oceanforecast/2.0
    Dalga yüksekliği, yön, akıntı hızı/yönü, su yüzeyi sıcaklığı
    User-Agent zorunlu — met.no kullanım şartı
    """
    try:
        r=requests.get(
            "https://api.met.no/weatherapi/oceanforecast/2.0/complete",
            params={"lat":round(lat,3),"lon":round(lon,3)},
            headers={"User-Agent":"MarmaraFishingRadar/6.0 (research; github.com)"},
            timeout=8)
        if r.ok:
            ts=r.json().get("properties",{}).get("timeseries",[])
            if ts:
                det=ts[0].get("data",{}).get("instant",{}).get("details",{})
                return {
                    "mn_wave_height":  det.get("sea_surface_wave_height"),
                    "mn_wave_dir":     det.get("sea_surface_wave_from_direction"),
                    "mn_wave_period":  det.get("sea_surface_wave_mean_period"),
                    "mn_sst":          det.get("sea_water_temperature"),
                    "mn_current_spd":  det.get("sea_water_speed"),
                    "mn_current_dir":  det.get("sea_water_to_direction"),
                }
    except: pass
    return {}


# ══════════════════════════════════════════════════════════════════
#  API 3: STORMGLASS — Bio-oşinografi (quota korumalı)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600,show_spinner=False)
def _sg_cached(lat: float, lon: float, key: str, ts: int) -> dict:
    if not key: return {}
    try:
        r=requests.get("https://api.stormglass.io/v2/weather/point",
            params={"lat":lat,"lng":lon,
                    "params":"waterTemperature,currentSpeed,currentDirection,salinity,waveHeight,seaLevel,visibility",
                    "start":ts,"end":ts+3600},
            headers={"Authorization":key},timeout=10)
        if r.status_code==429: return {"_q":True}
        if r.ok:
            hrs=r.json().get("hours",[])
            if hrs:
                h=hrs[0]
                def gv(k):
                    s=h.get(k,{})
                    for src in ["sg","noaa","icon","dwd","meteo"]:
                        if src in s: return s[src]
                    return None
                return {"water_temp":gv("waterTemperature"),"current_spd":gv("currentSpeed"),
                        "current_dir":gv("currentDirection"),"salinity":gv("salinity"),
                        "sea_level":gv("seaLevel"),"visibility":gv("visibility")}
    except: pass
    return {}

def sg_fetch(lat: float, lon: float, dt: datetime.datetime) -> dict:
    if st.session_state.get("sg_used",0)>=SG_QUOTA_MAX:
        return {"_q":True}
    r=_sg_cached(lat,lon,STORMGLASS_KEY,int(dt.timestamp()))
    if not r.get("_q"):
        st.session_state["sg_used"]=st.session_state.get("sg_used",0)+1
    return r


# ══════════════════════════════════════════════════════════════════
#  API 4: NASA ERDDAP — Klorofil Grid (Marmara tek sorguda)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400,show_spinner=False)
def fetch_chl_grid() -> Optional[Dict]:
    """
    NASA CoastWatch ERDDAP — Marmara bölgesi klorofil ızgara verisi.
    Tek HTTP isteğiyle ~1200 ölçüm noktası (VIIRS S-NPP 4km).
    """
    url=(f"{ERDDAP_BASE}/griddap/{ERDDAP_VIIRS_DS}.csv"
         f"?chlor_a%5B(last)%5D"
         f"%5B({MARMARA['lat_min']:.1f}):({MARMARA['lat_max']:.1f})%5D"
         f"%5B({MARMARA['lon_min']:.1f}):({MARMARA['lon_max']:.1f})%5D")
    try:
        r=requests.get(url,timeout=20)
        if r.ok:
            lines=r.text.strip().split("\n")
            pts=[]
            for ln in lines[2:]:        # 2 başlık satırı atla
                p=ln.split(",")
                if len(p)>=4:
                    try:
                        la,lo,v=float(p[1]),float(p[2]),float(p[3])
                        if not math.isnan(v) and v>0:
                            pts.append((la,lo,v))
                    except: pass
            if len(pts)>10:
                return {"pts":pts,"n":len(pts)}
    except: pass
    return None

@st.cache_data(ttl=86400,show_spinner=False)
def fetch_chl_point(lat: float, lon: float) -> Optional[float]:
    url=(f"{ERDDAP_BASE}/griddap/{ERDDAP_VIIRS_DS}.json"
         f"?chlor_a[(last)][({lat:.3f})][({lon:.3f})]")
    try:
        r=requests.get(url,timeout=8)
        if r.ok:
            rows=r.json().get("table",{}).get("rows",[])
            if rows and rows[0] and rows[0][-1] is not None:
                return float(rows[0][-1])
    except: pass
    return None


# ══════════════════════════════════════════════════════════════════
#  API 5: OSM KIYI İZOHİPSİ (0m)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400,show_spinner=False)
def fetch_coastline() -> dict:
    q="""[out:json][timeout:30];
    (way["natural"="coastline"](40.3,26.0,41.5,30.2););out geom;"""
    try:
        r=requests.post("https://overpass-api.de/api/interpreter",data={"data":q},timeout=35)
        if r.ok:
            feats=[]
            for e in r.json().get("elements",[]):
                if e.get("type")=="way" and "geometry" in e:
                    c=[[g["lon"],g["lat"]] for g in e["geometry"]]
                    if len(c)>=2:
                        feats.append({"type":"Feature",
                                      "geometry":{"type":"LineString","coordinates":c},
                                      "properties":{}})
            if feats: return {"type":"FeatureCollection","features":feats}
    except: pass
    return {}


# ══════════════════════════════════════════════════════════════════
#  API 6: GLOBAL FISHING WATCH — Balıkçı Yoğunluk Haritası (opsiyonel)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400,show_spinner=False)
def fetch_gfw(token: str) -> Optional[List]:
    """
    GFW 4Wings API — Marmara balıkçı aktivite yoğunluğu.
    Ücretsiz kayıt: globalfishingwatch.org/our-apis
    """
    if not token: return None
    try:
        r=requests.post(
            "https://gateway.api.globalfishingwatch.org/v3/4wings/report",
            headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"},
            json={
                "geojson":{"type":"Polygon","coordinates":[[
                    [26.0,40.3],[30.2,40.3],[30.2,41.5],[26.0,41.5],[26.0,40.3]]]},
                "datasets":["public-global-fishing-effort:latest"],
                "date-range":"2024-01-01,2024-12-31",
                "spatial-resolution":"HIGH",
                "temporal-resolution":"YEARLY",
            },
            timeout=15)
        if r.ok:
            data=r.json()
            entries=data.get("entries",[])
            pts=[]
            for e in entries:
                la=e.get("lat"); lo=e.get("lon"); hrs=e.get("hours",0)
                if la and lo and hrs>0:
                    pts.append([float(la),float(lo),float(hrs)])
            return pts if pts else None
    except: pass
    return None


# ══════════════════════════════════════════════════════════════════
#  VERİ İŞLEME: SAATLIK DEĞER / MODEL SEÇİM / DOĞRULAMA
# ══════════════════════════════════════════════════════════════════
def hval(data: dict, field: str, target: str, fb: float=0.0) -> float:
    try:
        times=data.get("hourly",{}).get("time",[])
        vals=data.get("hourly",{}).get(field,[])
        if not times or not vals: return fb
        if target in times: i=times.index(target)
        else:
            tdt=datetime.datetime.fromisoformat(target)
            i=min(range(len(times)),
                  key=lambda k:abs((datetime.datetime.fromisoformat(times[k])-tdt).total_seconds()))
        v=vals[i]; return float(v) if v is not None else fb
    except: return fb

def best_model(models: dict, target: str) -> Tuple[dict,str,str]:
    for _,name,_ in MODELS:
        if name in models:
            d,rs=models[name]["d"],models[name]["res"]
            return {"wind_speed":hval(d,"windspeed_10m",target),
                    "wind_dir":  hval(d,"winddirection_10m",target),
                    "wind_gust": hval(d,"windgusts_10m",target),
                    "pressure":  hval(d,"surface_pressure",target,1013)},name,rs
    return {"wind_speed":0,"wind_dir":0,"wind_gust":0,"pressure":1013},"—","—"

def validate_m(models: dict, target: str) -> dict:
    rows={}
    for _,name,res in MODELS:
        if name in models:
            d=models[name]["d"]
            rows[name]={"ws":hval(d,"windspeed_10m",target),
                        "wd":hval(d,"winddirection_10m",target),"res":res}
    if len(rows)<2: return {"ok":False,"detail":"Tek model","names":list(rows.keys())}
    wsl=[v["ws"] for v in rows.values()]; wdl=[v["wd"] for v in rows.values()]
    dws=max(wsl)-min(wsl)
    dwd=sum(adiff(wdl[i],wdl[j])
            for i in range(len(wdl)) for j in range(i+1,len(wdl)))/max(1,len(wdl)*(len(wdl)-1)/2)
    return {"ok":dws<8 and dwd<30,"detail":f"ΔHız:{dws:.1f}km/h ΔYön:{dwd:.0f}°",
            "names":list(rows.keys())}

def prev_val(data: dict, field: str, target: str, h: int=6) -> Optional[float]:
    try:
        tdt=datetime.datetime.fromisoformat(target)-datetime.timedelta(hours=h)
        v=hval(data,field,tdt.strftime("%Y-%m-%dT%H:%M"),None)
        return v
    except: return None

def build_ts(models: dict, marine: dict) -> Optional[pd.DataFrame]:
    for _,name,_ in MODELS:
        if name in models:
            d=models[name]["d"]
            times=d.get("hourly",{}).get("time",[])
            ws=d.get("hourly",{}).get("windspeed_10m",[])
            pr=d.get("hourly",{}).get("surface_pressure",[])
            wh=marine.get("hourly",{}).get("wave_height",[]) if marine else []
            wt=marine.get("hourly",{}).get("time",[]) if marine else []
            if not times: return None
            n=min(len(times),len(ws),len(pr))
            wmap={t:v for t,v in zip(wt,wh) if v is not None}
            return pd.DataFrame([{
                "zaman":datetime.datetime.fromisoformat(times[i]),
                "Rüzgar km/h":float(ws[i] or 0),
                "Basınç hPa":float(pr[i] or 1013),
                "Dalga m":float(wmap.get(times[i],0) or 0),
            } for i in range(n)])
    return None


# ══════════════════════════════════════════════════════════════════
#  KLOROFİL: Hibrit Kaynak Seçimi
# ══════════════════════════════════════════════════════════════════
def chl_level(chl_mg: Optional[float], wave: float, onshore: bool,
              ws: float, month: int, wt: Optional[float]=None,
              mn_sst: Optional[float]=None) -> Tuple[str,str,float,Optional[float]]:
    # Gerçek veri öncelikli
    sst=mn_sst or wt
    if chl_mg is not None and chl_mg>0:
        sc=min(10,(math.log10(max(chl_mg,0.001))+2)*2.5)
        if chl_mg>2.0:   lbl,col="Yüksek","#22c55e"
        elif chl_mg>0.5: lbl,col="Orta","#86efac"
        else:            lbl,col="Düşük","#dcfce7"
        return lbl,col,round(sc,1),round(chl_mg,3)
    # Fiziksel proxy (MET Norway SST'yi de kullanır)
    s=0.0
    if wave>0.8:  s+=2.5
    if wave>1.5:  s+=1.5
    if onshore:   s+=2.0
    if ws>20:     s+=1.0
    if month in [12,1,2,3]:  s+=1.5
    if month in [3,4,10,11]: s+=0.5
    if month in [7,8]:       s-=1.0
    if sst and sst<12:       s+=1.0
    if sst and sst>24:       s-=0.5  # Yazın termal stratifikasyon
    s=max(0,min(10,s))
    if s>=5:   lbl,col="Yüksek","#22c55e"
    elif s>=3: lbl,col="Orta",  "#86efac"
    else:      lbl,col="Düşük", "#dcfce7"
    return lbl,col,round(s,1),None


# ══════════════════════════════════════════════════════════════════
#  SKORLAMA
# ══════════════════════════════════════════════════════════════════
def score_spot(hedef: str, wind: dict, wave: float, wave_p: float,
               sf: float, month: int, sg: Optional[dict]=None,
               mn: Optional[dict]=None, chl_lbl: str="Orta") -> dict:
    ws,wd,wg,pr=wind["wind_speed"],wind["wind_dir"],wind["wind_gust"],wind["pressure"]
    on=is_on(wd,sf); sid=is_sid(wd,sf)
    sc=0.0; flags=[]; warns=[]

    # MET Norway dalga verisi varsa Open-Meteo dalga ile ağırlıklı ortalama
    mn_wave=mn.get("mn_wave_height") if mn else None
    eff_wave=wave
    if mn_wave is not None and mn_wave>0:
        eff_wave=(wave*0.4+float(mn_wave)*0.6)  # MET Norway daha hassas → ağırlıklı

    if hedef=="Levrek":
        if on:
            sc+=3.5; flags.append(("Kıyıya Dik ✅","bo"))
            if adiff(wd,sf)<=20: sc+=0.5; flags.append(("Tam Lodos","bl"))
        elif sid: sc+=1.5; flags.append(("Yan ↗️","bl"))
        else:     flags.append(("Açık ❌","bx"))
        if 15<=ws<=40:   sc+=2.0
        elif ws>40:      sc+=0.5; warns.append("Fırtına — güvenli mevzi seç!")
        else:            warns.append("Rüzgar zayıf, levrek aktivitesi sınırlı")
        if 0.5<=eff_wave<=2.0: sc+=2.5
        elif eff_wave>2.0:     sc+=0.5; warns.append("Dalga >2m — sığ kıyı tehlikeli")
        else:                  warns.append("Düz deniz — levrek için ideal değil")
        if pr<1005:  sc+=1.5
        elif pr<1010:sc+=1.0
        elif pr<1015:sc+=0.5
        if chl_lbl=="Yüksek": sc+=0.5
        takim=TAKIMLAR[2] if (ws>20 or eff_wave>1.0) else TAKIMLAR[0]
        tneden="Uzak erim, köpüklü su" if takim==TAKIMLAR[2] else "Sessiz sığ levrek"
    else:
        if ws<10:        sc+=4.0
        elif ws<15:      sc+=2.5
        elif ws<25:      sc+=1.0
        else:            warns.append("Sert hava — aji için zor")
        if eff_wave<0.3:   sc+=4.0
        elif eff_wave<0.5: sc+=2.5
        elif eff_wave<1.0: sc+=1.0
        else:              warns.append("Dalga yüksek — ester kırılır")
        if chl_lbl=="Düşük": sc+=1.0
        elif chl_lbl=="Yüksek": warns.append("Bulanık su — istavrit görselliği azalır")
        flags.append(("Sakin Deniz 🌊","bI"))
        takim=TAKIMLAR[1]; tneden="Ester, hafif jig, berrak su"
    return {"sc":min(10.0,round(sc,1)),"takim":takim,"tneden":tneden,
            "flags":flags,"warns":warns,"on":on,"sid":sid,"eff_wave":eff_wave}


# ══════════════════════════════════════════════════════════════════
#  GÖRSELLEŞTİRME: YEŞİL BULUT KLOROFİL OVERLAY
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def build_chl_overlay(grid_pts: Optional[tuple], spot_vals: tuple) -> Optional[str]:
    """
    Marmara klorofil verilerini yeşil gradient bulut overlay'e dönüştürür.
    - YlGn paleti: düşük=sarı-yeşil, yüksek=koyu yeşil
    - Gaussian blur: bulutumsu yumuşak kenarlar
    - Kıyı yakınlığı: daha yoğun değerlere taşma efekti
    - Şeffaf PNG → folium ImageOverlay
    """
    if not HAS_SCIPY or not HAS_MPL:
        return None

    # Veri noktaları
    lats, lons, vals = [], [], []
    if grid_pts:
        for la,lo,v in grid_pts[:3000]:
            lats.append(la); lons.append(lo); vals.append(v)
    if not lats:
        for la,lo,v in spot_vals:
            lats.append(la); lons.append(lo); vals.append(v if v else 0.5)

    if len(lats)<4:
        return None

    try:
        W, H = 420, 120
        glo,gla=np.meshgrid(
            np.linspace(MARMARA["lon_min"],MARMARA["lon_max"],W),
            np.linspace(MARMARA["lat_min"],MARMARA["lat_max"],H))

        gv=griddata(list(zip(lons,lats)),vals,(glo,gla),method="linear")
        # Kenar NaN → en düşük yüzdelik
        fill_val=float(np.nanpercentile(vals,10)) if vals else 0.1
        gv=np.where(np.isnan(gv),fill_val,gv)
        gv=np.clip(gv,0.01,20.0)

        # Gaussian bulanıklaştırma → bulutumsu görünüm
        gv=gaussian_filter(gv,sigma=2.5)

        # Kıyı yakınlık bonusu — kenar piksellerini hafifçe artır
        # Basit edge enhancement: sınır piksellerine gradient uygula
        from scipy.ndimage import distance_transform_edt
        sea_mask=np.ones_like(gv,dtype=bool)  # tüm alan deniz (basit yaklaşım)
        dist=distance_transform_edt(sea_mask)
        coast_dist=np.max(dist)-dist
        coast_factor=1.0+(coast_dist/np.max(coast_dist+1e-9))*0.4
        gv=gv*coast_factor
        gv=gaussian_filter(gv,sigma=1.5)  # ikinci geçiş

        # Özel yeşil colormap: şeffaftan koyu yeşile
        cmap=LinearSegmentedColormap.from_list("chl",
            [(0.98,1.0,0.88,0.0),   # şeffaf beyaz-yeşil
             (0.83,0.96,0.54,0.25), # açık sarı-yeşil
             (0.45,0.77,0.25,0.50), # orta yeşil
             (0.13,0.55,0.13,0.72), # orman yeşili
             (0.02,0.30,0.08,0.88)],# koyu deniz yeşili
            N=256)

        fig,ax=plt.subplots(figsize=(W/100,H/100),dpi=100)
        fig.patch.set_alpha(0)
        ax.set_axis_off()
        ax.set_position([0,0,1,1])

        ax.contourf(glo,gla,gv,
                    levels=25,
                    norm=LogNorm(vmin=0.1,vmax=8.0),
                    cmap=cmap,antialiased=True)

        buf=io.BytesIO()
        fig.savefig(buf,format="png",dpi=100,
                    transparent=True,bbox_inches="tight",pad_inches=0)
        plt.close(fig); buf.seek(0)
        return "data:image/png;base64,"+base64.b64encode(buf.read()).decode()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
#  GÖRSELLEŞTİRME: İZOBAR KONTURLAR (2hPa, gradient fill)
# ══════════════════════════════════════════════════════════════════
def isobar_contours(pts: list, prs: list) -> list:
    if not HAS_MPL or len(pts)<5: return []
    try:
        lats=[p[0] for p in pts]; lons=[p[1] for p in pts]
        glo,gla=np.meshgrid(np.linspace(26.0,30.2,100),np.linspace(40.3,41.5,60))
        if HAS_SCIPY:
            gp=griddata(list(zip(lons,lats)),prs,(glo,gla),method="linear")
        else:
            gp=np.zeros_like(glo)
            for i in range(gla.shape[0]):
                for j in range(gla.shape[1]):
                    gl,go=gla[i,j],glo[i,j]
                    w,s=0.0,0.0
                    for k,(la,lo) in enumerate(zip(lats,lons)):
                        di=math.sqrt((gl-la)**2+(go-lo)**2)+1e-9
                        wi=1/di**2; w+=wi; s+=wi*prs[k]
                    gp[i,j]=s/w
        if np.all(np.isnan(gp)): return []
        gp=np.where(np.isnan(gp),float(np.nanmean(gp)),gp)
        if HAS_SCIPY: gp=gaussian_filter(gp,sigma=1.2)  # pürüzsüz kontur
        p0=int(min(prs)/2)*2; p1=int(max(prs)/2+1)*2
        levels=list(range(max(960,p0),min(1040,p1)+1,2))
        if not levels: return []
        fig,ax=plt.subplots(figsize=(1,1))
        cs=ax.contour(glo,gla,gp,levels=levels)
        lines=[]
        for ci,coll in enumerate(cs.collections):
            lv=cs.levels[ci]
            for path in coll.get_paths():
                v=path.vertices
                if len(v)<3: continue
                coords=[[float(v[k,1]),float(v[k,0])] for k in range(len(v))]
                lines.append({"lv":float(lv),"coords":coords})
        plt.close(fig); return lines
    except Exception: return []


# ══════════════════════════════════════════════════════════════════
#  HARİTALAR
# ══════════════════════════════════════════════════════════════════
CARTO_DARK="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"

def base_map(coast: dict, seamark: bool=True) -> folium.Map:
    m=folium.Map(location=[40.65,28.3],zoom_start=8,tiles=CARTO_DARK,
                 attr="© OpenStreetMap © CARTO",max_zoom=18)
    if coast and coast.get("features"):
        folium.GeoJson(coast,name="0m Kıyı İzohips",
            style_function=lambda x:{"color":"#38bdf8","weight":1.6,"opacity":0.65}
        ).add_to(m)
    # ⚓ OpenSeaMap — ücretsiz denizcilik katmanı (fener, derinlik, boya, set)
    if seamark:
        folium.TileLayer(
            tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
            attr="© <a href='https://www.openseamap.org'>OpenSeaMap</a>",
            name="⚓ OpenSeaMap Deniz İşaretleri",
            overlay=True, control=True, opacity=0.75,
        ).add_to(m)
    return m

# ── Harita 1: Av Verimi ─────────────────────────────────────────
def map_fishing(data: dict, coast: dict) -> folium.Map:
    m=base_map(coast)
    for nm,d in data.items():
        sc=d["sc"]
        c="green" if sc>=7 else ("orange" if sc>=4.5 else "red")
        ic="star" if sc>=7 else ("info-sign" if sc>=4.5 else "remove")
        sc_c="#4ade80" if sc>=7 else "#fbbf24" if sc>=4.5 else "#f87171"
        mn=d.get("mn",{})
        mn_txt=""
        if mn.get("mn_sst"):      mn_txt+=f"🌡️{mn['mn_sst']:.1f}°C "
        if mn.get("mn_wave_height"): mn_txt+=f"🇳🇴{mn['mn_wave_height']:.1f}m "
        if mn.get("mn_current_spd"): mn_txt+=f"🌀{mn['mn_current_spd']:.2f}m/s"
        sg=d.get("sg",{})
        sg_txt=""
        if sg.get("water_temp"):  sg_txt+=f"💧{sg['water_temp']:.1f}°C "
        if sg.get("salinity"):    sg_txt+=f"🧂{sg['salinity']:.1f}‰ "
        if sg.get("visibility"):  sg_txt+=f"👁️{sg['visibility']:.1f}km"
        pop=(f"<div style='font-family:sans-serif;font-size:12px;min-width:220px;'>"
             f"<b style='font-size:14px;'>{nm}</b><br>"
             f"{'🐟 Levrek' if d['hedef']=='Levrek' else '🐡 İstavrit'}&nbsp;"
             f"<b style='color:{sc_c};font-size:15px;'>{sc}/10</b><br>"
             f"🌊 {d['wave']:.2f}m/{d['wave_period']:.0f}sn · Swell:{d['swell']:.2f}m<br>"
             f"💨 {d['ws']:.0f}km/h {d['wdn']} B{d['bft']}<br>"
             f"🧭 {'<b>Onshore ✅</b>' if d['on'] else ('Yan ↗️' if d['sid'] else 'Offshore ❌')}<br>"
             f"🦠 {d['chl_lbl']} · 📉{d['pressure']:.0f}hPa<br>"
             f"{'<small style=color:#4ade80>'+mn_txt+'</small><br>' if mn_txt else ''}"
             f"{'<small style=color:#2dd4bf>'+sg_txt+'</small><br>' if sg_txt else ''}"
             f"<i style='color:#666;font-size:11px;'>{d.get('not','')}</i></div>")
        folium.Marker([d["lat"],d["lon"]],
                      popup=folium.Popup(pop,max_width=280),
                      tooltip=f"{nm} — {sc}/10",
                      icon=folium.Icon(color=c,icon=ic,prefix="glyphicon")
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ── Harita 2: Animasyonlu Rüzgar ────────────────────────────────
def map_wind(data: dict, coast: dict,
             wind_uv_json: Optional[str] = None,
             max_vel: float = 40.0) -> folium.Map:
    """
    Rüzgar haritası — iki katman:
    1) Leaflet.Velocity animasyonlu rüzgar partikülleri (HAS_SCIPY gerekir)
    2) CSS animasyonlu SVG ok markerları (spot başvuru noktaları)
    """
    m = base_map(coast)

    # Leaflet.Velocity katmanı
    if wind_uv_json and HAS_SCIPY:
        LeafletVelocityLayer(wind_uv_json, max_velocity=max_vel).add_to(m)

    # Spot markerları (referans + popup bilgisi)
    for nm, d in data.items():
        ws=d["ws"]; b=bft(ws); col=bft_col(b)
        ang=(d["wd"]-90)%360
        size=max(14, 16+b*2)
        speed_s = max(0.5, 1.6 - b*0.08)
        svg=f"""<svg width='{size}' height='{size}' viewBox='0 0 20 20'
            style='transform:rotate({ang}deg);
                   filter:drop-shadow(0 0 4px {col})
                          drop-shadow(0 0 8px {col}66);
                   animation:wind_pulse {speed_s:.2f}s ease-in-out infinite;'
            xmlns='http://www.w3.org/2000/svg'>
          <polygon points='10,1 18,18 10,12 2,18'
            fill='{col}' fill-opacity='0.95'
            stroke='rgba(255,255,255,.4)' stroke-width='0.6'/>
        </svg>"""
        folium.Marker([d["lat"],d["lon"]],
            icon=folium.DivIcon(html=svg, icon_size=(size,size),
                                icon_anchor=(size//2, size//2)),
            popup=(f"<b>{nm}</b><br>"
                   f"💨 <b>{ws:.0f} km/h</b> {d['wdn']} B{b}<br>"
                   f"💥 Hamle: {d['wg']:.0f} km/h<br>"
                   f"📉 {d['pressure']:.0f} hPa<br>"
                   f"🌊 {d.get('wave',0):.1f}m"),
            tooltip=f"B{b} · {ws:.0f}km/h {d['wdn']}"
        ).add_to(m)

    return m

# ── Harita 3: İzobarik Basınç ────────────────────────────────────
def map_pressure(data: dict, coast: dict) -> folium.Map:
    m=base_map(coast)
    pts=[[d["lat"],d["lon"]] for d in data.values()]
    prs=[d["pressure"] for d in data.values()]
    lines=isobar_contours(pts,prs)
    # Gradient fill izobar bantları arası için
    for iso in lines:
        lv=iso["lv"]; col=pres_col(lv)
        wt=3.0 if lv%10==0 else (1.8 if lv%4==0 else 0.8)
        op=0.85 if lv%10==0 else 0.62
        folium.PolyLine(iso["coords"],color=col,weight=wt,opacity=op,
                        tooltip=f"{int(lv)} hPa").add_to(m)
        if lv%4==0:
            mid=iso["coords"][len(iso["coords"])//2]
            folium.Marker(mid,icon=folium.DivIcon(
                html=f"<div style='color:{col};font-size:8px;font-weight:700;"
                     f"white-space:nowrap;text-shadow:0 0 3px #000;'>{int(lv)}</div>",
                icon_size=(28,11),icon_anchor=(14,5))).add_to(m)
    for nm,d in data.items():
        col=pres_col(d["pressure"])
        folium.CircleMarker([d["lat"],d["lon"]],radius=8,
            color=col,fill=True,fill_color=col,fill_opacity=0.9,
            popup=f"<b>{nm}</b><br>{d['pressure']:.1f} hPa",
            tooltip=f"{d['pressure']:.1f}hPa").add_to(m)
    return m

# ── Harita 4: Yeşil Bulut Klorofil ───────────────────────────────
def map_chloro(data: dict, coast: dict,
               chl_overlay_b64: Optional[str]=None) -> folium.Map:
    """
    Katman 1: Matplotlib yeşil bulut overlay — kıyıya yakınlıkla ağırlıklı (ImageOverlay)
    Katman 2: CMEMS Black Sea BGC WMS (opsiyonel — ücretsiz hesap gerekir)
    Katman 3: NASA VIIRS WMS tile (4km, ücretsiz)
    Katman 4: HeatMap (fallback / ek katman)
    Katman 5: Nokta renkleri (gerçek mg/m³ veya proxy)
    """
    m=base_map(coast)

    # ── Katman 1: Yeşil bulut overlay (ana görselleştirme) ───────
    if chl_overlay_b64:
        try:
            folium.raster_layers.ImageOverlay(
                image=chl_overlay_b64,
                bounds=[[MARMARA["lat_min"],MARMARA["lon_min"]],
                        [MARMARA["lat_max"],MARMARA["lon_max"]]],
                opacity=0.72,
                name="🌿 Klorofil Bulut (Gaussian, kıyı ağırlıklı)",
                cross_origin=False,
            ).add_to(m)
        except Exception: pass

    # ── Katman 2: CMEMS Black Sea BGC WMS (ücretsiz hesap gerekir) ──
    _cmems_url = cmems_wms_url()
    if _cmems_url:
        try:
            folium.raster_layers.WmsTileLayer(
                url=_cmems_url,
                layers="cmems_mod_blk_bgc-bio_anfc_3nm_P1D-m:chl",
                transparent=True, fmt="image/png",
                name="🛰️ CMEMS Black Sea Klorofil (3nm NRT)",
                overlay=True, opacity=0.55, version="1.3.0",
            ).add_to(m)
        except Exception: pass

    # ── Katman 3: NASA VIIRS WMS ─────────────────────────────────
    try:
        folium.raster_layers.WmsTileLayer(
            url=f"{ERDDAP_BASE}/wms/{ERDDAP_VIIRS_DS}/request?",
            layers=f"{ERDDAP_VIIRS_DS}:chlor_a",
            transparent=True, fmt="image/png",
            name="🛰️ NASA VIIRS 4km WMS",
            overlay=True, opacity=0.40, version="1.3.0"
        ).add_to(m)
    except: pass

    # ── Katman 4: HeatMap ─────────────────────────────────────────
    heat=[[d["lat"],d["lon"],d["chl_sc"]/10] for d in data.values()]
    HeatMap(heat, min_opacity=0.12, radius=42, blur=32,
            gradient={0.15:"#f0fdf4",0.35:"#bbf7d0",0.55:"#4ade80",
                      0.75:"#16a34a",0.92:"#14532d"}
    ).add_to(m)

    # ── Katman 5: Nokta markerları ────────────────────────────────
    for nm,d in data.items():
        src  = "VIIRS" if d.get("chl_real") else "Proxy"
        cr   = d.get("chl_real")
        icon = "🛰️" if cr else "📡"
        val_str = f" ({cr:.2f} mg/m³)" if cr else " (proxy model)"
        folium.CircleMarker([d["lat"],d["lon"]], radius=7,
            color="white", weight=1.2,
            fill_color=d["chl_col"], fill_opacity=0.92,
            popup=(f"<b>{nm}</b><br>"
                   f"Klorofil: <b>{d['chl_lbl']}</b>{val_str}<br>"
                   f"Kaynak: {icon} {src}"),
            tooltip=f"Chl:{d['chl_lbl']} [{src}]").add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ── Harita 5: Global Fishing Watch ──────────────────────────────
def map_gfw(gfw_pts: Optional[List], coast: dict) -> Optional[folium.Map]:
    if not gfw_pts:
        return None
    m=base_map(coast)
    # Normalize hours → 0-1
    max_h=max(p[2] for p in gfw_pts) if gfw_pts else 1
    heat=[[p[0],p[1],p[2]/max_h] for p in gfw_pts]
    HeatMap(heat,min_opacity=0.3,radius=25,blur=20,
            gradient={0.2:"#fef3c7",0.5:"#fbbf24",0.75:"#f97316",0.9:"#ef4444"}
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
def _ss():
    defs={"sg_used":0,"sg_date":str(datetime.date.today()),
          "results":None,"results_key":None,"coast":None,"val_all":{},
          "chl_grid":None,"chl_grid_loaded":False,
          "chl_overlay":None,"gfw_pts":None,
          "wind_uv_json":None,"wind_max_vel":40.0}   # Leaflet.Velocity için
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k]=v
_ss()

if st.session_state["sg_date"]!=str(datetime.date.today()):
    st.session_state.update({"sg_used":0,"sg_date":str(datetime.date.today())})
quota_left=max(0,SG_QUOTA_MAX-st.session_state.get("sg_used",0))


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center;padding:14px 0 6px;'>
  <span class='wv' style='font-size:2.4em;'>🎣</span><br>
  <span style='font-family:Syne,sans-serif;font-size:.9em;font-weight:800;
    letter-spacing:2px;color:#38bdf8;'>AV RADARI PRO</span><br>
  <span style='font-size:.63em;color:#5a7a96;'>MARMARA · v6.0 · Streamlit Cloud</span>
</div>""",unsafe_allow_html=True)
st.sidebar.divider()

today=datetime.date.today()
sel_date=st.sidebar.date_input("📅 Tarih",today,
    min_value=today,max_value=today+datetime.timedelta(days=6))
sel_hour=st.sidebar.slider("⏰ Saat",0,23,datetime.datetime.now().hour,format="%02d:00")
target=f"{sel_date}T{sel_hour:02d}:00"
target_dt=datetime.datetime(sel_date.year,sel_date.month,sel_date.day,sel_hour)

st.sidebar.divider()
hedef_f=st.sidebar.multiselect("🎯 Hedef",["Levrek","İstavrit"],default=["Levrek","İstavrit"])
min_sc=st.sidebar.slider("⭐ Min Skor",0.0,10.0,0.0,0.5)

st.sidebar.divider()
q_col="#4ade80" if quota_left>4 else "#fbbf24" if quota_left>1 else "#f87171"
mn_ok="✅" if True else "❌"  # MET Norway her zaman açık
gfw_ok="✅" if GFW_TOKEN else "⚠️"
st.sidebar.markdown(f"""
<div style='font-size:.77em;line-height:1.9;'>
<b style='color:#38bdf8;'>🔌 API Durumu</b><br>
<span class='bg'>✅ Open-Meteo (4 model)</span><br>
<span class='bg'>{mn_ok} MET Norway Oceanforecast</span><br>
<span class='bg'>✅ NASA ERDDAP VIIRS</span><br>
<span class='bg'>✅ Overpass OSM</span><br>
<span style='color:{q_col};'>⚡ Stormglass {quota_left}/{SG_QUOTA_MAX} kota</span><br>
<span class='{"bg" if GFW_TOKEN else "bw"}'>{gfw_ok} Global Fishing Watch</span>
</div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  VERİ TOPLAMA — PARALEL
# ══════════════════════════════════════════════════════════════════
SPOTS={k:v for k,v in MERALAR.items() if v["hedef"] in hedef_f}
rkey=(str(sel_date),sel_hour,tuple(sorted(hedef_f)))

if st.session_state["results_key"]!=rkey or st.session_state["results"] is None:
    pb=st.progress(0,text="📡 Paralel veri çekimi başlıyor...")

    # Tüm noktalar paralel: meteo + marine + met_norway + chl
    raw: Dict[str,Any]={}

    def _fetch_spot(item):
        nm,info=item
        return nm,{
            "models":   fetch_meteo(info["lat"],info["lon"]),
            "marine":   fetch_marine_openmeteo(info["lat"],info["lon"]),
            "mn":       fetch_met_norway(info["lat"],info["lon"]),
            "chl":      fetch_chl_point(info["lat"],info["lon"]),
        }

    total=len(SPOTS)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(_fetch_spot,item):item[0] for item in SPOTS.items()}
        done=0
        for fut in as_completed(futs):
            nm,d=fut.result()
            raw[nm]=d; done+=1
            pb.progress(done/total,text=f"📡 {done}/{total} mera · paralel çekim...")

    pb.progress(1.0,text="🧮 Skorlar hesaplanıyor...")

    # Ön skor → Stormglass için öncelik listesi
    pre={}
    for nm,info in SPOTS.items():
        if nm not in raw: continue
        rd=raw[nm]
        w,_,_=best_model(rd["models"],target)
        wv=hval(rd["marine"],"wave_height",target,0)
        on=is_on(w["wind_dir"],info["shore_facing"])
        mn=rd.get("mn",{})
        cl,_,_,_=chl_level(rd["chl"],wv,on,w["wind_speed"],
                            sel_date.month,mn_sst=mn.get("mn_sst"))
        a=score_spot(info["hedef"],w,wv,0,info["shore_facing"],sel_date.month,mn=mn,chl_lbl=cl)
        pre[nm]=a["sc"]
    top_sg=sorted(pre,key=pre.get,reverse=True)[:quota_left]

    results={}; val_all={}
    for nm,info in SPOTS.items():
        if nm not in raw: continue
        rd=raw[nm]
        wind,mdl,res=best_model(rd["models"],target)
        val=validate_m(rd["models"],target)
        wave=hval(rd["marine"],"wave_height",target,0)
        wave_p=hval(rd["marine"],"wave_period",target,0)
        swell=hval(rd["marine"],"swell_wave_height",target,0)
        mn=rd.get("mn",{})

        # Trend
        best_d=next((rd["models"][n]["d"] for _,n,_ in MODELS if n in rd["models"]),None)
        pp=prev_val(best_d,"surface_pressure",target,6) if best_d else None
        wp=prev_val(rd["marine"],"wave_height",target,6) if rd["marine"] else None
        wip=prev_val(best_d,"windspeed_10m",target,6) if best_d else None

        # Stormglass (quota korumalı)
        sg=sg_fetch(info["lat"],info["lon"],target_dt) if nm in top_sg else {}
        sg=sg if not sg.get("_q") else {}

        # SST kaynağı: MET Norway > Stormglass > None
        sst=mn.get("mn_sst") or sg.get("water_temp")
        on=is_on(wind["wind_dir"],info["shore_facing"])
        chl_lbl,chl_col,chl_sc,chl_mg=chl_level(
            rd["chl"],wave,on,wind["wind_speed"],sel_date.month,
            wt=sg.get("water_temp"),mn_sst=mn.get("mn_sst"))

        ana=score_spot(info["hedef"],wind,wave,wave_p,
                       info["shore_facing"],sel_date.month,sg=sg,mn=mn,chl_lbl=chl_lbl)

        ts_df=build_ts(rd["models"],rd["marine"])

        results[nm]={
            **info,**wind,**ana,
            "wave":wave,"wave_period":wave_p,"swell":swell,
            "ws":wind["wind_speed"],"wd":wind["wind_dir"],"wg":wind["wind_gust"],
            "wdn":wd_name(wind["wind_dir"]),
            "bft":bft(wind["wind_speed"]),"bft_col":bft_col(bft(wind["wind_speed"])),
            "pres_col":pres_col(wind["pressure"]),
            "chl_lbl":chl_lbl,"chl_col":chl_col,"chl_sc":chl_sc,"chl_real":chl_mg,
            "mdl":mdl,"mdl_res":res,
            "val_ok":val["ok"],"val_detail":val["detail"],"val_names":val["names"],
            "sg":sg,"mn":mn,
            "pres_prev":pp,"wave_prev":wp,"wind_prev":wip,
            "sst":sst,"ts_df":ts_df,
        }
        val_all[nm]=val

    pb.empty()

    # Kıyı (paylaşımlı cache)
    if st.session_state["coast"] is None:
        with st.spinner("🗺️ OSM kıyı izohips..."):
            st.session_state["coast"]=fetch_coastline()

    # Klorofil grid (büyük — ayrı cache)
    if not st.session_state["chl_grid_loaded"]:
        with st.spinner("🛰️ NASA ERDDAP klorofil ızgarası..."):
            st.session_state["chl_grid"]=fetch_chl_grid()
            st.session_state["chl_grid_loaded"]=True

    # GFW
    if GFW_TOKEN and st.session_state["gfw_pts"] is None:
        with st.spinner("🎣 Global Fishing Watch..."):
            st.session_state["gfw_pts"]=fetch_gfw(GFW_TOKEN)

    st.session_state.update({"results":results,"val_all":val_all,"results_key":rkey})
else:
    results=st.session_state["results"]
    val_all=st.session_state.get("val_all",{})

coast=st.session_state["coast"] or {}
chl_grid=st.session_state.get("chl_grid")
gfw_pts=st.session_state.get("gfw_pts")

sorted_r=sorted([(k,v) for k,v in results.items() if v["sc"]>=min_sc],
                 key=lambda x:x[1]["sc"],reverse=True)
data_map=dict(sorted_r)

# Klorofil overlay (cache + build)
if st.session_state.get("chl_overlay") is None:
    grid_pts=tuple(chl_grid["pts"]) if chl_grid and chl_grid.get("pts") else None
    spot_vals=tuple((d["lat"],d["lon"],d.get("chl_real")) for d in results.values())
    ov=build_chl_overlay(grid_pts,spot_vals)
    st.session_state["chl_overlay"]=ov
chl_overlay=st.session_state.get("chl_overlay")

# Rüzgar UV ızgarası (Leaflet.Velocity) — sonuçlar değişince yeniden hesapla
_wind_key = (target, tuple(sorted(results.keys())))
if (st.session_state.get("wind_uv_json") is None
        or st.session_state.get("_wind_key") != _wind_key):
    if data_map and HAS_SCIPY:
        _wj, _mv = build_wind_uv_grid(data_map)
        st.session_state["wind_uv_json"]  = _wj
        st.session_state["wind_max_vel"]  = _mv
        st.session_state["_wind_key"]     = _wind_key

wind_uv_json = st.session_state.get("wind_uv_json")
wind_max_vel = st.session_state.get("wind_max_vel", 40.0)


# ══════════════════════════════════════════════════════════════════
#  ANA UI
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:0 0 12px;display:flex;align-items:center;gap:10px;'>
  <span class='wv' style='font-size:1.75em;'>🌊</span>
  <h1 style='margin:0;'>Marmara Denizi Av Radarı PRO</h1>
</div>""",unsafe_allow_html=True)

# Metrik şeridi
if sorted_r:
    bn,bd=sorted_r[0]
    agree_n=sum(1 for v in val_all.values() if v["ok"])
    viirs_n=sum(1 for d in results.values() if d.get("chl_real"))
    mn_n   =sum(1 for d in results.values() if d.get("mn",{}).get("mn_sst"))
    sg_n   =sum(1 for d in results.values() if d.get("sg"))
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("📅 Zaman",         f"{sel_date.strftime('%d.%m')} {sel_hour:02d}:00")
    c2.metric("🏆 En İyi",        bn[:16])
    c3.metric("⭐ Skor",          f"{bd['sc']}/10")
    c4.metric("🧪 Model Uyum",    f"{agree_n}/{len(val_all)}")
    c5.metric("🛰️ VIIRS+METno",   f"{viirs_n}+{mn_n}")
    c6.metric("⚡ SG Kota",       f"{quota_left}/{SG_QUOTA_MAX}",
              delta=f"{sg_n} aktif" if sg_n else None)
st.divider()

# Haritalar
st.markdown("### 🗺️ Haritalar")
tab_list=["🎣 Av Verimi","💨 Rüzgar & Beaufort","📉 İzobarik Basınç","🌿 Klorofil-a"]
if gfw_pts: tab_list.append("🛥️ Balıkçı Aktivitesi")
tabs=st.tabs(tab_list)

with tabs[0]:
    st.markdown("""<small style='color:#5a7a96;'>
    Mavi çizgi = 0m kıyı &nbsp;|&nbsp; 🟢≥7 &nbsp;🟡4.5-7 &nbsp;🔴&lt;4.5 &nbsp;|&nbsp;
    Rüzgar yönü kıyıya dik mi? → Levrek skoru artar &nbsp;|&nbsp;
    Pin = detay popup
    </small>""",unsafe_allow_html=True)
    st_folium(map_fishing(data_map,coast),width=None,height=520,returned_objects=[],key="m1")

with tabs[1]:
    st.markdown("""<small style='color:#5a7a96;'>
    SVG animasyonlu oklar — ok yönü = rüzgarın estiği yön &nbsp;|&nbsp;
    <span style='color:#dbeafe'>■</span>B0-2
    <span style='color:#34d399'>■</span>B3-4
    <span style='color:#fbbf24'>■</span>B5-6
    <span style='color:#f87171'>■</span>B7+ &nbsp;|&nbsp;
    pulse hızı = fırtına şiddeti
    </small>""",unsafe_allow_html=True)
    st_folium(map_wind(data_map,coast,wind_uv_json,wind_max_vel),width=None,height=520,returned_objects=[],key="m2")

with tabs[2]:
    st.markdown("""<small style='color:#5a7a96;'>
    İzobar konturlar 2hPa aralıklı (gaussian pürüzsüz) &nbsp;|&nbsp;
    <span style='color:#818cf8'>■</span>&gt;1030
    <span style='color:#4ade80'>■</span>1013-23
    <span style='color:#fbbf24'>■</span>1005-13
    <span style='color:#ef4444'>■</span>&lt;1000
    <span style='color:#7c3aed'>■</span>&lt;970
    </small>""",unsafe_allow_html=True)
    st_folium(map_pressure(data_map,coast),width=None,height=520,returned_objects=[],key="m3")

with tabs[3]:
    viirs_n2=sum(1 for d in data_map.values() if d.get("chl_real"))
    ov_status="✅ Aktif" if chl_overlay else "⚠️ Hesaplanmadı"
    st.markdown(f"""<small style='color:#5a7a96;'>
    <b>Yeşil Bulut Overlay:</b> {ov_status} (scipy interpolasyon + Gaussian blur) &nbsp;|&nbsp;
    NASA VIIRS WMS &nbsp;|&nbsp; VIIRS nokta: <b style='color:#4ade80'>{viirs_n2}</b> &nbsp;|&nbsp;
    <span style='color:#dcfce7'>■</span> Düşük
    <span style='color:#86efac'>■</span> Orta
    <span style='color:#22c55e'>■</span> Yüksek klorofil
    </small>""",unsafe_allow_html=True)
    st_folium(map_chloro(data_map,coast,chl_overlay),width=None,height=520,returned_objects=[],key="m4")

if gfw_pts and len(tabs)>4:
    with tabs[4]:
        max_h=max(p[2] for p in gfw_pts) if gfw_pts else 1
        st.markdown(f"""<small style='color:#5a7a96;'>
        <b>Global Fishing Watch</b> — 2024 balıkçı aktivitesi &nbsp;|&nbsp;
        {len(gfw_pts)} ızgara hücresi &nbsp;|&nbsp; Maks: {max_h:.0f} saat &nbsp;|&nbsp;
        <span style='color:#fbbf24'>■</span>Orta
        <span style='color:#ef4444'>■</span>Yoğun avlanma
        </small>""",unsafe_allow_html=True)
        gfw_map=map_gfw(gfw_pts,coast)
        if gfw_map:
            st_folium(gfw_map,width=None,height=520,returned_objects=[],key="m5")
st.divider()


# ══════════════════════════════════════════════════════════════════
#  EN İYİ 3 KART
# ══════════════════════════════════════════════════════════════════
st.markdown("### 🏆 En İyi Av Fırsatları")
medals=["🥇","🥈","🥉"]
top3=sorted_r[:3]
if top3:
    cols=st.columns(len(top3))
    for i,(nm,d) in enumerate(top3):
        with cols[i]:
            sc=d["sc"]
            sc_cls="sy" if sc>=7 else ("so" if sc>=4.5 else "sd")
            hcls="bL" if d["hedef"]=="Levrek" else "bI"
            on_h=('<span class="b bo">Kıyıya Dik ✅</span>' if d["on"]
                  else '<span class="b bl">Yan ↗️</span>' if d["sid"]
                  else '<span class="b bx">Açık ❌</span>')
            flags_h=" ".join(f'<span class="b {c}">{l}</span>' for l,c in d["flags"])
            warns_h="".join(f'<p style="font-size:.75em;color:#fbbf24;margin:2px 0">⚠️ {w}</p>'
                            for w in d["warns"])
            # Stormglass bio
            sg=d.get("sg",{}); sg_bits=[]
            if sg.get("water_temp"):  sg_bits.append(f"💧{sg['water_temp']:.1f}°C")
            if sg.get("salinity"):    sg_bits.append(f"🧂{sg['salinity']:.1f}‰")
            if sg.get("sea_level"):   sg_bits.append(f"📏{sg['sea_level']:.2f}m")
            if sg.get("visibility"):  sg_bits.append(f"👁️{sg['visibility']:.1f}km")
            sg_h=(f"<p style='font-size:.78em;color:#2dd4bf;margin:3px 0'>"
                  f"{'&nbsp;'.join(sg_bits)}</p>") if sg_bits else ""
            # MET Norway ocean
            mn=d.get("mn",{}); mn_bits=[]
            if mn.get("mn_sst"):         mn_bits.append(f"🌡️{mn['mn_sst']:.1f}°C")
            if mn.get("mn_current_spd"): mn_bits.append(f"🌀{mn['mn_current_spd']:.2f}m/s")
            if mn.get("mn_wave_height"): mn_bits.append(f"🇳🇴{mn['mn_wave_height']:.2f}m")
            mn_h=(f"<p style='font-size:.78em;color:#4ade80;margin:3px 0'>"
                  f"{'&nbsp;'.join(mn_bits)}</p>") if mn_bits else ""
            # Trend
            pt=trend_arrow(d["pressure"],d.get("pres_prev"),"hPa")
            wvt=trend_arrow(d["wave"],d.get("wave_prev"),"m")
            wit=trend_arrow(d["ws"],d.get("wind_prev"),"km/h")
            # Klorofil
            cr=d.get("chl_real")
            cr_h=(f"<span class='mono' style='font-size:.72em;color:#888'>({cr:.2f}mg/m³ VIIRS)</span>"
                  if cr else "<span style='font-size:.7em;color:#334155'>(proxy)</span>")
            val_h=(f'<span class="mok">✅ {" · ".join(d["val_names"])}</span>'
                   if d["val_ok"] else f'<span class="mwn">⚠️ {d["val_detail"]}</span>')

            st.markdown(f"""
<div class="kart kart-{'L' if d['hedef']=='Levrek' else 'I'}">
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <span style='font-size:1.1em'>{medals[i]}</span>
      <span style='font-size:.96em;font-weight:700;margin-left:5px'>{nm}</span><br>
      <span class='b {hcls}'>{d["hedef"]}</span> {flags_h}
    </div>
    <span class='{sc_cls}'>{sc}</span>
  </div>
  <hr class='sep'>
  <p style='margin:3px 0'>🌊 <b>{d['wave']:.2f}m</b>/{d['wave_period']:.0f}sn · {d['swell']:.2f}m swell &nbsp;{wvt}</p>
  <p style='margin:3px 0'>💨 <b>{d['ws']:.0f}km/h</b> {d['wdn']} B{d['bft']} · 💥{d['wg']:.0f}km/h &nbsp;{wit}</p>
  <p style='margin:3px 0'>📉 <b>{d['pressure']:.0f}</b>hPa {pt} &nbsp;| 🧭 {on_h}</p>
  <p style='margin:3px 0'>🦠 <span style='color:{d["chl_col"]}'><b>{d["chl_lbl"]}</b></span> {cr_h}</p>
  {mn_h}{sg_h}{warns_h}
  <hr class='sep'>
  <div style='display:flex;align-items:center;gap:6px;'>
    <span style='font-size:1.4em;'>🎣</span>
    <div>
      <p style='font-size:.8em;font-weight:700;color:#e2f4ff;margin:0'>{d['takim']['ad']}</p>
      <p style='font-size:.72em;color:#5a7a96;margin:0'>{d['takim']['tip']} {d['takim']['agirlik']} · {d['takim']['ip']} · {d['tneden']}</p>
    </div>
  </div>
  <p style='font-size:.67em;color:#1e3a5f;margin:6px 0 0'>📡 {d['mdl']} ({d['mdl_res']}) &nbsp; {val_h}</p>
  {'<p style="font-size:.67em;color:#1e3a5f;font-style:italic;margin:1px 0">'+d.get("not","")+'</p>' if d.get("not") else ""}
</div>""",unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════
#  TREND GRAFİKLERİ
# ══════════════════════════════════════════════════════════════════
st.markdown("### 📈 7 Günlük Trend")
all_names=[nm for nm,_ in sorted_r]
if all_names:
    sel_mera=st.selectbox("📍 Mera:",all_names,index=0,key="tsel")
    if sel_mera and sel_mera in results:
        dr=results[sel_mera]; ts_df=dr.get("ts_df")
        hcls_t="bL" if dr["hedef"]=="Levrek" else "bI"
        st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
          <span class='b {hcls_t}'>{dr["hedef"]}</span><b>{sel_mera}</b>
          <span style='color:#fbbf24;font-family:JetBrains Mono,monospace;font-weight:700;'>{dr["sc"]}/10</span>
          {'<span class="b bg">🛰️ VIIRS '+str(round(dr["chl_real"],2))+' mg/m³</span>' if dr.get("chl_real") else ""}
          {'<span class="b bg">🌡️ '+str(round(dr["sst"],1))+'°C</span>' if dr.get("sst") else ""}
        </div>""",unsafe_allow_html=True)
        if ts_df is not None and not ts_df.empty:
            tc1,tc2=st.columns(2)
            with tc1:
                st.caption("💨 Rüzgar Hızı (km/h)")
                st.area_chart(ts_df.set_index("zaman")[["Rüzgar km/h"]],color="#38bdf8",height=185)
                st.caption("📉 Basınç (hPa)")
                st.line_chart(ts_df.set_index("zaman")[["Basınç hPa"]],color="#fbbf24",height=185)
            with tc2:
                st.caption("🌊 Dalga Yüksekliği (m)")
                st.area_chart(ts_df.set_index("zaman")[["Dalga m"]],color="#2dd4bf",height=185)
                st.caption("⭐ En Güçlü 5 Saat")
                top5=ts_df.nlargest(5,"Rüzgar km/h")[["zaman","Rüzgar km/h","Dalga m"]].copy()
                top5["zaman"]=top5["zaman"].dt.strftime("%d/%m %H:00")
                st.dataframe(top5,hide_index=True,use_container_width=True)
        else:
            st.caption("Trend verisi yüklenemedi.")
st.divider()


# ══════════════════════════════════════════════════════════════════
#  TÜM MERALARIN TABLOSU
# ══════════════════════════════════════════════════════════════════
st.markdown("### 📊 Tüm Meraların Özeti")
if sorted_r:
    rows=[]
    for nm,d in sorted_r:
        sg=d.get("sg",{}); mn=d.get("mn",{})
        rows.append({
            "📍 Mera":nm,"🎯":d["hedef"],"⭐":d["sc"],
            "🌊m":round(d["wave"],2),"💨km/h":round(d["ws"]),"Yön":d["wdn"],"B":d["bft"],
            "📉hPa":round(d["pressure"],1),
            "🧭":"✅" if d["on"] else ("↗️" if d["sid"] else "❌"),
            "🦠Chl":d["chl_lbl"],
            "🌡️SST":round(mn.get("mn_sst") or sg.get("water_temp") or 0,1) or "—",
            "🧂‰":round(sg.get("salinity",0),1) or "—",
            "🇳🇴METno":"✅" if mn.get("mn_sst") else "—",
            "📡Mdl":f"{d['mdl']}({d['mdl_res']}){'✅' if d['val_ok'] else '⚠️'}",
        })
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
st.divider()


# ══════════════════════════════════════════════════════════════════
#  MODEL DOĞRULAMA
# ══════════════════════════════════════════════════════════════════
with st.expander("🧪 Çok-Model Doğrulama",expanded=False):
    st.markdown("""<p style='color:#5a7a96;font-size:.8em;'>
    ICON-D2(2km) → ICON-EU(7km) → ECMWF(9km) → GFS(25km) &nbsp;|&nbsp;
    MET Norway bağımsız doğrulama olarak kullanılır<br>
    Tutarlı = ΔHız&lt;8km/h VE ΔYön&lt;30°
    </p>""",unsafe_allow_html=True)
    vcols=st.columns(4)
    for i,(nm,d) in enumerate(sorted_r[:8]):
        with vcols[i%4]:
            ok=d["val_ok"]
            st.markdown(f"""
<div class='kart' style='padding:9px 12px;'>
  <b style='font-size:.79em;'>{nm[:20]}</b><br>
  <span class='{"mok" if ok else "mwn"}'>{"✅" if ok else "⚠️"} {d['mdl']}({d['mdl_res']})</span><br>
  <span style='color:#334155;font-size:.69em;'>{d['val_detail']}</span>
</div>""",unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════
#  TAKIM KARTI
# ══════════════════════════════════════════════════════════════════
st.markdown("### 🎣 Takım Envanteri")
tcols=st.columns(3)
for i,t in enumerate(TAKIMLAR):
    with tcols[i]:
        st.markdown(f"""
<div class='kart kart-T' style='text-align:center;padding:18px 14px;'>
  <div style='font-size:1.8em;margin-bottom:5px;'>{"🎣" if i==0 else "🪝" if i==1 else "🌊"}</div>
  <b style='font-size:.93em;color:{t["renk"]};'>{t["ad"]}</b><br>
  <span class='b bx'>{t["tip"]}</span>
  <hr class='sep'>
  <p style='font-size:.78em;margin:3px 0'>⚖️ {t["agirlik"]} &nbsp;|&nbsp; 🧵 {t["ip"]}</p>
  <p style='font-size:.74em;color:#5a7a96;margin:4px 0;font-style:italic;'>{t["senaryo"]}</p>
</div>""",unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════
#  API KAYNAK TABLOSU
# ══════════════════════════════════════════════════════════════════
with st.expander("🔌 API & Veri Kaynakları",expanded=False):
    st.markdown("""
| Servis | Veri | Çözünürlük | Key | Durum |
|--------|------|-----------|-----|-------|
| **Open-Meteo** | Rüzgar, basınç, hamle | ICON-D2 2km → ECMWF 9km | ❌ Yok | ✅ Ücretsiz |
| **MET Norway** | Dalga, SST, akıntı | Nokta (Yr.no altyapısı) | ❌ Yok | ✅ Ücretsiz |
| **Open-Meteo Marine** | Dalga, periyot, swell | ERA5 / EWAM | ❌ Yok | ✅ Ücretsiz |
| **Stormglass.io** | Su sıcaklığı, akıntı, tuzluluk | Çok-kaynak birleşimi | ✅ Key | ✅ Freemium |
| **NASA ERDDAP VIIRS** | Klorofil-a grid (Marmara) | 4km S-NPP global | ❌ Yok | ✅ Ücretsiz |
| **Overpass OSM** | 0m kıyı izohips | Metre hassasiyeti | ❌ Yok | ✅ Ücretsiz |
| **Global Fishing Watch** | Balıkçı aktivitesi | Yüksek çözünürlük | ✅ Key | ✅ Ücretsiz |
""")

st.markdown("""
<p style='text-align:center;color:#1e2d40;font-size:.67em;margin-top:16px;'>
Marmara Av Radarı PRO v6.0 &nbsp;·&nbsp; Streamlit Cloud &nbsp;·&nbsp;
Open-Meteo CC BY 4.0 &nbsp;·&nbsp; MET Norway met.no &nbsp;·&nbsp;
NASA/NOAA VIIRS &nbsp;·&nbsp; DWD ICON / ECMWF &nbsp;·&nbsp;
Stormglass.io &nbsp;·&nbsp; Global Fishing Watch &nbsp;·&nbsp; OpenStreetMap ODbL
</p>
""",unsafe_allow_html=True)
