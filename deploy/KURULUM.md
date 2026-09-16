# GBM-AID — sunucu kurulum rehberi

Bu rehber, **kiralanan bir Ubuntu VPS** üzerinde GBM-AID'i ayağa kaldırır.

> **Kendi bilgisayarına hiçbir şey kurmuyorsun.** Ubuntu sunucunun üzerinde
> çalışır; sen Windows'tan `ssh` ile bağlanıp komut yazarsın. Windows'ta `ssh`
> zaten kuruludur (PowerShell'de çalışır).

```
Senin bilgisayarın (Windows)  ──SSH──▶  VPS (Ubuntu 22.04)
  • kurulum yok                          • Python, kod, MR dosyaları burada
  • sadece komut yazıyorsun              • site burada çalışıyor
```

**Kararlar (2026-09-16, Barış):** site **şifresizdir** · Supabase/OpenAI
anahtarları **döndürülmeyecek** · sağlayıcı **Hostinger** (Ubuntu 22.04 veya
24.04 şablonu, VPS — paylaşımlı hosting değil).

---

## Ön gereksinimler

| | Ölçülen | Seçilecek |
|---|---|---|
| Bellek | tepe 653 MB | **≥ 4 GB** |
| Disk | kod 12 MB + MR 6,47 GB | **≥ 40 GB** |
| CPU | tek işçi yeter | 2 vCPU |
| 🔴 Kritik | soğuk başlangıç 19–540 sn | **idle-sleep / askıya alma OLMAYACAK** |

---

## 1 — Bağlan ve kullanıcı aç

Hostinger sana bir IP ve `root` şifresi verir.

```powershell
# Windows PowerShell'den
ssh root@SUNUCU-IP
```

Sunucuda (root olarak), servisin kendi kullanıcısını aç:

```bash
adduser --system --group --home /opt/gbmaid gbmaid
```

## 2 — Temel paketler

```bash
apt update && apt upgrade -y
apt install -y python3.10 python3.10-venv python3-pip git curl
```

Caddy (ters vekil + otomatik HTTPS):

```bash
apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | tee /etc/apt/sources.list.d/caddy-stable.list
apt update && apt install -y caddy
```

## 3 — Kodu çek

```bash
git clone https://github.com/gbm-aid/gbm-aid.git /opt/gbmaid
cd /opt/gbmaid
```

> Depo **herkese açıksa** bu komut şifresiz çalışır. Özelse bir
> *personal access token* sorar.

## 4 — Python ortamı

```bash
cd /opt/gbmaid
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

⚠️ `requirements-nnunet.txt` ve `requirements-pyradiomics.txt` **kurulmaz** —
ayrı ortamlar içindir ve `torch` sürümleri çakışır. nnU-Net yalnız yeni-hasta
demosunun fallback'idir ve varsayılan kapalıdır.

🟢 Burada Windows'ta **çalışmayan** şeyler çalışacak: `shap`, `numba`,
`llvmlite`, `SimpleITK`. Sebep: Smart App Control **Windows'a özgüdür**,
Linux'ta öyle bir engel yoktur. Yani `/predict`, SHAP ve XGBoost sunucuda
çalışır.

## 5 — MR dosyalarını kopyala (6,47 GB)

Bu en uzun adım; internet hızına göre **30 dk – 3 saat**.

Önce sunucuda klasörleri aç:

```bash
mkdir -p /srv/gbmaid/nas /srv/gbmaid/ucsf
chown -R gbmaid:gbmaid /srv/gbmaid
```

Sonra **kendi bilgisayarından** (PowerShell, yeni pencere):

```powershell
$KOK = "C:\Users\Barış\Desktop\GBM-AID Prototip"

# NAS tarafı (~5,61 GB): upenn-gbm, lumiere, tcga-gbm
scp -r "$KOK\<nas-klasoru>\upenn-gbm"  root@SUNUCU-IP:/srv/gbmaid/nas/
scp -r "$KOK\<nas-klasoru>\lumiere"    root@SUNUCU-IP:/srv/gbmaid/nas/
scp -r "$KOK\<nas-klasoru>\tcga-gbm"   root@SUNUCU-IP:/srv/gbmaid/nas/

# UCSF (~0,86 GB): içindeki UCSF-PDGM-XXXX_nifti klasörleri DOĞRUDAN
# /srv/gbmaid/ucsf altına gelmeli
scp -r "$KOK\PKG - UCSF-PDGM Version 5\UCSF-PDGM-v5\*" root@SUNUCU-IP:/srv/gbmaid/ucsf/
```

Kopya bitince sunucuda doğrula:

```bash
ls /srv/gbmaid/nas                     # upenn-gbm lumiere tcga-gbm
ls /srv/gbmaid/ucsf | head -3          # UCSF-PDGM-0004_nifti ...
du -sh /srv/gbmaid                     # ~6,5 GB
chown -R gbmaid:gbmaid /srv/gbmaid
```

## 6 — `.env` oluştur

```bash
cp /opt/gbmaid/deploy/env.sunucu.ornek /opt/gbmaid/.env
nano /opt/gbmaid/.env          # <...> yer tutucularını doldur
chmod 600 /opt/gbmaid/.env
chown gbmaid:gbmaid /opt/gbmaid/.env
```

Değerlerin çoğu geliştirme makinesindeki `.env` ile **aynıdır**. Sunucuda
değişen üçü:

```
GBMAID_NAS_ROOT=/srv/gbmaid/nas
GBMAID_UCSF_ROOT=/srv/gbmaid/ucsf
GBMAID_COX_CHECKPOINT_PATH=/opt/gbmaid/backend/models/cox_phm_v3b_lowvar_v2amgmt_2026-09-12.pkl
```

🔴 `.env` **asla git'e girmez** — `.gitignore` onu zaten dışarıda tutar.

## 7 — Kurulum öncesi doğrulama

Servisi başlatmadan önce yolların çözüldüğünü kontrol et:

```bash
cd /opt/gbmaid
sudo -u gbmaid PYTHONPATH=/opt/gbmaid/backend .venv/bin/python - <<'PY'
from api.main import app
import api.predict as P, api.similar as S
print("route sayısı      :", len([r for r in app.routes if hasattr(r, "path")]))
print("Cox checkpoint    :", P.DEFAULT_CHECKPOINT_PATH.exists())
print("FAISS klinik      :", S.DEFAULT_CLINICAL_RADIOMICS_INDEX_DIR.exists())
print("FAISS omics       :", S.DEFAULT_MOLECULAR_OMICS_INDEX_DIR.exists())
import db_connection as D; print("DB bağlantısı     :", bool(D.get_connection(readonly=True)))
PY
```

Hepsi `True` / 13+ route değilse **servisi başlatma**, önce yolu düzelt.

## 8 — Servisi kur

```bash
chown -R gbmaid:gbmaid /opt/gbmaid
cp /opt/gbmaid/deploy/gbmaid.service /etc/systemd/system/gbmaid.service
systemctl daemon-reload
systemctl enable --now gbmaid
systemctl status gbmaid
journalctl -u gbmaid -f        # ısıtma çağrısını canlı izle (Ctrl+C ile çık)
```

İlk açılış **ısıtma çağrısı** yüzünden birkaç dakika sürebilir — bu normaldir
ve ilk gerçek kullanıcının beklememesi için bilerek yapılır.

```bash
curl -s localhost:8000/health
```

## 9 — Alan adı + HTTPS

1. Alan adını al, DNS'te bir **A kaydı** aç: `@ → SUNUCU-IP`
2. Yayılmayı bekle (genelde dakikalar)
3. Caddy'yi kur:

```bash
cp /opt/gbmaid/deploy/Caddyfile /etc/caddy/Caddyfile
nano /etc/caddy/Caddyfile       # <ALAN-ADI> satırını doldur
systemctl reload caddy
journalctl -u caddy -f          # sertifika alımını izle
```

Sertifika **otomatik** gelir, elle bir şey yapmazsın.

> Alan adı henüz yoksa `Caddyfile`'ın sonundaki `:80` bloğunu kullanabilirsin —
> ama o **HTTPS'siz**dir ve jüriye verilecek adres değildir.

## 10 — Uçtan uca doğrulama

```bash
D=https://<ALAN-ADI>
curl -s   $D/health
curl -s   $D/patients?limit=1        | head -c 200
curl -s   $D/model_performance       | head -c 200
curl -s   $D/model_curves            | head -c 200
curl -s   $D/patient/Patient-002/similar | head -c 200
curl -s -o /dev/null -w "mr_slice %{http_code} %{size_download} bayt\n" \
          $D/patient/UCSF-PDGM-167/mr_slice
curl -s -X POST $D/predict/UCSF-PDGM-167 | head -c 300
curl -s -X POST $D/analyze_patient -H 'Content-Type: application/json' \
     -d '{"patient_id":"Patient-002"}' | head -c 300
```

Sonra tarayıcıda `https://<ALAN-ADI>/app/` — beş ekranı da gez.

🟢 **Windows'ta göremediğin şey burada görünmeli:** risk skoru, 24 SHAP çubuğu
ve XGBoost bloğu.

---

## Sorun giderme

| Belirti | Bak |
|---|---|
| `systemctl status gbmaid` → failed | `journalctl -u gbmaid -n 50` |
| `/predict` 503 | `GBMAID_COX_CHECKPOINT_PATH` doğru mu, dosya var mı |
| `/mr_slice` 503 | `GBMAID_NAS_ROOT` / `GBMAID_UCSF_ROOT`, klasör izinleri (`chown gbmaid`) |
| `/similar` 503 | `artifacts/week3/faiss_index` depoda yok → FAISS indeksi kopyalanmalı |
| Caddy sertifika alamıyor | DNS A kaydı yayılmış mı: `dig +short <ALAN-ADI>` |
| Site açılıyor ama boş | `journalctl -u gbmaid -f` ile API hatasına bak |

---

## 5b — Artefaktlar (696 KB) — **atlanırsa üç uç nokta 503 verir**

`artifacts/` klasörü **22 GB**'tır ve bilinçli olarak depo dışındadır (ara
işlem çıktıları). Ama servis yolunun üç ucu oradan dosya okur:

| Uç nokta | Okuduğu |
|---|---|
| `/similar` | `week3/faiss_index/` + `week3/molecular_omics/` (10 dosya, 380 KB) |
| `/model_performance` | 3 CSV + 2 JSON |
| `/model_curves` | 3 CSV |

🔴 **Ölçüldü (2026-09-16):** gereken toplam **18 dosya / 696 KB**. Özellikle
`week3/cox_model/` klasörü **78 MB / 169 dosyadır** ama servis ondan **yalnız
5 CSV (11 KB)** okur — klasörün tamamını taşımaya gerek yoktur.

Bunun için hazır script var; **kendi bilgisayarından** çalıştır:

```powershell
cd "C:\Users\Barış\Desktop\GBM-AID Prototip\gbm-aid mert"
.\deploy\artefakt_kopyala.ps1 -SunucuIP SUNUCU-IP
```

Script önce dosyaların varlığını denetler, eksik varsa **kopyalamadan durur**;
sonra uzak klasörleri açıp taşır ve sunucuda dosya sayısını doğrular
(beklenen: 17–18 dosya).
