#!/usr/bin/env bash
# =============================================================================
# GBM-AID -- sunucu on kurulumu (KURULUM.md adim 1-4)
# =============================================================================
# Ubuntu 22.04 / 24.04 uzerinde, root olarak calistirilir:
#
#   ssh root@SUNUCU-IP
#   curl -fsSL https://raw.githubusercontent.com/gbm-aid/gbm-aid/main/deploy/kurulum.sh -o kurulum.sh
#   less kurulum.sh          # 🔴 CALISTIRMADAN ONCE OKU
#   bash kurulum.sh
#
# Ne YAPAR : kullanici + paketler + Caddy + kod + Python ortami + klasorler
# Ne YAPMAZ: .env olusturmaz, MR/artefakt kopyalamaz, servisi BASLATMAZ,
#            veritabanina DOKUNMAZ. Onlar bilincli olarak elle yapilir.
#
# Tekrar calistirilabilir (idempotent): var olan adimlari atlar.
# =============================================================================
set -euo pipefail

KOD_DIZINI="/opt/gbmaid"
VERI_DIZINI="/srv/gbmaid"
DEPO="https://github.com/gbm-aid/gbm-aid.git"
SERVIS_KULLANICISI="gbmaid"

adim() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
bilgi() { printf "    %s\n" "$1"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "Bu script root olarak calistirilmali: sudo bash kurulum.sh" >&2
  exit 1
fi

adim "1/6  Sistem paketleri"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
  python3.10 python3.10-venv python3-pip git curl ca-certificates \
  debian-keyring debian-archive-keyring apt-transport-https rsync
bilgi "python3.10: $(python3.10 --version 2>&1)"

adim "2/6  Caddy (ters vekil + otomatik HTTPS)"
if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy
  bilgi "kuruldu: $(caddy version | head -1)"
else
  bilgi "zaten kurulu: $(caddy version | head -1)"
fi

adim "3/6  Servis kullanicisi ve klasorler"
if ! id -u "$SERVIS_KULLANICISI" >/dev/null 2>&1; then
  adduser --system --group --home "$KOD_DIZINI" "$SERVIS_KULLANICISI"
  bilgi "olusturuldu: $SERVIS_KULLANICISI"
else
  bilgi "zaten var: $SERVIS_KULLANICISI"
fi
mkdir -p "$VERI_DIZINI/nas" "$VERI_DIZINI/ucsf"
bilgi "veri klasorleri: $VERI_DIZINI/{nas,ucsf}"

adim "4/6  Kod deposu"
if [ -d "$KOD_DIZINI/.git" ]; then
  git -C "$KOD_DIZINI" pull --ff-only
  bilgi "guncellendi: $(git -C "$KOD_DIZINI" log -1 --format='%h %s' | cut -c1-70)"
else
  # adduser --home dizini olusturdugu icin klasor bos ama VAR olabilir
  if [ -d "$KOD_DIZINI" ] && [ -n "$(ls -A "$KOD_DIZINI" 2>/dev/null)" ]; then
    echo "HATA: $KOD_DIZINI bos degil ve git deposu da degil." >&2
    exit 1
  fi
  git clone --depth 1 "$DEPO" "$KOD_DIZINI"
  bilgi "klonlandi: $(git -C "$KOD_DIZINI" log -1 --format='%h %s' | cut -c1-70)"
fi

adim "5/6  Python sanal ortami ve bagimliliklar"
# ⚠️ YALNIZ requirements.txt. `requirements-nnunet.txt` ve
#    `requirements-pyradiomics.txt` AYRI ortamlar icindir; torch surumleri
#    cakisir ve nnU-Net varsayilan olarak KAPALIDIR.
cd "$KOD_DIZINI"
if [ ! -x ".venv/bin/python" ]; then
  python3.10 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
bilgi "kurulan paket sayisi: $(.venv/bin/pip list --format=freeze | wc -l)"

adim "6/6  Windows'ta engellenen paketler burada calisiyor mu"
# Gelistirme makinesinde Smart App Control `llvmlite.dll` ve SimpleITK'yi
# engelliyordu; SAC Windows'a ozgudur, burada calismalari BEKLENIR.
.venv/bin/python - <<'PY'
for ad in ("shap", "numba", "llvmlite.binding", "SimpleITK", "faiss", "torch", "xgboost", "lifelines"):
    try:
        __import__(ad)
        print("    %-20s OK" % ad)
    except Exception as e:
        print("    %-20s HATA: %s" % (ad, str(e)[:60]))
PY

chown -R "$SERVIS_KULLANICISI:$SERVIS_KULLANICISI" "$KOD_DIZINI" "$VERI_DIZINI"

cat <<'SON'

=============================================================================
ON KURULUM BITTI. Sirada ELLE yapilacaklar (KURULUM.md):

  5   MR dosyalarini kopyala (6,47 GB)      -> /srv/gbmaid/{nas,ucsf}
  5b  Artefaktlari kopyala (696 KB)         -> deploy/artefakt_kopyala.ps1
  6   .env olustur                          -> cp deploy/env.sunucu.ornek /opt/gbmaid/.env
                                               nano /opt/gbmaid/.env && chmod 600
  7   Kurulum oncesi dogrulama              -> KURULUM.md adim 7
  8   systemctl enable --now gbmaid
  9   Caddyfile + alan adi

🔴 Servis HENUZ BASLATILMADI -- once .env ve veri dosyalari hazir olmali.
=============================================================================
SON
