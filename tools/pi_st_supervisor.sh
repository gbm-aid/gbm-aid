#!/usr/bin/env bash
# GBM-AID -- Pi5 sentence-transformers olcumu icin NOBETCI (supervisor).
#
# NEDEN NOBETCI VAR (Baris talimati 2026-09-15):
#   "uzun isin basina ... yarida kesilmesin ve yanlis olcum yapmasin diye
#    nobetci ata."
#   Gecmis emsal: 2026-08-14'te B1 PyRadiomics kosusu ajan-turu omru yuzunden
#   [60/611]'de SESSIZCE oldu (o gun 3. kez). Cozum detached bir gozetmendi.
#
# MERT'IN 2026-08-14'TE BULDUGU IKI KUSUR BURADA BILEREK KAPATILDI:
#   (1) `b1_supervisor.ps1` durma kosulu olarak yalnizca manifest dosyasinin
#       VARLIGINA bakiyordu -> `run_complete:false` bir manifest de
#       "tamamlandi" sayilip gozetmen ERKEN durabiliyordu.
#       BURADA: manifest `python3` ile AYRISTIRILIR ve `run_complete === True`
#       kontrol edilir. Ayristirilamayan manifest "bitmedi" sayilir.
#   (2) Rapor yazimi atomik degildi -> surec tam yazim aninda olurse dosya
#       bozulurdu. BURADA: yazim benchmark tarafinda `tmp + os.replace`.
#
# NEDEN SSH'A BAGLI DEGIL: `setsid` ile baslatilir, SSH oturumu kopsa da
# (Tailscale dalgalanmasi, laptop uykusu) kosu devam eder.
#
# KULLANIM:
#   setsid nohup bash ~/gbmaid_bench/pi_st_supervisor.sh > /dev/null 2>&1 &
set -u

BENCH_DIR="${BENCH_DIR:-$HOME/gbmaid_bench}"
SCRIPT="$BENCH_DIR/pi_st_benchmark.py"
MANIFEST="$BENCH_DIR/manifest.json"
LOG="$BENCH_DIR/supervisor.log"
RUN_LOG="$BENCH_DIR/benchmark.out.log"
PIDFILE="$BENCH_DIR/supervisor.pid"
RUN_PIDFILE="$BENCH_DIR/benchmark.pid"

MAX_RESTART=3          # Mert'in emsalindeki sinirla ayni
POLL_S=60
HARD_CAP_S=21600       # 6 saat: sonsuz dongu koruyucusu

mkdir -p "$BENCH_DIR"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" >> "$LOG"; }

# Ayni anda iki nobetci kosmasin (kilit protokolunun Pi tarafi).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  log "BASLAMADI: zaten calisan bir nobetci var (pid $(cat "$PIDFILE"))"
  exit 0
fi
echo $$ > "$PIDFILE"

if [ ! -f "$SCRIPT" ]; then
  log "HATA: $SCRIPT yok -- once dosya kopyalanmali"
  exit 2
fi

# run_complete DOSYA VARLIGINA gore degil, ALAN DEGERINE gore okunur.
is_complete() {
  [ -f "$MANIFEST" ] || return 1
  python3 - "$MANIFEST" <<'PYCHECK'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        sys.exit(0 if json.load(fh).get("run_complete") is True else 1)
except Exception:
    sys.exit(1)          # ayristirilamayan manifest = BITMEDI
PYCHECK
}

start_run() {
  # `nice`: bu makine ayni zamanda MR NAS'i -- demo/NAS okumasi acliktan olmesin.
  nice -n 10 python3 "$SCRIPT" --out-dir "$BENCH_DIR" >> "$RUN_LOG" 2>&1 &
  echo $! > "$RUN_PIDFILE"
  log "kosu baslatildi, pid $(cat "$RUN_PIDFILE")"
}

log "===== nobetci basladi (pid $$), bench_dir=$BENCH_DIR ====="
log "python3: $(python3 -V 2>&1) | uname: $(uname -m)"

if is_complete; then
  log "BITMIS: manifest run_complete=true -- yeniden baslatilmadi"
  rm -f "$PIDFILE"
  exit 0
fi

restarts=0
started_at=$(date +%s)
start_run

while true; do
  sleep "$POLL_S"

  if is_complete; then
    log "TAMAMLANDI: run_complete=true (yeniden baslatma sayisi=$restarts)"
    break
  fi

  run_pid="$(cat "$RUN_PIDFILE" 2>/dev/null || echo 0)"
  if ! kill -0 "$run_pid" 2>/dev/null; then
    # Surec olmus ama is bitmemis.
    if [ "$restarts" -ge "$MAX_RESTART" ]; then
      log "DURDURULDU: surec oldu ve yeniden baslatma siniri ($MAX_RESTART) doldu."
      log "Kismi sonuclar KORUNDU: $BENCH_DIR/results.jsonl"
      break
    fi
    restarts=$((restarts + 1))
    log "surec olmus (pid $run_pid), yeniden baslatiliyor ($restarts/$MAX_RESTART) -- bitmis fazlar ATLANACAK"
    start_run
    continue
  fi

  elapsed=$(( $(date +%s) - started_at ))
  if [ "$elapsed" -ge "$HARD_CAP_S" ]; then
    log "DURDURULDU: sert zaman siniri ($HARD_CAP_S s) asildi, kosu pid $run_pid sonlandiriliyor"
    kill "$run_pid" 2>/dev/null
    sleep 10
    kill -9 "$run_pid" 2>/dev/null
    break
  fi

  # Her turda kisa bir nabiz: hangi faz calisiyor?
  faz="$(python3 - "$MANIFEST" <<'PYPHASE' 2>/dev/null || echo bilinmiyor
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        f = json.load(fh).get("fazlar", {})
    print(",".join("%s=%s" % (k, v.get("durum")) for k, v in f.items()) or "henuz-faz-yok")
except Exception:
    print("manifest-okunamadi")
PYPHASE
)"
  log "nabiz: gecen=${elapsed}s fazlar=$faz"
done

log "===== nobetci bitti ====="
rm -f "$PIDFILE"
exit 0
