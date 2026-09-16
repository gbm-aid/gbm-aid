# v2c (WT+TC) + v3a/v3b (dusuk-varyans filtresi + fold-guvenli
# standardizasyon) kosusu -- 2026-08-19, Baris talimati
# (gunluk-rapor-2026-08-18.md "Yarin icin sira" madde 1-2).
#
# NEDEN AYRI IKI SUREC (PARALEL):
#   * `week3_v2_variant_comparison.csv` (v2c) ve
#     `week3_v3_variant_comparison.csv` (v3) AYRI dosyalar -- carpismaz.
#   * v3a ve v3b AYNI surecte SIRAYLA kosar, cunku ikisi de AYNI v3
#     tablosunu yazar; paralel kosarlarsa biri digerinin tablosunu ezer
#     (2026-08-18'de Codex'in v2'de buldugu hata sinifi).
#   * v2c ise kendi v2 tablosunu tek basina yazar; dort v2 varyantinin
#     BIRLESIK tablosu kosu sonrasi `tools/merge_v2_comparison.py` ile
#     yeniden kurulur (o script her varyantin KENDI ciktisini okur).
#
# ONKOSUL: `check_training_pool_counts()` BOLGE-FARKINDA olmali
# (2026-08-19 degisikligi) -- yoksa v2c yine 609/583'te durur.

$ErrorActionPreference = "Continue"
$X = "X:"
$LOGDIR = "$X\artifacts\week3\cox_model\v2_logs"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

$stamp = { (Get-Date).ToString('HH:mm:ss') }
"[$(& $stamp)] v2c+v3 KOSUSU BASLADI (2 paralel surec)" |
    Out-File "$LOGDIR\_suite_0819.log" -Append -Encoding utf8

# --- Surec A: v2c (WT+TC, 186 radyomik aday) --------------------------
$jobA = Start-Process -FilePath "python" `
    -ArgumentList "$X\tools\train_cox_week3.py", "--variants", "v2c_mgmt_spline_wttc" `
    -RedirectStandardOutput "$LOGDIR\v2c_mgmt_spline_wttc.0819.out.log" `
    -RedirectStandardError  "$LOGDIR\v2c_mgmt_spline_wttc.0819.err.log" `
    -WindowStyle Hidden -PassThru

# --- Surec B: v3a -> v3b (SIRAYLA, ayni tabloyu yazdiklari icin) ------
$jobB = Start-Process -FilePath "python" `
    -ArgumentList "$X\tools\train_cox_week3.py", "--variants", "v3a_lowvar_v1referans", "v3b_lowvar_v2amgmt" `
    -RedirectStandardOutput "$LOGDIR\v3_suite.0819.out.log" `
    -RedirectStandardError  "$LOGDIR\v3_suite.0819.err.log" `
    -WindowStyle Hidden -PassThru

"[$(& $stamp)] BASLATILDI: v2c PID=$($jobA.Id) | v3a+v3b PID=$($jobB.Id)" |
    Out-File "$LOGDIR\_suite_0819.log" -Append -Encoding utf8
"$($jobA.Id)" | Out-File "$LOGDIR\_v2c.pid" -Encoding ascii
"$($jobB.Id)" | Out-File "$LOGDIR\_v3.pid" -Encoding ascii

$jobA.WaitForExit()
"[$(& $stamp)] BITTI: v2c (exit $($jobA.ExitCode))" |
    Out-File "$LOGDIR\_suite_0819.log" -Append -Encoding utf8

$jobB.WaitForExit()
"[$(& $stamp)] BITTI: v3a+v3b (exit $($jobB.ExitCode))" |
    Out-File "$LOGDIR\_suite_0819.log" -Append -Encoding utf8

"[$(& $stamp)] TUMU TAMAMLANDI" |
    Out-File "$LOGDIR\_suite_0819.log" -Append -Encoding utf8
