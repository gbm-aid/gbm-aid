# V2 dort varyanti SIRAYLA kosar (2026-08-18, Baris onayli).
#
# Neden ayri ayri: `run_v2_variant_suite()` kol-bazli checkpoint YAZMIYOR
# (modeling-agent'in acik risk #1'i) ve bu ortamda arka-plan surecleri
# 25-45 dk'da olebiliyor. Tek komutla kosulursa bir cokme TUM varyantlari
# goturur; ayri ayri kosunca yalniz cöken varyant yeniden kosulur.
#
# Her varyant kendi log dosyasina yazar. Bir varyant coksede dongu DEVAM eder.

$ErrorActionPreference = "Continue"
$X = "X:"
$LOGDIR = "$X\artifacts\week3\cox_model\v2_logs"
$VARIANTS = @("v1_referans", "v2a_mgmt", "v2b_mgmt_spline", "v2c_mgmt_spline_wttc")

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
"[$(Get-Date -Format 'HH:mm:ss')] V2 SUITE BASLADI - $($VARIANTS.Count) varyant" |
    Out-File "$LOGDIR\_suite.log" -Append -Encoding utf8

foreach ($v in $VARIANTS) {
    $t0 = Get-Date
    "[$($t0.ToString('HH:mm:ss'))] BASLIYOR: $v" |
        Out-File "$LOGDIR\_suite.log" -Append -Encoding utf8

    & python "$X\tools\train_cox_week3.py" --variants $v `
        1>> "$LOGDIR\$v.out.log" 2>> "$LOGDIR\$v.err.log"
    $code = $LASTEXITCODE

    $dk = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0) {
        "[$(Get-Date -Format 'HH:mm:ss')] BITTI: $v  ($dk dk, exit 0)" |
            Out-File "$LOGDIR\_suite.log" -Append -Encoding utf8
    } else {
        "[$(Get-Date -Format 'HH:mm:ss')] HATA: $v  ($dk dk, exit $code) - digerlerine DEVAM" |
            Out-File "$LOGDIR\_suite.log" -Append -Encoding utf8
    }
}

"[$(Get-Date -Format 'HH:mm:ss')] V2 SUITE TAMAMLANDI" |
    Out-File "$LOGDIR\_suite.log" -Append -Encoding utf8
