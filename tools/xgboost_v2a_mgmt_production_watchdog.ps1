# NOBETCI (watchdog) -- XGBoost uretim kosusu (v2a_mgmt + --reduce-
# collinearity, v3b Cox recete-sadakati duzeltmesi SONRASI, 2026-09-12).
#
# NEDEN GEREKLI
# -------------
# `variant_run_watchdog.ps1`'in AYNI deseni (Baris talimati: "sapmayla
# kosmayalim... nobetci olmadan kosu baslatma" + DEVAM-NOTU-2026-09-12.md
# 2. bolum koordinator talimati) -- bu makinede arka-plan surecleri
# sessizce olebiliyor VE oturumlar beklenmedik kapanabiliyor. Bu script
# `variant_run_watchdog.ps1`'den KASITLI olarak AYRI (o dosyaya
# DOKUNULMADI) -- TEK is (`tools/train_xgboost_week4.py`), farkli cikti
# dosya adlari.
#
# K12 UYARISI (Turkce-yol tuzagi): bu dosyanin KENDI KAYNAGINDA hicbir
# non-ASCII karakter YOK (dotless-i / s-cedilla / g-breve / c-cedilla /
# o-umlaut / u-umlaut KULLANILMADI) -- Windows PowerShell 5.1 script
# dosyalarini bazen sistem ANSI codepage'i ile okuyabiliyor, dosya
# icinde Turkce karakter varsa parse/degisken-deger bozulmasi riski var.
# Varsayilan `$BaseDir` da ASCII `X:` (subst edilmis surucu, `variant_
# run_watchdog.ps1` ile AYNI mitigasyon) -- proje kokundeki "Baris"
# klasor adinin KENDISI runtime'da hicbir sekilde bu script metnine
# GOMULMEZ.
#
# TAMAMLANMA OLCUTU (variant_run_watchdog.ps1'deki AYNI ilke -- dosya
# VARLIGI YETMEZ, run_metadata.json GECERLI JSON olarak parse edilebilmeli):
#   - week4_v2a_mgmt_aligned_fold_results.csv
#   - week4_v2a_mgmt_aligned_run_metadata.json (GECERLI JSON)
#   - week4_v2a_mgmt_ucsf_external_evaluation.json (GECERLI JSON,
#     --evaluate-ucsf-external ile uretilir -- BU KOSUDA ZORUNLU)
#
# SESSIZ BASARI YOK: nobetci her kararini _watchdog.log'a yazar.

param(
    [int] $PollSeconds = 60,
    [int] $MaxRestarts = 3,
    [string] $BaseDir = "X:"
)

$ErrorActionPreference = "Continue"
$ROOT = $BaseDir
$OUT = "$ROOT\artifacts\week4\xgboost_v2a_mgmt_reduce_collinearity_0912"
$WLOG = "$OUT\_watchdog.log"

New-Item -ItemType Directory -Force -Path $OUT | Out-Null

function Write-WLog([string] $Message) {
    $line = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $Message"
    $line | Out-File $WLOG -Append -Encoding utf8
    Write-Host $line
}

$job = @{
    Name     = "xgboost_v2a_mgmt_reduce_collinearity"
    Args     = @(
        "$ROOT\tools\train_xgboost_week4.py",
        "--variant", "v2a_mgmt",
        "--reduce-collinearity",
        "--evaluate-ucsf-external",
        "--output-dir", "$OUT"
    )
    OutLog   = "$OUT\xgboost_v2a_mgmt_reduce_collinearity.0912.out.log"
    ErrLog   = "$OUT\xgboost_v2a_mgmt_reduce_collinearity.0912.err.log"
    PidFile  = "$OUT\_run.pid"
    Restarts = 0
    Done     = $false
}

function Test-JobComplete($Job) {
    $foldResults = "$OUT\week4_v2a_mgmt_aligned_fold_results.csv"
    $runMeta = "$OUT\week4_v2a_mgmt_aligned_run_metadata.json"
    $ucsfMeta = "$OUT\week4_v2a_mgmt_ucsf_external_evaluation.json"
    if (-not (Test-Path $foldResults)) { return $false }
    if (-not (Test-Path $runMeta)) { return $false }
    if (-not (Test-Path $ucsfMeta)) { return $false }
    try {
        Get-Content $runMeta -Raw | ConvertFrom-Json | Out-Null
        Get-Content $ucsfMeta -Raw | ConvertFrom-Json | Out-Null
    } catch {
        Write-WLog "UYARI: metadata GECERSIZ JSON -- tamamlanmis SAYILMADI."
        return $false
    }
    return $true
}

function Test-ProcAlive($Job) {
    if (-not (Test-Path $Job.PidFile)) { return $false }
    $raw = (Get-Content $Job.PidFile -Raw).Trim()
    if (-not $raw) { return $false }
    $procId = 0
    if (-not [int]::TryParse($raw, [ref] $procId)) { return $false }
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($null -eq $p) { return $false }
    return ($p.ProcessName -eq "python")
}

function Start-Job2($Job) {
    $p = Start-Process -FilePath "python" -ArgumentList $Job.Args `
        -RedirectStandardOutput $Job.OutLog -RedirectStandardError $Job.ErrLog `
        -WorkingDirectory $ROOT -WindowStyle Hidden -PassThru
    "$($p.Id)" | Out-File $Job.PidFile -Encoding ascii
    return $p.Id
}

Write-WLog "NOBETCI BASLADI (poll=${PollSeconds}sn, maks yeniden baslatma=$MaxRestarts, is=$($job.Name))"

while ($true) {
    if (Test-JobComplete $job) {
        $job.Done = $true
        Write-WLog "TAMAMLANDI: $($job.Name) -- tum ciktilar mevcut ve metadata gecerli."
        break
    }

    if (Test-ProcAlive $job) {
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if ($job.Restarts -ge $MaxRestarts) {
        Write-WLog "VAZGECILDI: $($job.Name) -- cokme sonrasi $MaxRestarts kez yeniden baslatildi, hala tamamlanmadi. ELLE INCELEME GEREKIYOR ($($job.ErrLog))."
        break
    }

    $isFirstLaunch = -not (Test-Path $job.PidFile)
    if (-not $isFirstLaunch) { $job.Restarts++ }
    $newPid = Start-Job2 $job
    if ($isFirstLaunch) {
        Write-WLog "ILK BASLATMA: $($job.Name) (PID=$newPid). Sayac TUKETILMEDI (restart=$($job.Restarts)/$MaxRestarts)."
    } else {
        Write-WLog "OLU SUREC TESPIT EDILDI -> YENIDEN BASLATILDI: $($job.Name) (cokme sonrasi restart $($job.Restarts)/$MaxRestarts, yeni PID=$newPid). DIKKAT: checkpoint yok, kosu SIFIRDAN basliyor."
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-WLog "NOBETCI BITTI. Sonuc: $(if (Test-JobComplete $job) { 'BASARILI' } else { 'BASARISIZ' }) (cokme sonrasi yeniden baslatma: $($job.Restarts))"
