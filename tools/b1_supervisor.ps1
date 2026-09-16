# B1 GÖZETMEN — arka-plan süreç ölümüne karşı otomatik yeniden başlatıcı
#
# NEDEN (2026-08-14): Bu ortamda uzun arka-plan süreçleri ~25-60 dakikada
# SESSİZCE ölüyor. UPenn B1 koşusu 14:11'de başladı, 15:10'da Pass 3
# [60/611]'de öldü. Elle yeniden başlatmak her seferinde koordinatörün
# fark etmesini gerektiriyor.
#
# NEDEN GÜVENLİ: `run_pyradiomics_*.py` script'leri TAM İDEMPOTENT:
#   - Pass 1: N4 cache (sidecar + içerik hash) -> yeniden hesaplamaz
#   - Pass 2: refit-guard -> geçerli stats varsa REUSE eder (asla --fresh-fit)
#   - Pass 3: `_load_done_keys(report)` -> tamamlanmış (hasta,bölge) atlar
# Yani her yeniden başlatma İLERİ gider, iş tekrarlanmaz.
#
# DURMA KOŞULU: provenance manifest'i yazıldığında (= koşu tamamlandı) ya da
# MaxAttempts aşıldığında durur. Sonsuz döngü YOK.
#
# KULLANIM (detached başlatılmalı):
#   Start-Process powershell -ArgumentList '-NoProfile','-File',
#     'X:\tools\b1_supervisor.ps1','-Cohort','upenn','-Expected','611' -WindowStyle Hidden

param(
    [Parameter(Mandatory = $true)][ValidateSet('tcga', 'upenn', 'lumiere')][string]$Cohort,
    [Parameter(Mandatory = $true)][int]$Expected,
    [int]$MaxAttempts = 30
)

$ErrorActionPreference = 'Continue'

# X: = ASCII subst sürücüsü. `.resolve()` KULLANMA (Türkçe yola çözer, ITK bozulur).
$Root = 'X:\'
$Py = Join-Path $Root '.venv310_pyradiomics\Scripts\python.exe'
$Script = Join-Path $Root "tools\run_pyradiomics_$Cohort.py"
$OutDir = Join-Path $Root 'artifacts\week3\pyradiomics'
$Report = Join-Path $OutDir "${Cohort}_pyradiomics_c32.csv"
$Manifest = "$Report.provenance.json"
$Log = Join-Path $OutDir "B1_${Cohort}_run.log"
$SupLog = Join-Path $OutDir "B1_${Cohort}_supervisor.log"

function Write-Sup([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Add-Content -Path $SupLog -Value $line -Encoding utf8
}

Write-Sup "GOZETMEN BASLADI cohort=$Cohort expected=$Expected maxAttempts=$MaxAttempts"

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {

    if (Test-Path $Manifest) {
        Write-Sup "MANIFEST VAR -> kosu TAMAMLANDI, gozetmen duruyor (deneme=$attempt)"
        break
    }

    Write-Sup "deneme $attempt/$MaxAttempts baslatiliyor"

    # stderr'i log'a birlestir; PowerShell'in NativeCommandError davranisi
    # yuzunden exit code'a degil MANIFEST varligina bakiyoruz.
    & $Py $Script --expected-image-count $Expected *>> $Log

    $code = $LASTEXITCODE
    Write-Sup "deneme $attempt bitti (exitCode=$code)"

    if (Test-Path $Manifest) {
        Write-Sup "MANIFEST YAZILDI -> kosu TAMAMLANDI (deneme=$attempt)"
        break
    }

    # Gate reddi / yapilandirma hatasi gibi kalici hatalarda sonsuz denemeyi
    # onlemek icin: exit 2 (guard reddi) ust uste 3 kez gelirse dur.
    if ($code -eq 2) {
        $script:guardFails = $script:guardFails + 1
        if ($script:guardFails -ge 3) {
            Write-Sup "DUR: exit 2 (guard reddi) 3 kez ust uste -- kalici hata, insan mudahalesi gerekiyor"
            break
        }
    }
    else {
        $script:guardFails = 0
    }

    Start-Sleep -Seconds 10
}

if (-not (Test-Path $Manifest)) {
    Write-Sup "UYARI: gozetmen bitti ama MANIFEST YOK -- kosu tamamlanmadi"
}
Write-Sup "GOZETMEN BITTI"
