# =============================================================================
# GBM-AID -- servis yolunun ihtiyac duydugu ARTEFAKTLARI sunucuya kopyalar
# =============================================================================
# Windows PowerShell'den calistirilir (senin bilgisayarindan).
#
#   .\deploy\artefakt_kopyala.ps1 -SunucuIP 1.2.3.4
#
# 🔴 NEDEN AYRI BIR SCRIPT:
#   `artifacts/` klasoru 22 GB'dir ve `.gitignore` ile depo DISINDA tutulur --
#   bu bilincli bir karardir (ara islem ciktilari depoya girmez). Ama servis
#   yolunun uc ucu (/similar, /model_performance, /model_curves) o klasorden
#   BIRKAC KUCUK DOSYA okur. OLCULDU (2026-09-16): gereken toplam ~400 KB,
#   yani 22 GB'in tamamini degil, yalniz bu dosyalari tasimak yeterlidir.
#
#   Ozellikle: `artifacts/week3/cox_model/` klasoru 78 MB / 169 dosyadir ama
#   servis ondan YALNIZ 5 CSV (toplam 11 KB) okur.
# =============================================================================

param(
    [Parameter(Mandatory = $true)][string]$SunucuIP,
    [string]$Kullanici = "root",
    [string]$Hedef = "/opt/gbmaid/artifacts"
)

$ErrorActionPreference = "Stop"
$Kod = Split-Path -Parent $PSScriptRoot   # deploy/ -> gbm-aid mert/
Set-Location $Kod

# --- servis yolunun GERCEKTEN okudugu dosyalar (koddan cikarildi) -----------
$Dosyalar = @(
    # /model_performance + /model_curves
    "artifacts/week3/cox_model/week3_all_variants_comparison.csv",
    "artifacts/week3/cox_model/week3_external_extended_metrics_ci.csv",
    "artifacts/week3/cox_model/week3_v3b_calibration_summary.csv",
    "artifacts/week3/cox_model/week3_v3b_calibration_365.csv",
    "artifacts/week3/cox_model/week3_v3b_lowvar_v2amgmt_final_coefficients.csv",
    # /model_performance -- XGBoost katmani
    "artifacts/week4/xgboost_v2a_mgmt_reduce_collinearity_0912/week4_v2a_mgmt_aligned_run_metadata.json",
    "artifacts/week4/xgboost_v2a_mgmt_reduce_collinearity_0912/week4_v2a_mgmt_ucsf_external_evaluation.json"
)

# --- /similar icin FAISS indeksleri (klasor butun halinde) ------------------
$Klasorler = @(
    "artifacts/week3/faiss_index",
    "artifacts/week3/molecular_omics"
)

Write-Host "--- eksik dosya denetimi ---"
$eksik = @()
foreach ($d in $Dosyalar) { if (-not (Test-Path $d)) { $eksik += $d } }
foreach ($k in $Klasorler) { if (-not (Test-Path $k)) { $eksik += $k } }
if ($eksik.Count -gt 0) {
    Write-Host "EKSIK dosya/klasor var, kopyalama YAPILMADI:" -ForegroundColor Red
    $eksik | ForEach-Object { Write-Host "   $_" }
    exit 1
}

$toplam = 0
foreach ($d in $Dosyalar) { $toplam += (Get-Item $d).Length }
foreach ($k in $Klasorler) {
    $toplam += (Get-ChildItem $k -Recurse -File | Measure-Object Length -Sum).Sum
}
"{0} dosya + {1} klasor, toplam {2:N0} KB" -f $Dosyalar.Count, $Klasorler.Count, ($toplam / 1KB)

# 🔴 2026-09-16 HATA DUZELTMESI (canli kurulumda yakalandi):
#   Eskiden uzak dizin soyle hesaplaniyordu:
#       (Split-Path $_ -Parent).Replace("artifacts/", "")
#   `Split-Path` Windows'ta TERS EGIK CIZGI dondurur (`artifacts\week3\...`),
#   bu yuzden `"artifacts/"` araması ESLESMIYOR ve onek SILINMIYORDU. Sonuc:
#   dosyalar `/opt/gbmaid/artifacts/artifacts/week3/...` altina, yani
#   `artifacts` IKI KERE yazilmis bir yola gidiyordu (FAISS klasorleri dogru
#   yerdeydi cunku onlarin hedefi elle yazilmisti -- bu yuzden hata sessiz
#   kalmisti). Cozum: once ayraclari normalize et, SONRA oneki sil.
function UzakDizin([string]$yerelYol) {
    $ebeveyn = (Split-Path $yerelYol -Parent) -replace '\\', '/'
    $bagil = $ebeveyn -replace '^artifacts/', ''
    return "$Hedef/$bagil"
}

Write-Host "`n--- uzak klasorler aciliyor ---"
$uzakDizinler = ($Dosyalar | ForEach-Object { UzakDizin $_ }) | Select-Object -Unique
ssh "$Kullanici@$SunucuIP" ("mkdir -p " + ($uzakDizinler -join " ") + " $Hedef/week3")

Write-Host "`n--- dosyalar kopyalaniyor ---"
foreach ($d in $Dosyalar) {
    $uzakDizin = UzakDizin $d
    Write-Host "   $d  ->  $uzakDizin"
    scp $d "${Kullanici}@${SunucuIP}:$uzakDizin/"
}

Write-Host "`n--- FAISS indeksleri kopyalaniyor ---"
foreach ($k in $Klasorler) {
    Write-Host "   $k"
    scp -r $k "${Kullanici}@${SunucuIP}:$Hedef/week3/"
}

Write-Host "`n--- sunucuda dogrulama ---"
ssh "$Kullanici@$SunucuIP" "find $Hedef -type f | wc -l; du -sh $Hedef; chown -R gbmaid:gbmaid $Hedef 2>/dev/null || true"
Write-Host "`nBitti. Beklenen: 17 dosya (7 + 5 + 5)." -ForegroundColor Green
