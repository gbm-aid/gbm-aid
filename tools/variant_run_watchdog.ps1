# NOBETCI (watchdog) -- v2c ve v3 kosularini ayakta tutar.
# 2026-08-19, Baris talimati: "onlari birer nobetci ekle ki surecleri
# dusmesin".
#
# NEDEN GEREKLI
# -------------
# Bu ortamda arka-plan surecleri sessizce olebiliyor (2026-08-14'te 4 kez,
# 2026-08-18'de bir kez). `run_v2_variant_suite()`/`run_v3_variant_suite()`
# KOL-BAZLI CHECKPOINT YAZMIYOR (modeling-agent'in acik risk #1'i) -->
# bir olum = SIFIRDAN baslama. Nobetci olumu FARK EDER ve yeniden baslatir;
# fark etmezsek saatler bos gecer.
#
# `b1_supervisor.ps1`'in BILINEN IKI KUSURU (Mert'in 2026-08-14
# incelemesi, `AKTIF-GOREVLER.md` satir 67) BURADA NASIL ELE ALINIYOR:
#
#   KUSUR 1 -- "durma kosulu yalniz `Test-Path $Manifest` (dosya
#   VARLIGI); manifest icindeki `run_complete` alanini PARSE ETMIYOR,
#   `run_complete:false` bir manifest de 'tamamlandi' sayilip gozetmen
#   ERKEN durabilir."
#   -> DOGRUDAN KAPATILDI. Burada tamamlanma olcutu, varyantin UC
#      ciktisinin BIRDEN var olmasi (`_external_test.csv` +
#      `_run_metadata.json` + `_fold_results.csv`) VE
#      `_run_metadata.json`'un GECERLI JSON olarak PARSE EDILEBILMESIDIR.
#
#   KUSUR 2 -- "`flush()` (CSV rapor yazimi) ATOMIK DEGIL; surec tam
#   flush sirasinda olurse CSV bozulabilir."
#   -> DOLAYLI OLARAK azaltiliyor, TAM COZULMUYOR (durustce isaretli).
#      Azaltma su OLCULMUS gercege dayaniyor (2026-08-19, kod okunarak
#      dogrulandi): `write_variant_outputs()` dosyalari su SIRAYLA
#      yaziyor -- fold_results -> selection_frequency -> external_test
#      -> final_coefficients -> **run_metadata.json EN SON**. Yani
#      gecerli bir `run_metadata.json`'un varligi, yazicinin sonuna
#      kadar gittiginin makul bir VEKILIDIR. Ama bu bir ATOMIKLIK
#      GARANTISI DEGILDIR: teorik olarak erken bir CSV yirtilmis, sonra
#      metadata basariyla yazilmis olabilir. O ARTIK KALAN riske karsi
#      savunma bu script'te DEGIL, `tools/merge_all_variants.py`'dedir
#      (BOZUK / BOS / KOLON_EKSIK / DEGER_YOK kapilari).
#
# ⚠️ DUZELTME (2026-08-19, reviewer capraz dogrulamasi): bu basligin ILK
#    surumu ikinci kusuru "yeniden baslatma SINIRSIZ" diye yaziyordu.
#    BU YANLISTI -- `b1_supervisor.ps1` zaten `$MaxAttempts = 30` ile
#    sinirliydi (satir 24), boyle bir kusur HIC YOKTU. Asagidaki
#    `$MaxRestarts` iyi bir tasarim karari olarak KALIYOR, ama
#    b1_supervisor'in bir kusurunu kapattigi IDDIA EDILMIYOR.
#
# SESSIZ BASARI YOK: nobetci her kararini `_watchdog.log`'a yazar.

# $MaxRestarts = COKME SONRASI yeniden baslatma sayisi. Nobetci bir isi
# HIC baslatilmamis bulursa (PID dosyasi yok) o ILK baslatma bu sayaci
# TUKETMEZ -- yalniz daha once ayakta olup OLEN bir surecin yerine
# konan baslatmalar sayilir. (2026-08-19 reviewer LOW bulgusu: onceki
# surumde ilk baslatma da sayaci tuketiyordu, yani "maks 3 yeniden
# baslatma" fiilen "maks 2 cokme sonrasi restart" demekti.)
# 2026-09-11 EKLENDİ (Barış onayı, FİNAL KOŞU -- MODEL-SECIM-ANALIZI-
# 2026-09-11.md §4.3): `-BaseDir` parametresi eklendi -- bu oturumda `X:`
# subst'lenmemiş (mapping tarihsel/geçici, doğrulanamaz), yeniden subst
# kurmak yerine script'in TABAN DİZİNİ parametrize edildi. Varsayılan
# HÂLÂ `"X:"` -- eski çağrı biçimi (`-PollSeconds`/`-MaxRestarts`, argümansız
# `$BaseDir`) BİREBİR AYNI davranışı üretir (geriye dönük uyumlu). Yeni v3c
# işi için `-BaseDir "<repo>\gbm-aid mert"` ile çağrılır (bkz. bu görevin
# başlatma komutu) -- v2c/v3a/v3b işleri de AYNI `$BaseDir` altında ZATEN
# TAMAMLANMIŞ ÇIKTILARI görüp anında "TAMAMLANDI" işaretlenir (yeniden
# koşulmazlar), yalnız v3c aktif olarak izlenir/yeniden başlatılır.
param(
    [int] $PollSeconds = 60,
    [int] $MaxRestarts = 3,
    [string] $BaseDir = "X:"
)

$ErrorActionPreference = "Continue"
$X = $BaseDir
$OUT = "$X\artifacts\week3\cox_model"
$LOGDIR = "$OUT\v2_logs"
$WLOG = "$LOGDIR\_watchdog.log"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

function Write-WLog([string] $Message) {
    $line = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $Message"
    $line | Out-File $WLOG -Append -Encoding utf8
}

# Bir "is": ad + hangi varyantlarin bitmesi gerektigi + baslatma argumanlari
$jobs = @(
    @{
        Name     = "v2c"
        Variants = @("v2c_mgmt_spline_wttc")
        Args     = @("$X\tools\train_cox_week3.py", "--variants", "v2c_mgmt_spline_wttc")
        OutLog   = "$LOGDIR\v2c_mgmt_spline_wttc.0819.out.log"
        ErrLog   = "$LOGDIR\v2c_mgmt_spline_wttc.0819.err.log"
        PidFile  = "$LOGDIR\_v2c.pid"
        Restarts = 0
        Done     = $false
    },
    @{
        Name     = "v3"
        Variants = @("v3a_lowvar_v1referans", "v3b_lowvar_v2amgmt")
        Args     = @("$X\tools\train_cox_week3.py", "--variants", "v3a_lowvar_v1referans", "v3b_lowvar_v2amgmt")
        OutLog   = "$LOGDIR\v3_suite.0819.out.log"
        ErrLog   = "$LOGDIR\v3_suite.0819.err.log"
        PidFile  = "$LOGDIR\_v3.pid"
        Restarts = 0
        Done     = $false
    },
    # 2026-09-11 EKLENDİ (Barış onayı, FİNAL KOŞU -- duyarlılık kolu
    # `v3c_lowvar_wttc_mgmt_nospline`, MODEL-SECIM-ANALIZI-2026-09-11.md
    # §4.2/§4.3). `--output-dir` yukarıdaki iki işten FARKLI bir alt
    # klasöre (`final_v3c`) yazıyor -- v3b'nin final KOŞULMUŞ çıktısının
    # ÜZERİNE YAZILMASINI engellemek için (isim çakışması riski YOK, ama
    # ayrı klasör ekstra bir güvenlik katmanı). Bu yüzden `OutDir` ALANI
    # AYRI tutuluyor (bkz. `Test-JobComplete` -- artık `Job.OutDir` varsa
    # onu kullanıyor, yoksa global `$OUT`'a düşüyor -- eski iki iş İÇİN
    # DAVRANIŞ DEĞİŞMEDİ).
    @{
        Name     = "v3c"
        Variants = @("v3c_lowvar_wttc_mgmt_nospline")
        OutDir   = "$OUT\final_v3c"
        Args     = @(
            "$X\tools\train_cox_week3.py",
            "--variants", "v3c_lowvar_wttc_mgmt_nospline",
            "--outer-splits", "5", "--inner-splits", "5",
            "--n-bootstrap-stability", "200", "--n-bootstrap-external", "1000",
            "--seed", "42", "--primary-threshold", "0.6",
            "--expected-patient-count", "611", "--expected-event-count", "585",
            "--expected-ucsf-patient-count", "295", "--expected-ucsf-event-count", "169",
            "--output-dir", "$OUT\final_v3c"
        )
        OutLog   = "$LOGDIR\v3c_lowvar_wttc_mgmt_nospline.0911.out.log"
        ErrLog   = "$LOGDIR\v3c_lowvar_wttc_mgmt_nospline.0911.err.log"
        PidFile  = "$LOGDIR\_v3c.pid"
        Restarts = 0
        Done     = $false
    }
)

function Test-VariantComplete([string] $Variant, [string] $OutDir) {
    # UC cikti da olmali; metadata GECERLI JSON olmali (yarim yazim yakalanir).
    $ext  = "$OutDir\week3_${Variant}_external_test.csv"
    $meta = "$OutDir\week3_${Variant}_run_metadata.json"
    $fold = "$OutDir\week3_${Variant}_fold_results.csv"
    if (-not (Test-Path $ext))  { return $false }
    if (-not (Test-Path $meta)) { return $false }
    if (-not (Test-Path $fold)) { return $false }
    try {
        Get-Content $meta -Raw | ConvertFrom-Json | Out-Null
    } catch {
        Write-WLog "UYARI: $Variant metadata GECERSIZ JSON -- tamamlanmis SAYILMADI."
        return $false
    }
    return $true
}

function Test-JobComplete($Job) {
    # 2026-09-11 EKLENDİ: `Job.OutDir` verilmişse (v3c) ONU kullan, yoksa
    # global `$OUT`'a düş (v2c/v3 -- DAVRANIŞ DEĞİŞMEDİ).
    $jobOutDir = if ($Job.ContainsKey("OutDir")) { $Job.OutDir } else { $OUT }
    foreach ($v in $Job.Variants) {
        if (-not (Test-VariantComplete $v $jobOutDir)) { return $false }
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
    # Ayni PID baska bir surece yeniden atanmis olabilir -- adi dogrula.
    return ($p.ProcessName -eq "python")
}

function Start-Job($Job) {
    $p = Start-Process -FilePath "python" -ArgumentList $Job.Args `
        -RedirectStandardOutput $Job.OutLog -RedirectStandardError $Job.ErrLog `
        -WindowStyle Hidden -PassThru
    "$($p.Id)" | Out-File $Job.PidFile -Encoding ascii
    return $p.Id
}

Write-WLog "NOBETCI BASLADI (poll=${PollSeconds}sn, maks yeniden baslatma=$MaxRestarts)"

while ($true) {
    $allDone = $true

    foreach ($job in $jobs) {
        if ($job.Done) { continue }

        if (Test-JobComplete $job) {
            $job.Done = $true
            Write-WLog "TAMAMLANDI: $($job.Name) -- tum ciktilar mevcut ve metadata gecerli."
            continue
        }

        $allDone = $false

        if (Test-ProcAlive $job) { continue }

        # Surec olmus ama cikti tam degil.
        if ($job.Restarts -ge $MaxRestarts) {
            Write-WLog "VAZGECILDI: $($job.Name) -- cokme sonrasi $MaxRestarts kez yeniden baslatildi, hala tamamlanmadi. ELLE INCELEME GEREKIYOR ($($job.ErrLog))."
            $job.Done = $true   # dongu acisindan 'bitti', ama BASARISIZ
            continue
        }

        # ILK baslatma mi, yoksa OLEN bir surecin yerine mi?
        $isFirstLaunch = -not (Test-Path $job.PidFile)
        if (-not $isFirstLaunch) { $job.Restarts++ }
        # NOT: kol-bazli checkpoint OLMADIGI icin yeniden baslatma
        # SIFIRDAN baslar -- kayip suresi log'a yaziliyor ki rapora gecsin.
        $newPid = Start-Job $job
        if ($isFirstLaunch) {
            Write-WLog "ILK BASLATMA: $($job.Name) (PID=$newPid). Sayac TUKETILMEDI (restart=$($job.Restarts)/$MaxRestarts)."
        } else {
            Write-WLog "OLU SUREC TESPIT EDILDI -> YENIDEN BASLATILDI: $($job.Name) (cokme sonrasi restart $($job.Restarts)/$MaxRestarts, yeni PID=$newPid). DIKKAT: checkpoint yok, kosu SIFIRDAN basliyor."
        }
    }

    if ($allDone) { break }
    Start-Sleep -Seconds $PollSeconds
}

Write-WLog "NOBETCI BITTI. Sonuc ozeti:"
foreach ($job in $jobs) {
    $status = if (Test-JobComplete $job) { "BASARILI" } else { "BASARISIZ" }
    Write-WLog "   $($job.Name): $status (cokme sonrasi yeniden baslatma: $($job.Restarts))"
}
