# GBM-AID -- Pi5 sentence-transformers olcumunu BASLATAN / IZLEYEN launcher.
#
# NEDEN POWERSHELL (K12 sinifi tuzak, 2026-09-15'te olculdu):
#   Git Bash'in `ssh`si bu makinede HOME'u `/c/Users/Bar\375\376/.ssh` diye
#   COZUMLUYOR (Turkce 'i' bozuluyor) -> ne anahtar ne known_hosts okunabiliyor.
#   Windows'un kendi `C:\WINDOWS\System32\OpenSSH\ssh.exe`'si ayni yolu DOGRU
#   cozuyor. Pi'ye giden HER komut bu script uzerinden / PowerShell'den gitmeli.
#
# ON KOSUL (bir kez, Baris yapar -- parola bir kez yazilir):
#   ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519"
#   type "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh <KULLANICI>@barispi1 `
#     "mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
#
# KULLANIM:
#   .\pi_st_launch.ps1 -PiUser <kullanici> -Action start
#   .\pi_st_launch.ps1 -PiUser <kullanici> -Action status
#   .\pi_st_launch.ps1 -PiUser <kullanici> -Action fetch

param(
    [Parameter(Mandatory = $true)][string]$PiUser,
    [string]$PiHost = "barispi1",
    [ValidateSet("start", "status", "fetch", "stop")][string]$Action = "status",
    [string]$BenchDir = "~/gbmaid_bench"
)

$ErrorActionPreference = "Stop"
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$scp = "$env:WINDIR\System32\OpenSSH\scp.exe"
$target = "$PiUser@$PiHost"
$toolsDir = $PSScriptRoot
# Yerel cikti: proje disina yazmaz, `artifacts/week6/` altina toplar.
$localOut = Join-Path (Split-Path $toolsDir -Parent) "artifacts\week6\pi_sentence_transformers"

function Invoke-Pi([string]$Cmd) {
    & $ssh -o ConnectTimeout=12 -o BatchMode=yes $target $Cmd
    if ($LASTEXITCODE -ne 0) { throw "SSH komutu basarisiz (exit $LASTEXITCODE): $Cmd" }
}

switch ($Action) {
    "start" {
        Write-Output "[1/4] Baglanti ve ortam kontrolu"
        Invoke-Pi "uname -m; python3 -V; nproc; free -m | head -2; df -h / | tail -1"

        Write-Output "[2/4] Dosyalar kopyalaniyor -> ${BenchDir}"
        Invoke-Pi "mkdir -p $BenchDir"
        & $scp -o BatchMode=yes (Join-Path $toolsDir "pi_st_benchmark.py") "${target}:$BenchDir/"
        if ($LASTEXITCODE -ne 0) { throw "scp basarisiz: pi_st_benchmark.py" }
        & $scp -o BatchMode=yes (Join-Path $toolsDir "pi_st_supervisor.sh") "${target}:$BenchDir/"
        if ($LASTEXITCODE -ne 0) { throw "scp basarisiz: pi_st_supervisor.sh" }
        # Windows tarafindan gelen olasi CRLF'i temizle -- bash CRLF'te coker.
        Invoke-Pi "cd $BenchDir; sed -i 's/\r$//' pi_st_supervisor.sh pi_st_benchmark.py; chmod +x pi_st_supervisor.sh"

        Write-Output "[3/4] Nobetci DETACHED baslatiliyor (SSH kopsa da devam eder)"
        Invoke-Pi "cd $BenchDir; setsid nohup bash ./pi_st_supervisor.sh > /dev/null 2>&1 < /dev/null & sleep 3; echo BASLATILDI"

        Write-Output "[4/4] Ilk nabiz"
        Invoke-Pi "cd $BenchDir; cat supervisor.pid 2>/dev/null; tail -5 supervisor.log 2>/dev/null"
        Write-Output "Durumu gormek icin: .\pi_st_launch.ps1 -PiUser $PiUser -Action status"
    }

    "status" {
        Invoke-Pi @"
cd $BenchDir 2>/dev/null || { echo 'BENCH DIZINI YOK -- start calistirilmadi'; exit 0; }
echo '--- nobetci ---'
if [ -f supervisor.pid ] && kill -0 \$(cat supervisor.pid) 2>/dev/null; then echo "CALISIYOR pid \$(cat supervisor.pid)"; else echo 'CALISMIYOR'; fi
echo '--- manifest ---'
python3 -c "import json;d=json.load(open('manifest.json'));print('run_complete =',d.get('run_complete'));print('sonuc =',d.get('sonuc'));print('fazlar =',{k:v.get('durum') for k,v in d.get('fazlar',{}).items()})" 2>/dev/null || echo 'manifest yok/okunamadi'
echo '--- son log ---'
tail -8 supervisor.log 2>/dev/null
"@
    }

    "fetch" {
        New-Item -ItemType Directory -Force -Path $localOut | Out-Null
        foreach ($f in @("manifest.json", "results.jsonl", "supervisor.log", "benchmark.out.log")) {
            & $scp -o BatchMode=yes "${target}:$BenchDir/$f" $localOut 2>$null
        }
        Write-Output "Indirilenler -> $localOut"
        Get-ChildItem $localOut | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    }

    "stop" {
        Invoke-Pi "cd $BenchDir 2>/dev/null && { kill \$(cat supervisor.pid) 2>/dev/null; kill \$(cat benchmark.pid) 2>/dev/null; echo DURDURULDU; }"
        Write-Output "Not: kismi sonuclar KORUNUR ($BenchDir/results.jsonl); yeniden start bitmis fazlari ATLAR."
    }
}
