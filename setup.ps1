# Installation reproductible du pipeline (Windows).
# Voir docs/plan-technique.md, section 10 : point d'entree unique + setup.
#
# Usage :
#   .\setup.ps1

$ErrorActionPreference = "Stop"

function Test-CommandExists($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "== 1/3 : FFmpeg / FFprobe ==" -ForegroundColor Cyan
if (Test-CommandExists "ffmpeg") {
    Write-Host "FFmpeg deja installe."
} else {
    if (-not (Test-CommandExists "winget")) {
        throw "FFmpeg introuvable et winget indisponible. Installe FFmpeg manuellement : https://ffmpeg.org/download.html"
    }
    Write-Host "Installation de FFmpeg via winget (Gyan.FFmpeg)..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Write-Host "FFmpeg installe. Ouvre un nouveau terminal si 'ffmpeg' reste introuvable (PATH a rafraichir)." -ForegroundColor Yellow
}

Write-Host "`n== 2/3 : environnement virtuel Python ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Environnement virtuel cree dans .venv/"
} else {
    Write-Host ".venv/ existe deja."
}

Write-Host "`n== 3/3 : dependances Python ==" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nInstallation terminee." -ForegroundColor Green
Write-Host "Prochaines etapes :"
Write-Host "  1. Depose une video de test dans input/source.mp4"
Write-Host "  2. Telecharge les modeles :  .\.venv\Scripts\python.exe scripts\download_models.py"
Write-Host "  3. Lance le pipeline :        .\.venv\Scripts\python.exe run.py"
