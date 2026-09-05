#!/usr/bin/env bash
# Installation reproductible du pipeline (Linux/macOS).
# Voir docs/plan-technique.md, section 10 : point d'entrée unique + setup.
#
# Usage :
#   ./setup.sh
set -euo pipefail

echo "== 1/3 : FFmpeg / FFprobe =="
if command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg déjà installé."
else
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    else
        echo "Gestionnaire de paquets inconnu. Installe FFmpeg manuellement : https://ffmpeg.org/download.html" >&2
        exit 1
    fi
fi

echo
echo "== 2/3 : environnement virtuel Python =="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Environnement virtuel créé dans .venv/"
else
    echo ".venv/ existe déjà."
fi

echo
echo "== 3/3 : dépendances Python =="
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "Installation terminée."
echo "Prochaines étapes :"
echo "  1. Dépose une vidéo de test dans input/source.mp4"
echo "  2. Télécharge les modèles :  ./.venv/bin/python scripts/download_models.py"
echo "  3. Lance le pipeline :        ./.venv/bin/python run.py"
