# demo-auto-editor

Pipeline local et reproductible de montage automatisé de vidéos de démonstration SAP Fiori : transcription, traduction, voix off IA, sous-titres et incrustations visuelles synchronisées, jusqu'à l'export final.

Le plan technique complet est dans [`docs/plan-technique.md`](docs/plan-technique.md).

## Pourquoi local

Aucune donnée SAP (écran, audio, texte) ne doit être transmise à un service externe. Toutes les étapes (transcription, traduction, voix, montage) tournent sur le poste.

## Pipeline

```
vidéo source
  -> faster-whisper local (transcription FR)
  -> LLM local (traduction EN + conducteur de montage)
  -> moteur vocal local (Kokoro / Piper)
  -> recalage des timecodes
  -> FFmpeg (montage, overlays, sous-titres)
```

## Prérequis matériel

| Configuration | Modèles utilisables |
|---|---|
| 16 Go RAM, sans GPU | Whisper `small` + LLM 3B quantifié (plus lent) |
| 32 Go RAM, ~8 Go VRAM | Whisper `small`/`medium` + LLM 4B quantifié (recommandé) |
| 12+ Go VRAM | Modèles plus précis, traitement plus rapide |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
python scripts/download_models.py
```

## Utilisation

```bash
# Pipeline complet
python run.py --input input/source.mp4

# Étape par étape (débogage / reprise partielle)
python run.py --step transcribe
python run.py --step translate
python run.py --step narrate
python run.py --step render
```

## Organisation du dépôt

```
input/       vidéo source (non versionnée)
data/        métadonnées, transcription, conducteur de montage, sous-titres
audio/       audio source et narration générée (non versionné)
frames/      vignettes extraites (non versionné)
overlays/    assets d'incrustation (zoom, highlight, callout...)
scripts/     étapes du pipeline
output/      rendus finaux (non versionné)
logs/        journaux d'exécution
models/      poids des modèles téléchargés (non versionné)
```

## État du projet

Squelette initial. Les scripts du pipeline (`scripts/*.py`, `run.py`, `config.yaml`) restent à implémenter selon le plan technique.
