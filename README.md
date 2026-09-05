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

Deux briques préparent le montage automatique (synchroniser une incrustation
avec le moment où le narrateur désigne un élément d'interface) :

- l'étape optionnelle `screen` indexe par OCR local le texte affiché et à quel
  moment (`data/screen_elements.json`) ;
- l'étape `translate` fait déclarer au LLM, pour chaque segment, ce que le
  narrateur désigne (`ui_reference`) : un élément **nommé** par son libellé,
  une simple **position** ("en haut à gauche"), ou **rien**.

Aucune des deux ne décide encore d'incrustation : l'appariement entre les deux
reste à écrire. La distinction nommé / position est ce qui évite d'encadrer
« Top repositories » parce que le narrateur a dit « en haut de la page ».

## Prérequis

- **FFmpeg / FFprobe** (moteur de montage, extraction audio, ffprobe pour les métadonnées)
- **Python 3.11+**
- Matériel : voir le tableau ci-dessous

| Configuration | Modèles utilisables |
|---|---|
| 16 Go RAM, sans GPU | Whisper `small` + LLM 3B quantifié (plus lent) |
| 32 Go RAM, ~8 Go VRAM | Whisper `small`/`medium` + LLM 4B quantifié (recommandé) |
| 12+ Go VRAM | Modèles plus précis, traitement plus rapide |

## Installation

Script unique (installe FFmpeg si absent, crée le venv, installe les dépendances) :

```bash
./setup.sh          # Linux/macOS
```
```powershell
.\setup.ps1          # Windows
```

Puis télécharge les modèles (Whisper, LLM, voix Piper — plusieurs Go depuis Hugging Face) :

```bash
python scripts/download_models.py
```

<details>
<summary>Installation manuelle (sans les scripts setup.sh/setup.ps1)</summary>

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
python scripts/download_models.py
```

FFmpeg doit être installé séparément et disponible dans le PATH (sous Windows : `winget install --id Gyan.FFmpeg -e`).
</details>

## Utilisation

```bash
# Pipeline complet
python run.py --input input/source.mp4

# Étape par étape (débogage / reprise partielle)
python run.py --step transcribe
python run.py --step translate
python run.py --step narrate
python run.py --step retime
python run.py --step subtitles
python run.py --step render
python run.py --step validate

# Hors pipeline par défaut : index OCR du texte à l'écran (plusieurs minutes)
python run.py --step screen
python scripts/detect_screen_text.py --max-seconds 40   # essai sur une tranche
python scripts/detect_screen_text.py --regroup          # re-règle sans relancer l'OCR
```

Les étapes lisent et écrivent les fichiers de `data/` : après correction d'un
conducteur de montage à la main, il suffit de reprendre à `retime`.

## Organisation du dépôt

```
input/       vidéo source (non versionnée)
data/        métadonnées, transcription, conducteur de montage, sous-titres,
             index du texte à l'écran
audio/       audio source et narration générée (non versionné)
frames/      vignettes extraites (non versionné)
overlays/    assets d'incrustation (zoom, highlight, callout...)
scripts/     étapes du pipeline
output/      rendus finaux (non versionné)
logs/        journaux d'exécution
models/      poids des modèles téléchargés (non versionné)
```

## Tests

La logique métier pure (recalage temporel, découpage source, construction des
filtres FFmpeg, sous-titres) est couverte par des tests qui n'invoquent ni
FFmpeg ni les modèles :

```bash
python -m pytest
```

## État du projet

Toutes les étapes du pipeline (A à H) sont implémentées et **un passage bout en
bout a été fait sur un extrait réel** (2 min 47 s, 34 segments) : modèles
téléchargés, FFmpeg réellement invoqué, `output/final.mp4` produit et validé
par l'étape `validate`.

Ce que ce passage a corrigé :

- désynchronisation vidéo/audio sur source à débit d'images variable (`fps=`
  forcé avant `tpad`) ;
- gel de plan surdimensionné au recalage : l'extension était calculée sur la
  durée brute de la narration alors que l'audio est joué accéléré (~6 s
  d'image figée inutile sur 188 s) ;
- `drawbox` recevait des variables `main_w` / `main_h` qui n'existent pas dans
  ce filtre : le premier `visual_action` renseigné faisait échouer tout le
  rendu.

Ce qui reste à valider :

- **montage automatique** : l'étape `screen` produit l'inventaire du texte
  affiché (`data/screen_elements.json`), mais rien ne le consomme encore.
  L'appariement entre ce que dit le narrateur et ce que montre l'écran reste à
  écrire, avec ses règles de rejet — sur l'extrait de référence, 40 % des
  libellés apparaissent à plusieurs endroits et 108 éléments sont visibles par
  segment en moyenne, donc un appariement naïf produirait surtout des erreurs.
- **incrustations visuelles** : `visual_action` est toujours à `null` en sortie
  de l'étape `translate` et doit être renseigné à la main. Seul `highlight` a
  été exercé sur du réel à ce jour ; `zoom`, `callout`, `popup` et
  `cursor_emphasis` ne l'ont pas été. Les effets texte (`callout`, `popup`)
  exigent `overlays.font_path` sous Windows.
- **terminologie SAP** : le glossaire est vide et l'extrait de test ne porte pas
  sur SAP — le cœur métier du projet n'a donc encore rien validé.
- **découpage des phrases** : Whisper coupe au milieu des phrases et chaque
  segment part au LLM isolément, ce qui produit des traductions fragmentées.
