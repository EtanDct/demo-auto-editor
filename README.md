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

L'étape optionnelle `crop` retire au préalable le bandeau du navigateur —
onglets, URL, favoris — pour ne garder que la page présentée. Aucune hauteur
n'est écrite en dur : la frontière est mesurée sur chaque vidéo, d'abord à la
couleur de la barre supérieure de l'application (`#354a5f` sur le thème Fiori
Belize, réglable dans `crop.anchor_colors`), sinon en repérant à partir d'où
l'image cesse d'être figée. On cherche où commence l'application plutôt qu'où
finit le navigateur : le bandeau du haut n'a aucune signature stable — thème
clair ou sombre, avec ou sans favoris, un navigateur ou un autre — alors que la
barre applicative a une teinte connue et pleine largeur. Elle est conservée,
elle fait partie du produit montré.

L'étape produit une vidéo de travail sur laquelle tout l'aval se recale, et une
image de contrôle dans `logs/crop_preview.jpg`. Si la teinte est absente et que
la seconde méthode ne tranche pas franchement, elle refuse de rogner plutôt que
d'entamer l'application.

Deux briques préparent le montage automatique (synchroniser une incrustation
avec le moment où le narrateur désigne un élément d'interface) :

- l'étape optionnelle `screen` indexe par OCR local le texte affiché et à quel
  moment (`data/screen_elements.json`) ;
- l'étape `translate` fait déclarer au LLM, pour chaque segment, ce que le
  narrateur désigne (`ui_reference`) : un élément **nommé** par son libellé,
  une simple **position** ("en haut à gauche"), ou **rien** ;
- l'étape `cursor` suit le pointeur de souris (`data/cursor_track.json`), signal
  indépendant du texte donc insensible au décalage de langue entre la narration
  et l'interface.

L'étape `match` rapproche le tout et propose une incrustation quand la
correspondance ne laisse pas de place au doute. Elle est réglée pour la
précision, pas pour le rappel : elle refuse sur score insuffisant, sur
ambiguïté (le même libellé affiché à deux endroits), sur élément trop fugace ou
sur boîte aberrante, et consigne le motif de chaque refus. Seule la position du
pointeur peut sauver un candidat écarté pour ambiguïté, et uniquement si elle
tombe **sur** lui : le pointeur ne fabrique jamais une correspondance à partir
de rien et ne renverse jamais un appariement déjà net. Rien n'atteint le
conducteur de montage sans `--apply`, et `--contact-sheet` produit une planche
de relecture (cadre dessiné sur la frame, légende avec score et durée
d'affichage) : valider une correspondance à l'œil prend deux secondes, saisir
les coordonnées à la main en prend deux minutes.

La distinction nommé / position est ce qui évite d'encadrer « Top
repositories » parce que le narrateur a dit « en haut de la page ».

Une incrustation retenue ne couvre que la portion du segment où l'élément est
effectivement affiché (`start_offset` / `end_offset` de `visual_action`), avec
un fondu d'entrée et de sortie.

**Les incrustations pilotées par la souris** (`cursor_overlay` dans
`config.yaml`) ne passent pas par l'appariement du tout : le libellé sur lequel
la souris se pose est encadré. Comme rien n'y dépend du texte prononcé, elles
fonctionnent quelle que soit la langue de l'interface et quoi que dise le
narrateur — c'est la voie qui produit effectivement des incrustations
aujourd'hui.

Un marqueur suivant le pointeur existe aussi (`follow_enabled`) mais est
désactivé : il reste figé sur la dernière position tenue pendant les creux de
détection, donc là où la souris n'est plus. Tenir la position vaut pour déduire
un survol, corroboré par l'élément qui se trouve dessous ; pas pour un marqueur
qui prétend dire où est la souris.

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

Puis télécharge les modèles (Whisper et LLM depuis Hugging Face, voix Kokoro depuis GitHub — plusieurs Go) :

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

# Régénérer seulement les chapitres, sans retraduire
python scripts/translate.py --chapters-only

# Hors pipeline par défaut : retrait du bandeau de navigateur
python run.py --step crop

# Hors pipeline par défaut : index OCR du texte à l'écran (plusieurs minutes)
python run.py --step screen
python scripts/detect_screen_text.py --max-seconds 40   # essai sur une tranche
python scripts/detect_screen_text.py --regroup          # re-règle sans relancer l'OCR

# Hors pipeline par défaut : suivi du pointeur (réutilise les frames de `screen`)
python run.py --step cursor

# Hors pipeline par défaut : appariement narrateur / écran
python run.py --step match                              # rapport seul
python scripts/match_overlays.py --contact-sheet        # + planche de relecture
python scripts/match_overlays.py --apply                # reporter dans l'EDL
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
work/        vidéo recadrée, cartons d'intro et de fin (non versionné)
scripts/     étapes du pipeline
output/      rendus finaux (non versionné)
logs/        journaux d'exécution
models/      poids des modèles téléchargés (non versionné)
```

## Voix off

La narration est synthétisée en local par **Kokoro-82M** (licence Apache 2.0), voix
`af_heart` par défaut, via le paquet `kokoro-onnx` qui tourne sur onnxruntime,
sans GPU. Les fichiers (325 Mo de modèle, 28 Mo de voix) sont tirés une fois par
`download_models.py`.

Sur la démo Sales Report, comparée à l'ancienne voix Piper `en_US-amy-medium` :

| | Piper | Kokoro |
|---|---|---|
| Synthèse des 15 phrases | 18 s | 43 s (chargement compris) |
| Durée de la narration | 85,9 s | 73,7 s |
| Mots reconnus par Whisper | 96 % | 99 % |

Le débit plus rapide laisse davantage de blanc à retirer : la vidéo livrée passe
de 111 s à 100 s. Piper reste disponible — `tts.engine: piper` avec
`tts.voice: en_US-amy-medium` — et `tts.speed` règle le débit de Kokoro.

## Cartons d'introduction et de fin

La vidéo livrée s'ouvre sur un carton de cinq secondes. Le titre est produit par
le LLM local à partir du début de la transcription et écrit dans
`data/intro.json` ; `intro.title` et `intro.subtitle` dans `config.yaml` le
remplacent sans relancer la traduction, et `intro.enabled: false` le supprime.

Elle se ferme sur un carton de quatre secondes aux mêmes couleurs, qui reprend
ce titre avec « Thanks for watching » (`outro.title`, `outro.subtitle`,
`outro.enabled`).

Les cartons sont encodés à part aux paramètres exacts du master puis collés de
part et d'autre sans réencodage : rien à décaler côté audio, sous-titres ou
timeline.

## Mise en page de livraison

Le recadrage retire un bandeau dont la hauteur change d'une capture à l'autre.
Plutôt que de livrer des formats différents, l'image est posée sur un canevas
fixe (**1920×1080** par défaut, `layout.width` / `layout.height`) et la place
libérée devient deux bandes, sur fond `#354a5f` qui prolonge la barre Fiori :

- **en haut**, le chapitre en cours (« 2/4 Sales Order Report »), le rappel en
  pastille du nom de l'élément encadré, un logo optionnel, et un filet de
  progression marqué à chaque changement de chapitre, qui sépare la bande de
  l'application ;
- **en bas**, les sous-titres, centrés dans la bande : ils ne masquent plus
  l'interface.

Aucune hauteur n'est fixée en dur : les bandes sont ce qui reste une fois
l'image posée. Quand la place manque, la bande du haut disparaît d'abord, puis
les sous-titres reviennent sur l'image. Une capture trop grande est réduite
juste assez pour tenir ; une plus petite est agrandie tant que les bandes
tiennent encore. `layout.enabled: false` livre l'image à sa taille rognée.

**Chapitres.** La narration est découpée en tranches de durées voisines (une
pour ~35 s, `layout.chapter_seconds`) que le LLM ne fait que nommer, à
température nulle. Prié de choisir lui-même où couper, un modèle 3B faisait un
chapitre par phrase. Le résultat, `data/chapters.json`, se corrige à la main ;
sans chapitres valides, la bande affiche le titre de la vidéo.

**Rappel.** Seuls les cadres posés par l'appariement narrateur/écran sont
nommés dans la bande : les survols de la souris produisent encore trop de faux
positifs pour qu'on écrive leur nom en toutes lettres.

**Logo.** `layout.logo_path` accepte un PNG, de préférence à fond transparent,
aligné à droite de la bande du haut. Préférez le vôtre à celui d'un éditeur :
un logo de marque tierce peut faire passer la vidéo pour une production
officielle.

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

- **montage automatique** : la chaîne `screen` / `cursor` -> `translate` ->
  `match` est en place, mais l'extrait de référence ne permet pas de la valider.
  Le narrateur y décrit au lieu de nommer, et l'interface est en anglais alors
  que la narration est en français : sur 34 segments, 11 désignent un élément
  nommé, aucun n'est retrouvable à l'écran, et `match` en retient donc 0 —
  comportement correct sur cette vidéo, mais qui ne prouve pas que les seuils
  sont bons. Chaque brique est donc vérifiée séparément sur les données réelles :
  appariement (contrôle positif "Pull requests", score 1.00, boîte correcte),
  détection du pointeur (6 positions sur 6 justes, flèche comme main), et
  concordance des repères (36 des 86 positions du pointeur tombent dans la boîte
  d'un libellé, 22 autres à moins de 0.02). Il faut un extrait où le narrateur
  nomme des libellés écrits à l'écran, dans la même langue, pour mesurer
  précision et rappel.
- **incrustations visuelles** : `visual_action` est toujours à `null` en sortie
  de l'étape `translate` et doit être renseigné à la main (ou par `match
  --apply`). `highlight` est exercé sur du réel, minutage et fondu compris ;
  `zoom`, `callout`, `popup` et `cursor_emphasis` ne le sont pas encore. Les
  effets texte (`callout`, `popup`) exigent `overlays.font_path` sous Windows,
  et un `zoom` ne peut pas être minuté (FFmpeg n'expose pas `crop` à la
  timeline).
- **terminologie SAP** : le glossaire est vide et l'extrait de test ne porte pas
  sur SAP — le cœur métier du projet n'a donc encore rien validé.
- **découpage des phrases** : Whisper coupe au milieu des phrases et chaque
  segment part au LLM isolément, ce qui produit des traductions fragmentées.
