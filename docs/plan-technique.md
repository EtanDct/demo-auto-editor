# Plan technique : localisation et montage automatisé SAP Fiori

## 1. Objectif technique

Construire un pipeline reproductible qui transforme une vidéo de démonstration SAP Fiori en une version anglaise comprenant :

- une transcription française horodatée ;
- une adaptation anglaise validée ;
- une voix off IA premium ;
- des sous-titres anglais incrustés ;
- des zooms et incrustations visuelles synchronisés avec les actions SAP ;
- un export vidéo professionnel et contrôlé automatiquement.

Le pipeline doit conserver la vidéo source autant que possible et produire des fichiers intermédiaires permettant de corriger une étape sans tout recommencer.

## 2. Architecture recommandée

### Architecture 100 % locale

L'ensemble du traitement doit fonctionner localement afin qu'aucun écran SAP, aucune donnée visible dans l'interface et aucun contenu audio ne soit transmis à un service externe.

Pipeline recommandé :

```
vidéo source
  -> faster-whisper local
  -> transcription française horodatée
  -> modèle de langage local léger
  -> traduction anglaise + conducteur de montage
  -> moteur vocal local
  -> recalage des timecodes
  -> FFmpeg : montage, overlays et sous-titres
```

La transcription, la traduction, le conducteur de montage et la voix doivent tous être exécutés sur le poste. Cette approche maximise la confidentialité et supprime les coûts d'API, mais impose de tester attentivement la qualité des voix locales.

### Modèles IA locaux

- **Transcription** : `faster-whisper`, avec le modèle `small` comme compromis initial entre précision et consommation de ressources. `tiny` convient aux tests rapides ; `medium` améliore la précision mais nécessite davantage de ressources.
- **Traduction et adaptation** : modèle de langage local Qwen de 3B ou 4B quantifié, exécuté via `llama.cpp` ou Ollama. Le modèle reçoit le glossaire SAP et doit retourner une structure validable, pas uniquement du texte libre.
- **Voix locale** : Kokoro pour rechercher un rendu plus naturel ; Piper pour privilégier la légèreté et la vitesse. Plusieurs voix doivent être comparées sur un extrait représentatif.

Configuration indicative :

- **16 Go de RAM, sans GPU dédié** : Whisper `small` et modèle de langage 3B quantifié ; traitement possible mais plus lent.
- **32 Go de RAM et environ 8 Go de VRAM** : Whisper `small` ou `medium` et modèle 4B quantifié ; configuration recommandée.
- **12 Go ou plus de VRAM** : modèles plus précis et traitement plus rapide.

Le pipeline doit conserver les modèles, versions, paramètres de quantification et réglages de génération dans un manifeste pour rendre les résultats reproductibles.

### Orchestration

- **Python 3.12+** pour piloter les étapes, valider les données et appeler les services externes.
- **Pydantic** ou JSON Schema pour valider les fichiers de configuration et le conducteur de montage.
- **CLI** avec Typer ou argparse pour lancer chaque étape séparément.
- **YAML** pour les paramètres lisibles par un humain ; JSON pour les échanges structurés.
- **Git** pour versionner les scripts, le glossaire, les textes et les paramètres, sans versionner les vidéos lourdes.

### Traitement vidéo et audio

- **FFmpeg** comme moteur de référence pour l'extraction, le mixage, les filtres, l'incrustation et l'export.
- **ffprobe** pour lire automatiquement la durée, la résolution, la fréquence d'images, les codecs et les pistes.
- **WAV PCM 48 kHz** pour les fichiers audio intermédiaires afin d'éviter les pertes liées aux compressions successives.
- **Vidéo intermédiaire ProRes ou DNxHR**, si plusieurs générations d'effets sont nécessaires ; sinon, filtres FFmpeg directement depuis la source.

### Génération des annotations

Deux options sont possibles :

1. **FFmpeg avec filtres `zoompan`, `drawbox`, `drawtext`, `overlay`**
   - léger et automatisable ;
   - adapté aux zooms, cadres, flèches simples et pop-ups ;
   - moins pratique pour des animations graphiques complexes.

2. **Composition via MoviePy, Remotion ou After Effects**
   - plus souple pour les animations et les modèles graphiques ;
   - plus lourd à déployer et à rendre ;

**Choix recommandé** : FFmpeg pour le premier prototype, avec un modèle d'overlay en SVG ou PNG généré par le script. Passer à Remotion ou After Effects uniquement si les effets dépassent les capacités des filtres FFmpeg.

## 3. Pipeline détaillé

### Étape A : inspection et préparation de la vidéo

1. Recevoir la vidéo source et calculer son empreinte SHA-256.
2. Utiliser `ffprobe` pour produire `source_metadata.json`.
3. Extraire l'audio en WAV mono ou stéréo selon la source.
4. Extraire des vignettes régulières pour inspecter les changements d'écran.
5. Conserver la fréquence d'images et la résolution d'origine, sauf exigence de livraison contraire.

Sorties :

- `source_metadata.json` ;
- `source_audio.wav` ;
- `frames/` ;
- rapport d'inspection.

### Étape B : transcription française

Options :

- **Whisper local** : meilleure maîtrise des données et coût marginal faible, mais besoin de ressources de calcul ;
- **API de transcription** : mise en œuvre rapide et qualité variable selon le fournisseur ;
- **Transcription manuelle** : meilleure maîtrise métier, mais délai plus important.

**Choix recommandé** : transcription automatique initiale avec Whisper, suivie d'un nettoyage programmatique et d'une vérification ciblée des termes SAP.

Pour le fonctionnement local, utiliser `faster-whisper` plutôt que l'implémentation Python standard lorsque la vitesse et la consommation mémoire sont prioritaires. Le modèle doit être choisi après un test sur un extrait représentatif contenant des termes SAP.

Le résultat doit utiliser un format segmenté :

```json
{
  "id": "seg-001",
  "start": 12.340,
  "end": 17.850,
  "text_fr": "Sélectionnez la société dans le champ correspondant."
}
```

Les timecodes servent d'abord à repérer le contenu. Ils ne doivent pas être considérés comme définitifs pour la version anglaise.

### Étape C : traduction et conducteur maître

Créer un fichier `edit_decision_list.yaml` contenant, pour chaque segment :

```yaml
- id: seg-001
  source_start: 12.340
  source_end: 17.850
  text_fr: "Sélectionnez la société dans le champ correspondant."
  text_en: "Select the company code in the relevant field."
  sap_terms:
    - "Company Code"
  visual_action:
    type: highlight
    target: company_code
    x: 0.42
    y: 0.31
    width: 0.21
    height: 0.06
  narration:
    voice: professional_female_01
    pause_before_ms: 150
    pause_after_ms: 250
```

Les coordonnées doivent être normalisées entre 0 et 1, afin de fonctionner quelle que soit la résolution de la vidéo. Pour les interfaces fixes, elles peuvent être définies manuellement. Pour les interfaces variables, une détection visuelle peut proposer une zone, mais la validation automatique seule est insuffisante.

### Étape D : voix off IA

Le script doit envoyer chaque segment anglais au moteur de synthèse vocale et récupérer :

- le fichier audio ;
- la durée réelle ;
- les timecodes de mots ou de phonèmes, si le fournisseur les expose ;
- les métadonnées de voix et de génération.

Choix possibles :

- **Moteur vocal local** : confidentialité maximale et absence de dépendance fournisseur, mais qualité et naturel à vérifier sur les termes SAP ;
- **Moteur local** : contrôle des données et reproductibilité, mais qualité vocale et installation potentiellement plus complexes ;
- **Voix humaine** : qualité maximale, mais hors du périmètre automatisé actuel.

**Choix recommandé** : moteur vocal local Kokoro ou Piper, avec génération par segments, cache local des résultats et contrôle explicite de la prononciation des termes SAP. Cette option doit faire l'objet d'un test d'écoute sur plusieurs segments, car la qualité perçue et la prononciation des noms SAP peuvent varier selon le modèle et la voix sélectionnée.

Le pipeline doit générer `narration_manifest.json` :

```json
{
  "segment_id": "seg-001",
  "audio_file": "audio/seg-001.wav",
  "duration": 4.92,
  "provider": "premium-tts",
  "voice": "professional_female_01"
}
```

### Étape E : recalage temporel

Calculer les nouveaux intervalles à partir de la durée réelle de chaque segment audio.

Priorité d'ajustement :

1. reformuler légèrement le texte anglais ;
2. modifier les pauses naturelles ;
3. maintenir ou prolonger un plan lorsque le contenu visuel le permet ;
4. modifier la vitesse audio très légèrement, dans une limite définie ;
5. éviter les compressions audibles ou les accélérations excessives.

Le système doit détecter :

- un segment audio plus long que le plan disponible ;
- un chevauchement entre deux segments ;
- une annotation située hors de la durée de la scène ;
- une différence entre la durée vidéo et la durée audio finale.

### Étape F : génération des effets

Définir des types d'effets paramétrables :

- `zoom` : agrandissement progressif d'une zone ;
- `highlight` : cadre ou halo autour d'un champ ;
- `callout` : encadré textuel relié à une zone ;
- `popup` : information courte apparaissant près de l'interface ;
- `cursor_emphasis` : mise en valeur du pointeur si celui-ci est visible.

Règles techniques :

- utiliser des coordonnées relatives à l'image ;
- conserver une marge autour des contrôles SAP ;
- ne jamais masquer le champ expliqué ;
- limiter le nombre d'animations simultanées ;
- prévoir une durée d'entrée et de sortie ;
- utiliser une zone réservée aux sous-titres en bas de l'image.

### Étape G : sous-titres

Produire d'abord un fichier `subtitles_en.srt` ou `subtitles_en.vtt`, puis l'incruster dans le rendu final avec FFmpeg.

Règles recommandées :

- deux lignes maximum ;
- longueur limitée pour conserver une lecture confortable ;
- segmentation alignée sur les groupes de sens ;
- timecodes issus de l'audio anglais final ;
- police sans empattement lisible.

### Étape H : rendu final

Commande conceptuelle :

```
source video + narration mix + overlays + subtitles
  -> FFmpeg filter graph
  -> master video
  -> delivery MP4 H.264/AAC
```

Paramètres de départ recommandés :

- conteneur : MP4 ;
- vidéo : H.264, profil adapté à la résolution source ;
- audio : AAC, 48 kHz ;
- fréquence d'images : identique à la source ;
- conservation du ratio et des pixels de l'interface SAP ;
- débit vidéo à ajuster selon la résolution et la qualité attendue.

Produire également un master de meilleure qualité si le fichier doit être réédité ultérieurement.

## 4. Contrôles automatisés

Avant livraison, exécuter un contrôle de qualité qui vérifie :

- présence des pistes audio et vidéo ;
- durée audio et vidéo ;
- absence de segments qui se chevauchent ;
- validité des timecodes.

Un contrôle audio peut mesurer les silences anormaux, les crêtes excessives et les niveaux incohérents entre segments.

Le contrôle visuel doit rester humain pour confirmer :

- la pertinence du zoom ;
- la lisibilité de l'interface ;
- le naturel de la voix ;
- la prononciation SAP ;
- l'absence de décalage perceptible ;
- la qualité des animations et la non-obstruction des éléments importants.

## 5. Organisation des fichiers

```
project/
  input/
    source.mp4
  data/
    source_metadata.json
    transcript_fr.json
    edit_decision_list.yaml
    narration_manifest.json
    subtitles_en.srt
    glossary.yaml
  audio/
    source_audio.wav
    narration/
  frames/
  overlays/
  scripts/
    inspect_source.py
    transcribe.py
    build_narration.py
    build_timeline.py
    render_video.py
    validate_output.py
  output/
    preview.mp4
    final.mp4
    final.srt
  logs/
```

## 6. Décisions techniques à prendre

### Cloud ou local

- **Local 100 %** : obligatoire pour ce projet afin de préserver la confidentialité des données SAP ; demande une machine suffisamment puissante et davantage d'installation.

La configuration finale dépend du matériel disponible. Il faudra relever le processeur, la RAM, le GPU, la VRAM et l'espace disque avant de sélectionner les modèles.

### Montage FFmpeg ou éditeur dédié

- **FFmpeg** : meilleur choix pour un rendu automatisé, reproductible et piloté par données.
- **DaVinci Resolve ou Premiere** : meilleur confort pour une retouche humaine, mais automatisation et déploiement plus complexes.

### Détection automatique des champs

- **Coordonnées définies dans le conducteur** : robuste pour une interface connue et stable.
- **Vision par ordinateur** : utile pour proposer les zones, mais nécessite une vérification et peut échouer sur des changements d'écran ou de thème.

**Choix recommandé pour une première version** : coordonnées normalisées définies dans le conducteur, avec détection automatique utilisée uniquement comme aide.

## 7. Découpage de réalisation

1. Construire un prototype sur une séquence de 30 à 60 secondes.
2. Valider la qualité de transcription, traduction, voix et synchronisation.
3. Implémenter les modèles de zoom, highlight et pop-up.
4. Ajouter les validations automatiques et les journaux d'exécution.
5. Traiter la vidéo complète.
6. Générer une version de prévisualisation basse résolution.
7. Effectuer le contrôle final et produire les exports de livraison.

Le prototype est important : il permet de tester le naturel de la voix et la lisibilité des overlays avant de générer la totalité de l'audio et du rendu.

## 8. Risques techniques

- L'interface SAP peut être trop petite ou compressée pour permettre un zoom propre.
- Une traduction plus longue que le français peut dépasser les plans disponibles.
- Une prononciation incorrecte d'un terme métier peut rendre la voix IA inutilisable.
- Des coordonnées fixes peuvent devenir fausses si la vidéo contient des changements de résolution ou de fenêtre.
- L'incrustation des sous-titres empêche leur désactivation après export.
- Une automatisation sans validation intermédiaire peut reporter les erreurs jusqu'au rendu final.
- Le traitement local doit être vérifié afin de confirmer qu'aucun fichier ou appel réseau ne transmet les données visibles dans SAP.

## 9. Stack recommandée en synthèse

- Python 3.12+
- FFmpeg et ffprobe
- `faster-whisper` local, modèle `small` par défaut
- Modèle de langage local Qwen 3B/4B quantifié via `llama.cpp` ou Ollama
- Kokoro ou Piper pour une voix locale
- YAML/JSON avec schéma validé
- Git pour scripts et données textuelles
- Stockage séparé pour médias lourds
- Exécution locale ou conteneurisée selon les contraintes de confidentialité

Cette stack privilégie la reproductibilité, la possibilité de corriger chaque étape séparément et un rendu automatisé contrôlable, tout en conservant une vérification humaine finale indispensable.

## 10. Mise en place et reproductibilité

Objectif : qu'une personne puisse cloner le dépôt, lancer une seule commande d'installation, puis une seule commande d'exécution, sans configuration manuelle complexe.

### Point d'entrée unique

- Fournir un CLI unique (ex. `python run.py --input source.mp4`) qui enchaîne automatiquement les étapes A à H.
- Conserver la possibilité de lancer chaque étape séparément (`python run.py --step transcribe`) pour le débogage et les reprises partielles, conformément au pipeline détaillé en section 3.
- Un `Makefile` ou des scripts `setup.sh` / `setup.ps1` (Windows/Linux) peuvent envelopper les commandes courantes : installation, test rapide, exécution complète.

### Installation automatisée des modèles

- Écrire un script `download_models.py` qui télécharge les poids nécessaires (Whisper, LLM GGUF, voix TTS) avec vérification de checksum (SHA-256) avant utilisation.
- Ne jamais committer les modèles ou fichiers lourds dans Git ; les stocker dans un dossier ignoré (`models/`, ajouté au `.gitignore`) et les télécharger à la demande.
- Documenter la taille et le temps de téléchargement approximatifs de chaque modèle, pour que l'utilisateur sache à quoi s'attendre.

### Dépendances verrouillées

- Utiliser un fichier de dépendances figé (`requirements.txt` généré avec versions exactes, ou `poetry.lock` / `uv.lock`) pour garantir un environnement identique d'une machine à l'autre.
- Isoler l'environnement Python dans un virtualenv créé automatiquement par le script de setup.

### Configuration centralisée

- Regrouper tous les réglages modifiables (chemins, taille du modèle Whisper, voix choisie, seuils de recalage, paramètres d'export) dans un fichier unique `config.yaml` avec des valeurs par défaut fonctionnelles dès l'installation.
- Éviter de disperser des paramètres dans plusieurs fichiers ou variables d'environnement séparées ; un seul fichier à éditer si besoin de personnalisation.

### Détection automatique du matériel

- Détecter au lancement la présence d'un GPU compatible (CUDA/ROCm) et la VRAM disponible, puis sélectionner automatiquement la configuration adaptée (voir section 2, configurations indicatives), avec repli sur CPU si aucun GPU n'est détecté.
- Journaliser la configuration matérielle détectée et les modèles choisis en conséquence, pour la traçabilité.

### Exemple minimal de validation

- Fournir dans le dépôt un court extrait vidéo d'exemple (quelques secondes, sans donnée SAP réelle) accompagné de sa sortie attendue.
- Un utilisateur doit pouvoir lancer ce cas d'exemple juste après l'installation pour vérifier en quelques minutes que le pipeline fonctionne correctement, avant de traiter sa propre vidéo.

### Conteneurisation (optionnelle)

- Évaluer un `Dockerfile` figeant FFmpeg, Python et les dépendances système, pour éliminer les écarts d'environnement entre machines.
- Le passthrough GPU dans Docker ajoute de la complexité d'installation (pilotes, runtime NVIDIA) ; à proposer comme option avancée plutôt que comme chemin d'installation par défaut, pour ne pas complexifier le cas courant.

### Documentation minimale attendue

- Un `README.md` avec : prérequis matériel, commande d'installation, commande d'exécution sur l'exemple, commande d'exécution sur une vidéo réelle, et emplacement des sorties.
- Un tableau récapitulatif des temps de traitement approximatifs selon la configuration matérielle, pour fixer les attentes de l'utilisateur.
