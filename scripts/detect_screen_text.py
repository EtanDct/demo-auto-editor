"""Étape A2 : index du texte affiché à l'écran.

Première brique du montage automatique : produire l'inventaire de ce qui est
écrit à l'écran et *quand*, pour pouvoir plus tard rapprocher ce que dit le
narrateur de ce que montre l'interface. Cette étape ne décide d'aucune
incrustation : elle constate.

Sortie : `data/screen_elements.json` (schemas.ScreenTextIndex).

Deux points de conception :

1. **L'OCR n'est lancé que sur les frames qui changent.** Une capture d'écran
   est immobile la plupart du temps ; l'OCR coûte ~5 s par image contre ~12 ms
   pour un écart de frames. On échantillonne donc densément (2 im/s par défaut)
   mais on ne repasse l'OCR que lorsque l'image bouge assez, les autres frames
   héritant des détections précédentes. Sans ça, l'étape coûterait une demi
   heure sur trois minutes de vidéo.

2. **Les détections sont agrégées en éléments stables.** Un même bouton
   détecté sur 40 frames consécutives doit devenir *un* élément avec une plage
   de visibilité, pas 40 lignes. Le regroupement se fait sur le texte normalisé
   et le recouvrement des boîtes (IoU) ; deux occurrences simultanées du même
   libellé restent donc deux éléments distincts, ce qui permettra à l'étape
   d'appariement de détecter l'ambiguïté au lieu de tirer à pile ou face.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import typer

from pipeline_config import PipelineConfig, load_config
from schemas import BoundingBox, ScreenElement, ScreenTextIndex

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

SAMPLES_SUBDIR = "screen_text"


@dataclass
class Detection:
    """Une détection OCR brute, rapportée à un instant de la vidéo."""

    text: str
    box: BoundingBox
    confidence: float
    timestamp: float


def normalize_text(text: str) -> str:
    """Forme lisible canonique : sans casse, sans accents, espaces normalisés."""
    stripped = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(c for c in stripped if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", without_accents)


def match_key(text: str) -> str:
    """Clé d'identité d'un libellé, espaces compris.

    L'OCR colle les mots de façon instable d'une frame à l'autre selon le
    crénage : sur l'extrait de référence, le même libellé ressort tantôt
    "Top repositories", tantôt "Toprepositories". Les traiter comme deux
    éléments distincts fragmente l'index et fabrique de fausses ambiguïtés,
    donc la clé ignore les espaces. Le regroupement reste contraint par le
    recouvrement des boîtes, ce qui borne le risque de fusion abusive.
    """
    return normalize_text(text).replace(" ", "")


def load_inspection_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Rapport d'inspection introuvable : {path}. Lance d'abord "
            "'python run.py --step inspect'."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def extract_sample_frames(
    video_path: Path, out_dir: Path, sample_fps: float, max_seconds: float | None = None
) -> list[Path]:
    """Échantillonne la vidéo à `sample_fps` images par seconde dans `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("sample_*.jpg"):
        existing.unlink()

    cmd = ["ffmpeg", "-y"]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-i", str(video_path), "-vf", f"fps={sample_fps}", str(out_dir / "sample_%05d.jpg")]
    subprocess.run(cmd, capture_output=True, check=True)

    return sorted(out_dir.glob("sample_*.jpg"))


def changed_pixel_ratio(previous_gray, current_gray, pixel_delta: int) -> float:
    """Fraction de pixels ayant changé de plus de `pixel_delta` entre deux frames.

    On ne se sert pas de l'écart *moyen* : les changements qui nous intéressent
    (un menu qui s'ouvre, une ligne surlignée) sont localisés, et leur moyenne
    sur toute l'image se noie dans le bruit de compression. Mesuré sur 40 s de
    l'extrait de référence, l'écart moyen ne dépasse jamais 13/255 alors que la
    proportion de pixels touchés sépare nettement l'écran figé (~0.002 %) des
    vrais changements (0.05 % à 3.6 %).
    """
    import cv2
    import numpy as np

    return float(np.mean(cv2.absdiff(previous_gray, current_gray) > pixel_delta))


def _quad_to_box(quad, width: int, height: int) -> BoundingBox:
    """Convertit le quadrilatère rendu par l'OCR en rectangle normalisé."""
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(height), max(ys))
    return BoundingBox(
        x=x0 / width,
        y=y0 / height,
        # Un texte d'un pixel de large n'existe pas : on garde un plancher pour
        # rester dans le domaine de validité du schéma (width/height > 0).
        width=max(x1 - x0, 1.0) / width,
        height=max(y1 - y0, 1.0) / height,
    )


def is_usable_label(text: str, min_length: int) -> bool:
    """Écarte le bruit d'OCR sur les icônes.

    Une interface est pleine de pictogrammes que l'OCR rend en symboles isolés
    ("口", "←", "★", "N"). Ils n'ont aucune chance d'être le libellé qu'un
    narrateur prononce, mais ils polluent l'index et créent des candidats
    parasites à l'appariement. On exige donc une longueur minimale et au moins
    un caractère alphanumérique.
    """
    stripped = text.strip()
    return len(stripped) >= min_length and any(c.isalnum() for c in stripped)


def ocr_frame(
    ocr, frame_path: Path, width: int, height: int, min_confidence: float, min_text_length: int
) -> list[Detection]:
    raw, _ = ocr(str(frame_path))
    detections = []
    for quad, text, confidence in raw or []:
        if confidence < min_confidence or not is_usable_label(text, min_text_length):
            continue
        detections.append(
            Detection(
                text=text.strip(),
                box=_quad_to_box(quad, width, height),
                confidence=float(confidence),
                timestamp=0.0,  # renseigné par l'appelant, qui connaît le pas d'échantillonnage
            )
        )
    return detections


class _Track:
    """Un élément en cours d'observation, étendu tant qu'on le revoit."""

    def __init__(self, detection: Detection):
        self.best_text = detection.text
        self.best_confidence = detection.confidence
        self.box = detection.box
        self.first_seen = detection.timestamp
        self.last_seen = detection.timestamp
        self.confidence_sum = detection.confidence
        self.occurrences = 1

    def extend(self, detection: Detection) -> None:
        self.last_seen = detection.timestamp
        self.confidence_sum += detection.confidence
        self.occurrences += 1
        # On retient la graphie la mieux reconnue plutôt que la première vue :
        # c'est elle qu'on comparera au libellé annoncé par le narrateur.
        if detection.confidence > self.best_confidence:
            self.best_text = detection.text
            self.best_confidence = detection.confidence
            self.box = detection.box

    def to_element(self, index: int, frame_interval: float) -> ScreenElement:
        return ScreenElement(
            id=f"scr-{index:04d}",
            text=self.best_text,
            box=self.box,
            first_seen=round(self.first_seen, 3),
            # Un élément vu sur une seule frame est visible au moins jusqu'à la
            # frame suivante : sans ça sa plage serait vide et il n'apparaîtrait
            # visible pendant aucun segment.
            last_seen=round(max(self.last_seen, self.first_seen + frame_interval), 3),
            confidence=round(min(1.0, self.confidence_sum / self.occurrences), 4),
            occurrences=self.occurrences,
        )


def group_into_elements(
    frames: list[list[Detection]], merge_iou: float, max_gap_seconds: float, frame_interval: float
) -> list[ScreenElement]:
    """Agrège les détections frame par frame en éléments stables dans le temps.

    Une détection prolonge un élément ouvert si sa clé de libellé (`match_key`)
    est identique et que les boîtes se recouvrent (IoU >= `merge_iou`). Le même
    libellé affiché à deux endroits produit donc deux éléments : c'est
    volontaire, l'ambiguïté doit rester visible en aval plutôt qu'être tranchée
    ici au hasard.

    Une même frame ne peut pas prolonger deux fois le même élément (`claimed`),
    sinon un libellé dupliqué à l'écran s'effondrerait en un seul élément dont
    la boîte sauterait d'un endroit à l'autre.
    """
    open_tracks: list[_Track] = []
    finished: list[_Track] = []

    for detections in frames:
        if not detections:
            continue
        current_time = detections[0].timestamp

        expired = [t for t in open_tracks if current_time - t.last_seen > max_gap_seconds]
        finished.extend(expired)
        open_tracks = [t for t in open_tracks if t not in expired]

        claimed: set[int] = set()
        for detection in detections:
            key = match_key(detection.text)
            candidates = [
                (t.box.iou(detection.box), t)
                for t in open_tracks
                if id(t) not in claimed and match_key(t.best_text) == key
            ]
            best = max(candidates, key=lambda pair: pair[0], default=(0.0, None))
            if best[1] is not None and best[0] >= merge_iou:
                best[1].extend(detection)
                claimed.add(id(best[1]))
            else:
                track = _Track(detection)
                open_tracks.append(track)
                claimed.add(id(track))

    finished.extend(open_tracks)
    finished.sort(key=lambda t: (t.first_seen, t.box.y, t.box.x))
    return [t.to_element(i + 1, frame_interval) for i, t in enumerate(finished)]


def write_detections(
    analysed: list[tuple[float, list[Detection]]], frames_sampled: int, out_path: Path
) -> None:
    """Met en cache les détections OCR brutes, par frame réellement analysée.

    L'OCR est le seul coût lourd de l'étape (plusieurs minutes) alors que le
    regroupement se règle par essais successifs (`merge_iou`, `max_gap_seconds`).
    On garde donc la sortie brute pour pouvoir reconstruire l'index sans
    repasser par l'OCR (`--regroup`). Seules les frames analysées sont
    stockées : les autres sont identiques à la précédente par construction.
    """
    payload = {
        "frames_sampled": frames_sampled,
        "analysed": [
            {
                "timestamp": timestamp,
                "detections": [
                    {"text": d.text, "box": d.box.model_dump(), "confidence": d.confidence}
                    for d in detections
                ],
            }
            for timestamp, detections in analysed
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Détections OCR brutes mises en cache dans %s", out_path)


def load_detections(path: Path, frame_interval: float) -> tuple[list[list[Detection]], int]:
    """Reconstruit la séquence frame par frame depuis le cache de `write_detections`."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cache des détections introuvable : {path}. Lance d'abord l'étape complète "
            "('python run.py --step screen') avant de rejouer le regroupement."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    analysed = [
        (
            entry["timestamp"],
            [
                Detection(
                    text=d["text"],
                    box=BoundingBox.model_validate(d["box"]),
                    confidence=d["confidence"],
                    timestamp=0.0,
                )
                for d in entry["detections"]
            ],
        )
        for entry in payload["analysed"]
    ]
    return expand_to_frames(analysed, payload["frames_sampled"], frame_interval), len(analysed)


def expand_to_frames(
    analysed: list[tuple[float, list[Detection]]], frames_sampled: int, frame_interval: float
) -> list[list[Detection]]:
    """Réétend les frames analysées sur toutes les frames échantillonnées.

    Chaque frame non analysée hérite de la dernière analysée qui la précède :
    par construction, l'écran n'y a pas assez changé pour valoir un OCR.
    """
    frames: list[list[Detection]] = []
    cursor = -1
    for i in range(frames_sampled):
        timestamp = round(i * frame_interval, 3)
        while cursor + 1 < len(analysed) and analysed[cursor + 1][0] <= timestamp + 1e-9:
            cursor += 1
        source = analysed[cursor][1] if cursor >= 0 else []
        frames.append([Detection(d.text, d.box, d.confidence, timestamp) for d in source])
    return frames


def select_frames_to_analyse(
    frame_count: int,
    load_gray,
    pixel_delta: int,
    change_ratio: float,
    max_frames: int,
) -> tuple[list[int], float]:
    """Choisit les frames à passer à l'OCR, sous un plafond de coût.

    Le seuil `change_ratio` par défaut est calibré sur une capture d'écran
    ordinaire, très statique. Une source au mouvement permanent (lecteur vidéo,
    animation, incrustation webcam, indicateur de chargement) ferait franchir le
    seuil à *toutes* les frames et l'étape durerait des dizaines de minutes.

    On relève donc le seuil jusqu'à tenir sous `max_frames`, ce qui borne la
    durée de l'étape quel que soit le contenu : on analyse toujours les
    changements les plus francs, seuls les plus ténus sont sacrifiés. Le seuil
    effectivement retenu est renvoyé pour être journalisé — s'il a été relevé,
    c'est le signe que la vidéo mérite un réglage explicite.
    """
    threshold = change_ratio
    selected = _greedy_selection(frame_count, load_gray, pixel_delta, threshold)

    while len(selected) > max_frames and threshold < 1.0:
        threshold *= 2
        selected = _greedy_selection(frame_count, load_gray, pixel_delta, threshold)
        logger.warning(
            "Trop de frames à analyser pour le plafond de %d : seuil de changement "
            "relevé à %.5f (%d frames retenues). Source plus animée qu'une capture "
            "d'écran classique ; ajuste screen_text.change_ratio si le résultat est "
            "trop grossier.",
            max_frames, threshold, len(selected),
        )

    return selected[:max_frames], threshold


def _greedy_selection(frame_count: int, load_gray, pixel_delta: int, threshold: float) -> list[int]:
    """Frames dont l'écran s'écarte assez de la dernière frame retenue.

    La comparaison porte sur la dernière frame *retenue*, pas sur la précédente :
    un défilement lent finit ainsi par déclencher un nouvel OCR, alors qu'il
    resterait indéfiniment sous le seuil frame à frame.
    """
    selected: list[int] = []
    reference = None
    for i in range(frame_count):
        gray = load_gray(i)
        if reference is None or changed_pixel_ratio(reference, gray, pixel_delta) >= threshold:
            selected.append(i)
            reference = gray
    return selected


def detect(config: PipelineConfig, max_seconds: float | None = None) -> ScreenTextIndex:
    import cv2

    from rapidocr_onnxruntime import RapidOCR

    data_dir = config.paths.resolve("data_dir")
    inspection = load_inspection_report(data_dir / "inspection_report.json")
    video_path = Path(inspection["video_path"])
    width, height = (int(v) for v in inspection["resolution"].split("x"))

    settings = config.screen_text
    frame_interval = 1.0 / settings.sample_fps
    samples_dir = config.paths.resolve("frames_dir") / SAMPLES_SUBDIR

    logger.info("Échantillonnage de %s à %s im/s", video_path, settings.sample_fps)
    frame_paths = extract_sample_frames(video_path, samples_dir, settings.sample_fps, max_seconds)
    logger.info("%d frames échantillonnées dans %s", len(frame_paths), samples_dir)

    def load_gray(index: int):
        return cv2.imread(str(frame_paths[index]), cv2.IMREAD_GRAYSCALE)

    selected, threshold = select_frames_to_analyse(
        len(frame_paths),
        load_gray,
        settings.change_pixel_delta,
        settings.change_ratio,
        settings.max_ocr_frames,
    )
    logger.info(
        "%d frames à passer à l'OCR / %d échantillonnées (%.0f%% évitées, seuil %.5f)",
        len(selected), len(frame_paths),
        100 * (1 - len(selected) / len(frame_paths)) if frame_paths else 0,
        threshold,
    )

    ocr = RapidOCR()
    analysed_frames: list[tuple[float, list[Detection]]] = []
    for position, index in enumerate(selected, start=1):
        timestamp = round(index * frame_interval, 3)
        detections = ocr_frame(
            ocr, frame_paths[index], width, height, settings.min_confidence, settings.min_text_length
        )
        analysed_frames.append((timestamp, detections))
        if position % 10 == 0:
            logger.info("  OCR %d/%d (t=%.1fs)", position, len(selected), timestamp)

    write_detections(analysed_frames, len(frame_paths), data_dir / "screen_detections.json")

    frames = expand_to_frames(analysed_frames, len(frame_paths), frame_interval)
    analysed = len(analysed_frames)
    elements = group_into_elements(
        frames, settings.merge_iou, settings.max_gap_seconds, frame_interval
    )
    return ScreenTextIndex(
        sample_fps=settings.sample_fps,
        frames_sampled=len(frame_paths),
        frames_analysed=analysed,
        elements=elements,
    )


def regroup(config: PipelineConfig) -> ScreenTextIndex:
    """Reconstruit l'index depuis le cache OCR, sans relancer l'OCR."""
    settings = config.screen_text
    frame_interval = 1.0 / settings.sample_fps
    data_dir = config.paths.resolve("data_dir")

    frames, analysed = load_detections(data_dir / "screen_detections.json", frame_interval)
    logger.info("Regroupement depuis le cache (%d frames, %d analysées)", len(frames), analysed)

    elements = group_into_elements(
        frames, settings.merge_iou, settings.max_gap_seconds, frame_interval
    )
    return ScreenTextIndex(
        sample_fps=settings.sample_fps,
        frames_sampled=len(frames),
        frames_analysed=analysed,
        elements=elements,
    )


def write_index(index: ScreenTextIndex, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Index du texte à l'écran écrit dans %s (%d éléments)", out_path, len(index.elements))


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Chemin vers config.yaml."),
    max_seconds: float = typer.Option(
        None, "--max-seconds", help="N'analyser que les N premières secondes (mise au point)."
    ),
    regroup_only: bool = typer.Option(
        False,
        "--regroup",
        help="Reconstruire l'index depuis le cache OCR, sans relancer l'OCR "
        "(pour régler merge_iou / max_gap_seconds).",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    index = regroup(config) if regroup_only else detect(config, max_seconds=max_seconds)
    write_index(index, config.paths.resolve("data_dir") / "screen_elements.json")


if __name__ == "__main__":
    app()
