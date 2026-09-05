"""Étape A3 : suivi du pointeur de souris.

Second signal du montage automatique, indépendant du texte. Dans une démo, le
narrateur amène la souris sur l'élément dont il parle, souvent avant même de le
nommer. Quand deux libellés identiques se disputent une incrustation, la
position du pointeur tranche — et comme elle ne repose sur aucune comparaison
de chaînes, elle reste valable quand la narration et l'interface ne sont pas
dans la même langue.

Sortie : `data/cursor_track.json`, une position (ou rien) par frame
échantillonnée.

**Principe.** Aucun gabarit de curseur n'est utilisé : la forme change selon le
contexte (flèche, main, barre de saisie), le thème et la mise à l'échelle. On
part de ce qui est vrai dans tous les cas — le pointeur est une petite chose
qui se déplace sur un écran par ailleurs immobile.

Un pointeur qui bouge laisse **deux** taches dans l'écart entre deux frames :
là où il était, là où il est arrivé. Rien ne les distingue dans ce seul écart.
Mais sa position à l'instant *i* est la seule qui apparaisse à la fois dans
`diff(i-1, i)` et dans `diff(i, i+1)` : c'est cette intersection qui l'isole,
sans jamais avoir à reconnaître sa forme.

**Limites assumées.** Un pointeur immobile ne produit aucune tache et reste
introuvable ; sa dernière position connue est alors reconduite, dans la limite
de `hold_seconds`. Une frame où plusieurs petites zones changent au même
endroit (animation, curseur de saisie clignotant) ne donne aucune position
plutôt qu'une position douteuse.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from detect_screen_text import extract_sample_frames, load_inspection_report
from pipeline_config import PipelineConfig, load_config
from schemas import CursorSample, CursorTrack

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

SAMPLES_SUBDIR = "cursor"


def _moving_blobs(previous_gray, current_gray, pixel_delta: int, min_area: int, max_area: int):
    """Centres des petites zones ayant changé entre deux frames."""
    import cv2
    import numpy as np

    mask = (cv2.absdiff(previous_gray, current_gray) > pixel_delta).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [
        (float(centroids[i][0]), float(centroids[i][1]))
        for i in range(1, count)
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area
    ]


def _common_position(before, after, tolerance: float):
    """Unique position présente dans les deux écarts, ou None si c'est indécidable.

    `before` porte les taches de `diff(i-1, i)`, `after` celles de
    `diff(i, i+1)`. Le pointeur à l'instant *i* est dans les deux. Deux
    appariements possibles signifient que quelque chose d'autre bouge au même
    endroit : on préfère ne rien dire.
    """
    pairs = [
        (bx, by)
        for bx, by in before
        for ax, ay in after
        if abs(bx - ax) <= tolerance and abs(by - ay) <= tolerance
    ]
    if len(pairs) != 1:
        return None
    return pairs[0]


def track_cursor(
    frame_count: int, load_gray, config: PipelineConfig, frame_interval: float
) -> list[CursorSample]:
    settings = config.cursor
    samples: list[CursorSample] = []
    if frame_count < 3:
        return samples

    height, width = load_gray(0).shape[:2]
    min_area = max(1, int(settings.min_area_fraction * width * height))
    max_area = max(min_area + 1, int(settings.max_area_fraction * width * height))
    tolerance = settings.match_tolerance_fraction * max(width, height)

    previous = load_gray(0)
    current = load_gray(1)
    for i in range(1, frame_count - 1):
        following = load_gray(i + 1)
        before = _moving_blobs(previous, current, settings.pixel_delta, min_area, max_area)
        after = _moving_blobs(current, following, settings.pixel_delta, min_area, max_area)

        position = _common_position(before, after, tolerance)
        if position is not None:
            samples.append(
                CursorSample(
                    timestamp=round(i * frame_interval, 3),
                    x=round(position[0] / width, 5),
                    y=round(position[1] / height, 5),
                )
            )

        previous, current = current, following

    return samples


def position_at(track: CursorTrack, timestamp: float, hold_seconds: float):
    """Position du pointeur à un instant, en reconduisant la dernière connue.

    Un pointeur immobile ne se détecte pas, mais il est toujours là : sa
    dernière position reste valable tant qu'elle n'est pas trop ancienne.
    """
    latest = None
    for sample in track.samples:
        if sample.timestamp > timestamp:
            break
        latest = sample
    if latest is None or timestamp - latest.timestamp > hold_seconds:
        return None
    return latest.x, latest.y


def detect(config: PipelineConfig) -> CursorTrack:
    import cv2

    data_dir = config.paths.resolve("data_dir")
    inspection = load_inspection_report(data_dir / "inspection_report.json")
    video_path = Path(inspection["video_path"])
    samples_dir = config.paths.resolve("frames_dir") / SAMPLES_SUBDIR
    frame_interval = 1.0 / config.cursor.sample_fps

    logger.info(
        "Échantillonnage de %s à %s im/s pour le suivi du pointeur",
        video_path, config.cursor.sample_fps,
    )
    frame_paths = extract_sample_frames(video_path, samples_dir, config.cursor.sample_fps)

    def load_gray(index: int):
        return cv2.imread(str(frame_paths[index]), cv2.IMREAD_GRAYSCALE)

    samples = track_cursor(len(frame_paths), load_gray, config, frame_interval)
    logger.info(
        "Pointeur localisé sur %d frames / %d (%.0f%%)",
        len(samples), len(frame_paths),
        100 * len(samples) / len(frame_paths) if frame_paths else 0,
    )
    return CursorTrack(sample_fps=config.cursor.sample_fps, samples=samples)


def write_track(track: CursorTrack, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(track.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Trajectoire du pointeur écrite dans %s", out_path)


def load_track(path: Path) -> CursorTrack:
    if not path.exists():
        raise FileNotFoundError(
            f"Trajectoire du pointeur introuvable : {path}. Lance d'abord "
            "'python run.py --step cursor'."
        )
    return CursorTrack.model_validate_json(path.read_text(encoding="utf-8"))


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    track = detect(config)
    write_track(track, config.paths.resolve("data_dir") / "cursor_track.json")


if __name__ == "__main__":
    app()
