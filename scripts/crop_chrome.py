"""Étape A1 : retrait du bandeau de navigateur.

Produit une vidéo de travail dont le haut figé — onglets, URL, favoris — a été
retiré, pour ne garder que la page présentée.

**Principe.** Aucune mise en page n'est encodée : le détecteur ne sait pas ce
qu'est une barre de favoris, ni un onglet, ni quel navigateur est utilisé. Il
ne pose qu'une question, ligne par ligne : à partir d'où l'image cesse-t-elle
d'être figée ? Le bandeau d'un navigateur ne change pas pendant que la page
change constamment, et cette propriété vaut quels que soient le navigateur, le
thème, la langue et la résolution.

Vérifié en fabriquant les autres mises en page à partir d'une source réelle :
une barre de favoris en moins fait remonter la frontière d'exactement sa
hauteur, une barre en plus la fait descendre d'autant, une capture plein écran
ne donne aucun recadrage.

**Garde-fous.** Le mode d'échec à éviter est de rogner un en-tête applicatif
figé — la barre supérieure de SAP Fiori, par exemple, fait partie du produit
montré. D'où un plafond (`max_fraction`), l'exigence d'une transition franche
et durable plutôt que graduelle, une image de contrôle systématique, et la
possibilité de figer ou d'annuler la valeur dans `config.yaml`. Tous les autres
cas — plein écran, changements d'onglets fréquents, redimensionnement en cours
d'enregistrement — échouent du côté sûr : on rogne moins, voire pas du tout.

**Place dans le pipeline.** Le recadrage doit intervenir tôt. Tout l'aval
travaille en coordonnées normalisées sur l'image : boîtes OCR, positions du
pointeur, incrustations. Recadrer au rendu les décalerait toutes. On produit
donc une vidéo de travail et on pointe le rapport d'inspection dessus : un seul
repère, rien à changer ailleurs.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import typer

from detect_screen_text import extract_sample_frames, load_inspection_report
from pipeline_config import PipelineConfig, load_config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

SAMPLES_SUBDIR = "crop"
WORK_VIDEO_NAME = "cropped.mp4"


def detect_chrome_height(frames, max_fraction: float, run_rows: int) -> tuple[int, str]:
    """Hauteur du bandeau figé, en pixels, avec le motif de la décision."""
    import numpy as np

    stack = np.stack(frames)
    height = stack.shape[1]
    row_variation = stack.std(axis=0).mean(axis=1)

    # Le régime « contenu » est estimé sur les deux tiers bas, qui ne peuvent
    # pas être du bandeau ; le seuil se place à la moitié de ce niveau.
    threshold = row_variation[height // 3 :].mean() * 0.5
    if threshold <= 0:
        return 0, "image entièrement figée : rien à distinguer"

    run = 0
    boundary = None
    for index, value in enumerate(row_variation):
        run = run + 1 if value >= threshold else 0
        if run >= run_rows:
            boundary = index - run_rows + 1
            break

    if boundary is None:
        return 0, "aucune zone de contenu franche détectée"
    if boundary == 0:
        return 0, "la page commence au bord supérieur : aucun bandeau"
    if boundary > height * max_fraction:
        return 0, (
            f"bande figée de {boundary}px ({boundary / height:.0%} de la hauteur), au-delà du "
            f"plafond de {max_fraction:.0%} : probablement un en-tête applicatif, on ne rogne pas"
        )

    calm = row_variation[:boundary].mean()
    content = row_variation[boundary:].mean()
    return boundary, (
        f"bandeau figé sur {boundary}px ({boundary / height:.1%} de la hauteur) ; "
        f"variation {calm:.2f} au-dessus contre {content:.2f} en dessous"
    )


def resolve_crop_top(config: PipelineConfig, frames) -> tuple[int, str]:
    """Applique le réglage : `auto`, `off`, ou une hauteur imposée en pixels."""
    mode = str(config.crop.top).strip().lower()
    if mode in {"off", "none", "0"}:
        return 0, "recadrage désactivé dans config.yaml"
    if mode != "auto":
        return int(mode), f"hauteur imposée dans config.yaml ({mode}px)"
    return detect_chrome_height(frames, config.crop.max_fraction, config.crop.run_rows)


def write_preview(frame_path: Path, crop_top: int, out_path: Path) -> None:
    """Image de contrôle : la frontière tracée, et le résultat en dessous."""
    import cv2
    import numpy as np

    image = cv2.imread(str(frame_path))
    marked = image.copy()
    cv2.line(marked, (0, crop_top), (image.shape[1], crop_top), (0, 0, 255), 3)
    cropped = image[crop_top:, :]
    padded = cv2.copyMakeBorder(
        cropped, 0, crop_top, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
    )
    width = 820
    scale = width / image.shape[1]
    size = (width, int(image.shape[0] * scale))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(out_path), np.vstack([cv2.resize(marked, size), cv2.resize(padded, size)])
    )


def encode_cropped(video_path: Path, crop_top: int, out_path: Path, config: PipelineConfig) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"crop=w=iw:h=ih-{crop_top}:x=0:y={crop_top}",
            "-c:v", config.export.video_codec,
            # Qualité supérieure à celle du rendu : cette vidéo sera réencodée
            # une seconde fois, et deux passages à la même qualité se voient.
            "-crf", str(config.crop.intermediate_crf),
            "-c:a", "copy",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )


def probe_resolution(video_path: Path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    stream = next(
        s for s in json.loads(result.stdout)["streams"] if s["codec_type"] == "video"
    )
    return f"{stream['width']}x{stream['height']}"


def crop(config: PipelineConfig) -> dict:
    import cv2

    data_dir = config.paths.resolve("data_dir")
    report_path = data_dir / "inspection_report.json"
    report = load_inspection_report(report_path)

    # Repartir toujours de la source : relancer l'étape ne doit pas rogner une
    # vidéo déjà rognée.
    source = Path(report.get("original_video_path") or report["video_path"])
    samples_dir = config.paths.resolve("frames_dir") / SAMPLES_SUBDIR
    frame_paths = extract_sample_frames(source, samples_dir, config.crop.sample_fps)
    step = max(1, len(frame_paths) // config.crop.sample_frames)
    selected = frame_paths[::step][: config.crop.sample_frames]
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in selected]
    logger.info("%d images analysées pour situer le bandeau", len(frames))

    crop_top, reason = resolve_crop_top(config, frames)
    logger.info("%s", reason)

    preview_path = config.paths.resolve("logs_dir") / "crop_preview.jpg"
    write_preview(selected[len(selected) // 2], crop_top, preview_path)
    logger.info("Image de contrôle : %s", preview_path)

    if crop_top <= 0:
        report["crop_top"] = 0
        report["original_video_path"] = str(source)
        report["video_path"] = str(source)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    work_path = config.paths.resolve("work_dir") / WORK_VIDEO_NAME
    logger.info("Encodage de la vidéo recadrée vers %s", work_path)
    encode_cropped(source, crop_top, work_path, config)

    report["original_video_path"] = str(source)
    # `setdefault` et non affectation : relancer l'étape ne doit pas prendre la
    # résolution déjà rognée pour l'originale.
    report.setdefault("original_resolution", report["resolution"])
    report["video_path"] = str(work_path)
    # Relue et non calculée : FFmpeg arrondit la hauteur au pair le plus proche
    # (contrainte du H.264), et un rapport qui annonce une autre résolution que
    # le fichier fausserait la conversion des coordonnées normalisées.
    report["resolution"] = probe_resolution(work_path)
    report["crop_top"] = crop_top
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Rapport d'inspection mis à jour : %s -> %s (bandeau de %dpx retiré)",
        report["original_resolution"], report["resolution"], crop_top,
    )
    return report


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crop(load_config(config_path))


if __name__ == "__main__":
    app()
