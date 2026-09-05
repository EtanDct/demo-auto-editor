"""Contrôles automatisés avant livraison (plan-technique.md, section 4).

Vérifie automatiquement :
- présence des pistes audio et vidéo dans output/final.mp4 ;
- durée audio et durée vidéo cohérentes (écart < 0.2s) ;
- absence de chevauchement et validité des timecodes dans la timeline ;
- niveaux audio globaux (silence total ou saturation).

Le contrôle visuel (pertinence des zooms, lisibilité, naturel de la voix,
prononciation SAP) reste humain, comme indiqué dans le plan technique : ce
script ne remplace pas un visionnage final.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import typer

from pipeline_config import PipelineConfig, load_config
from schemas import TimelineEntry

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def ffprobe_streams(video_path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def check_streams_and_duration(video_path: Path, report: ValidationReport) -> None:
    if not video_path.exists():
        report.errors.append(f"Fichier de sortie introuvable : {video_path}")
        return

    metadata = ffprobe_streams(video_path)
    video_streams = [s for s in metadata["streams"] if s["codec_type"] == "video"]
    audio_streams = [s for s in metadata["streams"] if s["codec_type"] == "audio"]

    if not video_streams:
        report.errors.append("Aucune piste vidéo dans le fichier de sortie.")
    if not audio_streams:
        report.errors.append("Aucune piste audio dans le fichier de sortie.")
    if not video_streams or not audio_streams:
        return

    video_duration = float(video_streams[0].get("duration") or metadata["format"]["duration"])
    audio_duration = float(audio_streams[0].get("duration") or metadata["format"]["duration"])
    if abs(video_duration - audio_duration) > 0.2:
        report.errors.append(
            f"Écart durée vidéo/audio : vidéo={video_duration:.2f}s, audio={audio_duration:.2f}s"
        )


def check_timeline(timeline_entries: list[TimelineEntry], report: ValidationReport) -> None:
    ordered = sorted(timeline_entries, key=lambda e: e.new_start)
    for entry in ordered:
        if entry.new_end <= entry.new_start:
            report.errors.append(f"{entry.id} : timecode invalide (new_end <= new_start)")
    for previous, current in zip(ordered, ordered[1:]):
        if current.new_start < previous.new_end - 1e-6:
            report.errors.append(
                f"Chevauchement final : {previous.id} (fin {previous.new_end:.2f}s) "
                f"et {current.id} (début {current.new_start:.2f}s)"
            )
    for entry in ordered:
        if entry.needs_review:
            report.warnings.append(
                f"{entry.id} : marqué needs_review lors du recalage (extension importante), "
                "à vérifier manuellement."
            )


def check_audio_levels(video_path: Path, report: ValidationReport) -> None:
    if not video_path.exists():
        return
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    mean_match = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", result.stderr)
    max_match = re.search(r"max_volume:\s*(-?\d+\.?\d*) dB", result.stderr)
    if not mean_match or not max_match:
        report.warnings.append("Analyse des niveaux audio impossible (sortie ffmpeg inattendue).")
        return

    mean_db, max_db = float(mean_match.group(1)), float(max_match.group(1))
    if mean_db < -50:
        report.warnings.append(f"Niveau audio moyen très bas ({mean_db} dB) : silence anormal possible.")
    if max_db > -0.1:
        report.warnings.append(f"Crête audio proche de 0 dB ({max_db} dB) : saturation possible.")


def validate(config: PipelineConfig) -> ValidationReport:
    report = ValidationReport()
    output_path = config.paths.resolve("output_dir") / "final.mp4"
    data_dir = config.paths.resolve("data_dir")

    check_streams_and_duration(output_path, report)
    check_audio_levels(output_path, report)

    timeline_path = data_dir / "timeline.json"
    if timeline_path.exists():
        raw = json.loads(timeline_path.read_text(encoding="utf-8"))
        entries = [TimelineEntry.model_validate(item) for item in raw["entries"]]
        check_timeline(entries, report)
    else:
        report.warnings.append(f"Timeline introuvable ({timeline_path}), contrôle des timecodes ignoré.")

    return report


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    report = validate(config)

    for warning in report.warnings:
        logger.warning(warning)
    for error in report.errors:
        logger.error(error)

    if report.ok:
        logger.info(
            "Contrôles automatisés OK (%d avertissement(s)). "
            "Le contrôle visuel/audio humain reste indispensable avant livraison.",
            len(report.warnings),
        )
    else:
        logger.error("Contrôles automatisés en échec (%d erreur(s)).", len(report.errors))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
