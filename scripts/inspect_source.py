"""Étape A : inspection et préparation de la vidéo (plan-technique.md, section 3).

1. Calcule l'empreinte SHA-256 de la vidéo source.
2. Utilise ffprobe pour produire data/source_metadata.json.
3. Extrait l'audio en WAV PCM 48 kHz vers audio/source_audio.wav.
4. Extrait des vignettes régulières vers frames/.
5. Écrit un rapport d'inspection dans data/inspection_report.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer

from pipeline_config import PipelineConfig, load_config

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _parse_frame_rate(value: str) -> float:
    """ffprobe renvoie r_frame_rate sous forme de fraction, ex: "30000/1001"."""
    num, _, den = value.partition("/")
    return float(num) / float(den) if den else float(num)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_ffprobe(video_path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            str(audio_path),
        ],
        capture_output=True,
        check=True,
    )


def extract_frames(video_path: Path, frames_dir: Path, interval_seconds: float) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for existing in frames_dir.glob("frame_*.jpg"):
        existing.unlink()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps=1/{interval_seconds}",
            str(frames_dir / "frame_%05d.jpg"),
        ],
        capture_output=True,
        check=True,
    )
    return len(list(frames_dir.glob("frame_*.jpg")))


def inspect(video_path: Path, config: PipelineConfig) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(
            f"Vidéo source introuvable : {video_path}. Dépose ton fichier dans input/ "
            "et vérifie paths.input_video dans config.yaml."
        )

    data_dir = config.paths.resolve("data_dir")
    audio_dir = config.paths.resolve("audio_dir")
    frames_dir = config.paths.resolve("frames_dir")
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Calcul du SHA-256 de %s", video_path)
    checksum = sha256_file(video_path)

    logger.info("Lecture des métadonnées via ffprobe")
    metadata = run_ffprobe(video_path)
    (data_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audio_path = audio_dir / "source_audio.wav"
    logger.info("Extraction audio vers %s", audio_path)
    extract_audio(video_path, audio_path)

    logger.info("Extraction des vignettes (une toutes les %ss)", config.inspection.thumbnail_interval_seconds)
    frame_count = extract_frames(video_path, frames_dir, config.inspection.thumbnail_interval_seconds)

    video_stream = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in metadata["streams"] if s["codec_type"] == "audio"), None)

    report = {
        "video_path": str(video_path),
        "sha256": checksum,
        "duration_seconds": float(metadata["format"]["duration"]),
        "resolution": f"{video_stream['width']}x{video_stream['height']}" if video_stream else None,
        "fps": _parse_frame_rate(video_stream["r_frame_rate"]) if video_stream else None,
        "video_codec": video_stream["codec_name"] if video_stream else None,
        "audio_codec": audio_stream["codec_name"] if audio_stream else None,
        "audio_channels": audio_stream["channels"] if audio_stream else None,
        "extracted_audio_path": str(audio_path),
        "frames_dir": str(frames_dir),
        "frame_count": frame_count,
        "thumbnail_interval_seconds": config.inspection.thumbnail_interval_seconds,
        "inspected_at": datetime.now(timezone.utc).isoformat(),
    }
    (data_dir / "inspection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Rapport d'inspection écrit dans %s", data_dir / "inspection_report.json")
    return report


@app.command()
def main(
    input_path: Path = typer.Option(None, "--input", help="Vidéo source (défaut : paths.input_video)."),
    config_path: Path = typer.Option(None, help="Chemin vers config.yaml."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    video_path = input_path or (config.paths.resolve("input_video"))
    inspect(video_path, config)


if __name__ == "__main__":
    app()
