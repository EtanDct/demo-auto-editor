"""Étape B : transcription française horodatée (plan-technique.md, section 3).

Utilise faster-whisper (local, pas d'appel réseau) pour transcrire
audio/source_audio.wav en segments horodatés, écrits dans
data/transcript_fr.json au format défini dans scripts/schemas.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from hardware import detect_hardware
from pipeline_config import PipelineConfig, load_config
from schemas import TranscriptSegment

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _resolve_device_and_compute_type(config: PipelineConfig) -> tuple[str, str]:
    if config.hardware.device != "auto":
        device = config.hardware.device
    else:
        device = "cuda" if detect_hardware().gpu_available else "cpu"

    if config.whisper.compute_type != "auto":
        compute_type = config.whisper.compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def transcribe(audio_path: Path, config: PipelineConfig) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel  # import tardif : coûteux, inutile pour --help

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio introuvable : {audio_path}. Lance d'abord l'étape 'inspect' "
            "(python run.py --step inspect)."
        )

    model_dir = config.paths.resolve("models_dir") / "whisper" / config.whisper.model_size
    model_source = str(model_dir) if model_dir.exists() else config.whisper.model_repo
    device, compute_type = _resolve_device_and_compute_type(config)

    logger.info(
        "Chargement Whisper (%s, device=%s, compute_type=%s)", model_source, device, compute_type
    )
    model = WhisperModel(model_source, device=device, compute_type=compute_type)

    logger.info("Transcription de %s (langue=%s)", audio_path, config.whisper.language)
    raw_segments, info = model.transcribe(str(audio_path), language=config.whisper.language)
    logger.info("Langue détectée=%s, probabilité=%.2f", info.language, info.language_probability)

    segments = [
        TranscriptSegment(
            id=f"seg-{i + 1:03d}",
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text_fr=seg.text.strip(),
        )
        for i, seg in enumerate(raw_segments)
    ]
    return segments


def write_transcript(segments: list[TranscriptSegment], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([s.model_dump() for s in segments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Transcription écrite dans %s (%d segments)", out_path, len(segments))


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    audio_path = config.paths.resolve("audio_dir") / "source_audio.wav"
    segments = transcribe(audio_path, config)
    write_transcript(segments, config.paths.resolve("data_dir") / "transcript_fr.json")


if __name__ == "__main__":
    app()
