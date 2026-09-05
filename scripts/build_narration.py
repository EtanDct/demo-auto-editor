"""Étape D : voix off IA locale (plan-technique.md, section 3).

Synthétise chaque `text_en` du conducteur de montage avec Piper (local, pas
d'appel réseau) et écrit un fichier WAV par segment, plus
data/narration_manifest.json avec la durée réelle de chaque segment.

Les pauses (`narration.pause_before_ms` / `pause_after_ms`) ne sont pas
incrustées dans l'audio ici : elles sont appliquées au niveau de la
timeline par l'étape de recalage (étape E), qui a besoin des durées réelles
pour décider des ajustements.
"""

from __future__ import annotations

import json
import logging
import wave
from pathlib import Path

import typer
import yaml

from download_models import parse_piper_voice
from pipeline_config import PipelineConfig, load_config
from schemas import EditDecision, NarrationManifestEntry

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def load_edl(path: Path) -> list[EditDecision]:
    if not path.exists():
        raise FileNotFoundError(
            f"Conducteur de montage introuvable : {path}. Lance d'abord "
            "'python run.py --step translate'."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [EditDecision.model_validate(item) for item in raw]


def load_voice(config: PipelineConfig):
    from piper import PiperVoice  # import tardif : coûteux, inutile pour --help

    lang_family, lang_code, speaker, quality = parse_piper_voice(config.tts.voice)
    model_path = (
        config.paths.resolve("models_dir")
        / "piper"
        / lang_family
        / lang_code
        / speaker
        / quality
        / f"{config.tts.voice}.onnx"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Voix Piper introuvable : {model_path}. Lance d'abord "
            "'python scripts/download_models.py --only tts'."
        )
    logger.info("Chargement de la voix Piper %s", config.tts.voice)
    return PiperVoice.load(str(model_path))


def synthesize_segment(voice, text: str, out_path: Path) -> float:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize(text, wav_file)
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
    return round(n_frames / frame_rate, 3)


def build_narration(decisions: list[EditDecision], config: PipelineConfig) -> list[NarrationManifestEntry]:
    voice = load_voice(config)
    narration_dir = config.paths.resolve("audio_dir") / "narration"
    entries = []
    for decision in decisions:
        out_path = narration_dir / f"{decision.id}.wav"
        logger.info("Synthèse de %s -> %s", decision.id, out_path)
        duration = synthesize_segment(voice, decision.text_en, out_path)
        entries.append(
            NarrationManifestEntry(
                segment_id=decision.id,
                audio_file=str(out_path.relative_to(config.paths.resolve("audio_dir").parent)),
                duration=duration,
                provider="piper",
                voice=decision.narration.voice,
            )
        )
    return entries


def write_manifest(entries: list[NarrationManifestEntry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([e.model_dump() for e in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Manifeste de narration écrit dans %s (%d segments)", out_path, len(entries))


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    decisions = load_edl(config.paths.resolve("data_dir") / "edit_decision_list.yaml")
    entries = build_narration(decisions, config)
    write_manifest(entries, config.paths.resolve("data_dir") / "narration_manifest.json")


if __name__ == "__main__":
    app()
