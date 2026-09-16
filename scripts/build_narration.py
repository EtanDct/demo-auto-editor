"""Étape D : voix off IA locale (plan-technique.md, section 3).

Synthétise chaque `text_en` du conducteur de montage avec Kokoro-82M, via
`kokoro-onnx` sur onnxruntime (local, sans appel réseau ; voir
scripts/download_models.py pour ses fichiers), et écrit un fichier WAV par
segment, plus data/narration_manifest.json avec la durée réelle de chaque
segment.

Le modèle est chargé une fois pour toute la vidéo. Au-delà de 510 phonèmes, le
paquet découpe lui-même le texte en morceaux et les recolle : une phrase longue
n'est pas tronquée. Le WAV est à 24 kHz ; le rendu rééchantillonne tout à la
fréquence d'export.

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

import numpy as np
import typer
import yaml

from download_models import kokoro_paths
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


def write_wav(samples, sample_rate: int, out_path: Path) -> float:
    """WAV mono 16 bits depuis des échantillons flottants ; renvoie la durée.

    16 bits plutôt que flottant : le format le plus largement relu, par FFmpeg
    comme par les outils d'écoute.
    """
    audio = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    if audio.size == 0:
        raise ValueError(f"Synthèse vide pour {out_path.name} : aucun échantillon produit.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio * 32767).astype("<i2").tobytes())
    return round(audio.size / sample_rate, 3)


class KokoroVoice:
    """Kokoro chargé une fois, pour synthétiser tous les segments d'une vidéo."""

    def __init__(self, config: PipelineConfig):
        model_path, voices_path = kokoro_paths(config, config.paths.resolve("models_dir"))
        for path in (model_path, voices_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier Kokoro introuvable : {path}. Lance d'abord "
                    "'python scripts/download_models.py --only tts'."
                )
        from kokoro_onnx import Kokoro

        # La phonémisation (espeak, via phonemizer) signale « words count
        # mismatch » dès qu'elle découpe les mots autrement que le texte — une
        # ligne sur deux sur la démo Sales Report. Sans conséquence : retranscrites
        # par Whisper, les 15 phrases rendaient 99 % des mots (l'ancienne voix
        # Piper : 96 %), y compris celles qui déclenchaient l'avertissement. Seul
        # ce journal-là est rabaissé, les erreurs passent toujours.
        logging.getLogger("phonemizer").setLevel(logging.ERROR)

        self.engine = Kokoro(str(model_path), str(voices_path))
        self.voice = config.tts.voice
        self.speed = config.tts.speed
        self.lang = config.tts.lang
        # Vérifié avant la première phrase : une faute de frappe dans le nom de
        # la voix échouerait sinon au milieu d'une erreur de tableau numpy.
        available = self.engine.get_voices()
        if self.voice not in available:
            close = [v for v in available if v[:2] == self.voice[:2]] or available
            raise ValueError(
                f"Voix Kokoro inconnue : {self.voice!r}. Voix proches : {', '.join(close[:12])}."
            )

    def synthesize(self, text: str, out_path: Path) -> float:
        samples, sample_rate = self.engine.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        return write_wav(samples, sample_rate, out_path)


def build_narration(decisions: list[EditDecision], config: PipelineConfig) -> list[NarrationManifestEntry]:
    voice = KokoroVoice(config)

    narration_dir = config.paths.resolve("audio_dir") / "narration"
    entries = []
    for decision in decisions:
        out_path = narration_dir / f"{decision.id}.wav"
        logger.info("Synthèse de %s -> %s", decision.id, out_path)
        duration = voice.synthesize(decision.text_en, out_path)
        entries.append(
            NarrationManifestEntry(
                segment_id=decision.id,
                audio_file=str(out_path.relative_to(config.paths.resolve("audio_dir").parent)),
                duration=duration,
                provider="kokoro",
                # La voix réellement utilisée, et non celle notée dans le
                # conducteur au moment de la traduction : changer de voix ne
                # demande pas de retraduire.
                voice=config.tts.voice,
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
