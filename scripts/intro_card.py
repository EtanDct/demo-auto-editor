"""Cartons d'introduction et de fin encadrant la vidéo livrée.

Le titre est produit par le LLM local à partir de la transcription (étape C,
écrit dans `data/intro.json`) et peut être remplacé dans `config.yaml`. Le
carton de fin reprend ce titre par défaut, avec son propre sous-titre.

Le carton est fabriqué comme un fichier à part, aux mêmes paramètres que le
master — résolution, cadence, codecs — puis collé devant lui par le
démultiplexeur `concat`, sans réencodage. Le construire dans le graphe de
filtres du rendu aurait obligé à décaler tous les `adelay` audio et tous les
timecodes de sous-titres du même montant : trois endroits à tenir en accord,
pour un résultat identique.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from overlays import _fontfile_arg, text_source
from pipeline_config import PipelineConfig
from schemas import IntroText

logger = logging.getLogger(__name__)

CARD_NAME = "intro.mp4"
OUTRO_NAME = "outro.mp4"
JOINED_NAME = "with_cards.mp4"


def load_intro_text(config: PipelineConfig) -> IntroText | None:
    """Titre à afficher : celui de `config.yaml` s'il est renseigné, sinon celui du LLM."""
    settings = config.intro
    if settings.title:
        return IntroText(title=settings.title, subtitle=settings.subtitle or "")

    path = config.paths.resolve("data_dir") / "intro.json"
    if not path.exists():
        return None
    text = IntroText.model_validate_json(path.read_text(encoding="utf-8"))
    if settings.subtitle:
        text = IntroText(title=text.title, subtitle=settings.subtitle)
    return text


def load_outro_text(config: PipelineConfig) -> IntroText | None:
    """Texte du carton de fin : son titre s'il est réglé, sinon celui de l'introduction."""
    settings = config.outro
    title = settings.title
    if not title:
        intro = load_intro_text(config)
        title = intro.title if intro else None
    if not title:
        return None
    return IntroText(title=title, subtitle=settings.subtitle or "")


def _drawtext(text: str, config: PipelineConfig, size: int, y_expr: str, alpha: str) -> str:
    return (
        f"drawtext={text_source(text, config)}{_fontfile_arg(config, 'intro')}:"
        f"fontsize={size}:fontcolor={config.intro.text_color}:"
        f"x=(w-text_w)/2:y={y_expr}:alpha={alpha}"
    )


def build_card_filter(
    text: IntroText,
    config: PipelineConfig,
    duration: float | None = None,
    fade: float | None = None,
) -> str:
    """Filtre du carton : titre, sous-titre, fondu d'entrée et de sortie.

    Durée et fondu sont ceux de l'introduction par défaut ; le carton de fin
    passe les siens. Couleurs et tailles restent communes : les deux cartons
    doivent se répondre.
    """
    settings = config.intro
    fade = settings.fade_seconds if fade is None else fade
    duration = settings.duration_seconds if duration is None else duration
    # `drawtext` accepte une expression pour l'alpha : le fondu est réel, pas
    # approché par paliers comme pour les cadres.
    alpha = (
        f"'if(lt(t,{fade:.2f}),t/{fade:.2f},"
        f"if(lt(t,{duration - fade:.2f}),1,({duration:.2f}-t)/{fade:.2f}))'"
    )
    # Le titre s'appuie au-dessus du milieu, le sous-titre en dessous, avec un
    # écart franc : centrés au plus près, les deux se touchaient presque.
    fragments = [
        _drawtext(text.title, config, settings.title_size, "h/2-text_h-h/40", alpha)
    ]
    if text.subtitle:
        fragments.append(
            _drawtext(text.subtitle, config, settings.subtitle_size, "h/2+h/40", alpha)
        )
    return ",".join(fragments)


def render_card(
    text: IntroText,
    config: PipelineConfig,
    width: int,
    height: int,
    fps: float,
    name: str = CARD_NAME,
    duration: float | None = None,
    fade: float | None = None,
) -> Path:
    """Encode un carton aux paramètres exacts du master, pour un collage sans réencodage."""
    out_path = config.paths.resolve("work_dir") / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = config.intro
    duration = settings.duration_seconds if duration is None else duration
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={settings.background}:s={width}x{height}:r={fps}:d={duration}",
            "-f", "lavfi",
            # Stéréo, comme le master (forcé en stéréo au rendu) : le collage
            # sans réencodage reprend l'en-tête du premier fichier pour tous.
            "-i", f"anullsrc=r={config.export.audio_sample_rate}:cl=stereo",
            "-vf", build_card_filter(text, config, duration, fade),
            "-c:v", config.export.video_codec,
            "-crf", str(config.export.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", config.export.audio_codec,
            "-ar", str(config.export.audio_sample_rate),
            "-shortest",
            "-t", f"{duration:.3f}",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    return out_path


def concat(parts: list[Path], config: PipelineConfig) -> Path:
    """Colle les morceaux bout à bout, sans réencodage (paramètres identiques)."""
    work_dir = config.paths.resolve("work_dir")
    listing = work_dir / "concat.txt"
    listing.write_text(
        "".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8"
    )
    joined = work_dir / JOINED_NAME
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(joined),
        ],
        capture_output=True,
        check=True,
    )
    return joined


def add_cards(master_path: Path, config: PipelineConfig, width: int, height: int, fps: float):
    """Encadre le master de ses cartons. Sans titre disponible, n'ajoute rien.

    Un seul collage pour les deux cartons : chaque passage recopie la vidéo
    entière, inutile de le faire deux fois.
    """
    parts = [master_path]

    if config.intro.enabled:
        text = load_intro_text(config)
        if text is None:
            logger.info(
                "Pas de titre d'introduction (ni config.yaml, ni data/intro.json) : "
                "carton ignoré. Relance 'python run.py --step translate' pour en générer un."
            )
        else:
            logger.info("Carton d'introduction : %r / %r", text.title, text.subtitle)
            parts.insert(0, render_card(text, config, width, height, fps))

    if config.outro.enabled:
        text = load_outro_text(config)
        if text is None:
            logger.info("Pas de titre pour le carton de fin : carton ignoré.")
        else:
            logger.info("Carton de fin : %r / %r", text.title, text.subtitle)
            parts.append(
                render_card(
                    text, config, width, height, fps, name=OUTRO_NAME,
                    duration=config.outro.duration_seconds, fade=config.outro.fade_seconds,
                )
            )

    if len(parts) == 1:
        return master_path
    joined = concat(parts, config)
    joined.replace(master_path)
    return master_path
