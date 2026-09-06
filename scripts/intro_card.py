"""Carton d'introduction placé en tête de la vidéo livrée.

Le titre est produit par le LLM local à partir de la transcription (étape C,
écrit dans `data/intro.json`) et peut être remplacé dans `config.yaml`.

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
JOINED_NAME = "with_intro.mp4"


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


def _drawtext(text: str, config: PipelineConfig, size: int, y_expr: str, alpha: str) -> str:
    return (
        f"drawtext={text_source(text, config)}{_fontfile_arg(config, 'intro')}:"
        f"fontsize={size}:fontcolor={config.intro.text_color}:"
        f"x=(w-text_w)/2:y={y_expr}:alpha={alpha}"
    )


def build_card_filter(text: IntroText, config: PipelineConfig) -> str:
    """Filtre du carton : titre, sous-titre, fondu d'entrée et de sortie."""
    settings = config.intro
    fade = settings.fade_seconds
    duration = settings.duration_seconds
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


def render_card(text: IntroText, config: PipelineConfig, width: int, height: int, fps: float) -> Path:
    """Encode le carton aux paramètres exacts du master, pour un collage sans réencodage."""
    out_path = config.paths.resolve("work_dir") / CARD_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = config.intro
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={settings.background}:s={width}x{height}:r={fps}:d={settings.duration_seconds}",
            "-f", "lavfi",
            "-i", f"anullsrc=r={config.export.audio_sample_rate}:cl=stereo",
            "-vf", build_card_filter(text, config),
            "-c:v", config.export.video_codec,
            "-crf", str(config.export.crf),
            "-pix_fmt", "yuv420p",
            "-c:a", config.export.audio_codec,
            "-ar", str(config.export.audio_sample_rate),
            "-shortest",
            "-t", f"{settings.duration_seconds:.3f}",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    return out_path


def concat(card_path: Path, master_path: Path, config: PipelineConfig) -> Path:
    """Colle le carton devant le master, sans réencodage (paramètres identiques)."""
    work_dir = config.paths.resolve("work_dir")
    listing = work_dir / "concat.txt"
    listing.write_text(
        f"file '{card_path.as_posix()}'\nfile '{master_path.as_posix()}'\n", encoding="utf-8"
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


def prepend_intro(master_path: Path, config: PipelineConfig, width: int, height: int, fps: float):
    """Place le carton en tête du master. Sans titre disponible, ne fait rien."""
    if not config.intro.enabled:
        return master_path

    text = load_intro_text(config)
    if text is None:
        logger.info(
            "Pas de titre d'introduction (ni config.yaml, ni data/intro.json) : "
            "carton ignoré. Relance 'python run.py --step translate' pour en générer un."
        )
        return master_path

    logger.info("Carton d'introduction : %r / %r", text.title, text.subtitle)
    card_path = render_card(text, config, width, height, fps)
    joined = concat(card_path, master_path, config)
    joined.replace(master_path)
    return master_path
