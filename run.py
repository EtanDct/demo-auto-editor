"""Point d'entrée unique du pipeline (plan-technique.md, section 10).

    python run.py                          # pipeline complet
    python run.py --input input/source.mp4 # override de la vidéo source
    python run.py --step transcribe        # une seule étape (débogage / reprise)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

STEP_ORDER = [
    "inspect", "transcribe", "translate", "narrate", "retime", "subtitles", "render", "validate",
]

SIMPLE_STEPS = {
    "transcribe": "transcribe",
    "translate": "translate",
    "narrate": "build_narration",
    "retime": "build_timeline",
    "subtitles": "subtitles",
    "validate": "validate_output",
}


def _run_step(step: str, config_path: Path | None, input_override: Path | None) -> None:
    if step == "inspect":
        import inspect_source

        config = inspect_source.load_config(config_path)
        video_path = input_override or config.paths.resolve("input_video")
        inspect_source.inspect(video_path, config)
    elif step == "render":
        import render_video

        render_video.main(config_path=config_path, dry_run=False)
    elif step in SIMPLE_STEPS:
        import importlib

        module = importlib.import_module(SIMPLE_STEPS[step])
        module.main(config_path=config_path)
    else:
        raise typer.BadParameter(f"Étape inconnue : '{step}'. Attendu l'une de : {STEP_ORDER}")


@app.command()
def main(
    step: str = typer.Option(
        None,
        "--step",
        help=f"N'exécuter qu'une étape ({', '.join(STEP_ORDER)}). Omis = pipeline complet.",
    ),
    input: Path = typer.Option(
        None, "--input", help="Vidéo source, override de paths.input_video pour cette exécution."
    ),
    config: Path = typer.Option(None, "--config", help="Chemin vers config.yaml."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    steps_to_run = [step] if step else STEP_ORDER
    for s in steps_to_run:
        logger.info("=== Étape : %s ===", s)
        _run_step(s, config, input)


if __name__ == "__main__":
    app()
