"""Point d'entrée unique du pipeline (plan-technique.md, section 10).

    python run.py --input input/demo.mp4   # pipeline complet, de la vidéo au rendu
    python run.py --from translate         # reprendre à une étape
    python run.py --step render            # une seule étape (débogage)

Le pipeline complet enchaîne toutes les étapes, dans l'ordre où chacune trouve
ce qu'elle attend :

    inspect    métadonnées et audio de la vidéo source
    crop       retrait du bandeau de navigateur ; tout l'aval travaille ensuite
               sur la vidéo recadrée
    transcribe transcription française, horodatée au mot
    translate  adaptation anglaise, titre, chapitres
    screen     index du texte affiché à l'écran (OCR, l'étape la plus longue)
    cursor     trajectoire du pointeur
    match      rapprochement entre ce que dit le narrateur et ce qui est à l'écran,
               reporté dans le conducteur de montage
    narrate    voix off
    retime     recalage de la vidéo sur la voix
    subtitles  sous-titres
    render     montage final
    validate   contrôles automatiques du rendu

Chaque étape lit et écrit les fichiers de `data/` : après une correction à la
main du conducteur de montage, `--from narrate` suffit. Une étape qui échoue
arrête le passage en indiquant la commande de reprise.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import typer

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

STEP_ORDER = [
    "inspect", "crop", "transcribe", "translate", "screen", "cursor", "match",
    "narrate", "retime", "subtitles", "render", "validate",
]


def _run_step(step: str, config_path: Path | None, input_override: Path | None) -> None:
    # Chaque `main` est appelé avec TOUS ses paramètres : hors de typer, une
    # option omise reçoit son objet `typer.Option`, qui vaut vrai. Omettre
    # `chapters_only` réduisait ainsi la traduction aux seuls chapitres.
    if step == "inspect":
        import inspect_source

        config = inspect_source.load_config(config_path)
        video_path = input_override or config.paths.resolve("input_video")
        inspect_source.inspect(video_path, config)
    elif step == "crop":
        import crop_chrome

        crop_chrome.main(config_path=config_path)
    elif step == "transcribe":
        import transcribe

        transcribe.main(config_path=config_path)
    elif step == "translate":
        import translate

        translate.main(config_path=config_path, chapters_only=False)
    elif step == "screen":
        import detect_screen_text

        detect_screen_text.main(config_path=config_path, max_seconds=None, regroup_only=False)
    elif step == "cursor":
        import detect_cursor

        detect_cursor.main(config_path=config_path)
    elif step == "match":
        import match_overlays

        # Dans le pipeline, les correspondances retenues sont reportées dans le
        # conducteur : c'est ce qui pose les cadres au rendu. La planche de
        # contact permet de les relire. Pour un rapport seul, sans rien écrire :
        # python scripts/match_overlays.py
        match_overlays.main(config_path=config_path, apply=True, contact_sheet=True)
    elif step == "narrate":
        import build_narration

        build_narration.main(config_path=config_path)
    elif step == "retime":
        import build_timeline

        build_timeline.main(config_path=config_path)
    elif step == "subtitles":
        import subtitles

        subtitles.main(config_path=config_path)
    elif step == "render":
        import render_video

        render_video.main(config_path=config_path, dry_run=False)
    elif step == "validate":
        import validate_output

        validate_output.main(config_path=config_path)
    else:
        raise typer.BadParameter(f"Étape inconnue : '{step}'. Attendu l'une de : {STEP_ORDER}")


def plan_steps(step: str | None, start_from: str | None) -> list[str]:
    """Étapes à exécuter, dans l'ordre du pipeline."""
    if step and start_from:
        raise typer.BadParameter("--step et --from s'excluent : l'un lance une étape, l'autre reprend.")
    for name in (step, start_from):
        if name and name not in STEP_ORDER:
            raise typer.BadParameter(f"Étape inconnue : '{name}'. Attendu l'une de : {STEP_ORDER}")
    if step:
        return [step]
    if start_from:
        return STEP_ORDER[STEP_ORDER.index(start_from):]
    return list(STEP_ORDER)


def run_steps(
    steps: list[str], config_path: Path | None, input_override: Path | None, runner=_run_step
) -> list[tuple[str, float]]:
    """Exécute les étapes ; en cas d'échec, dit comment reprendre."""
    durations: list[tuple[str, float]] = []
    for index, step in enumerate(steps, 1):
        logger.info("=== Étape %d/%d : %s ===", index, len(steps), step)
        started = time.monotonic()
        try:
            runner(step, config_path, input_override)
        except Exception:
            logger.error(
                "Échec à l'étape '%s'. Une fois le problème corrigé, reprendre avec : "
                "python run.py --from %s",
                step, step,
            )
            raise
        durations.append((step, time.monotonic() - started))
    return durations


def summary(durations: list[tuple[str, float]]) -> str:
    width = max((len(step) for step, _ in durations), default=0)
    lines = [f"  {step:<{width}}  {seconds:7.1f} s" for step, seconds in durations]
    total = sum(seconds for _, seconds in durations)
    return "\n".join(["Durée par étape :", *lines, f"  {'total':<{width}}  {total:7.1f} s"])


@app.command()
def main(
    step: str = typer.Option(
        None, "--step", help=f"N'exécuter qu'une étape ({', '.join(STEP_ORDER)})."
    ),
    start_from: str = typer.Option(
        None, "--from", help="Reprendre le pipeline à cette étape, jusqu'à la fin."
    ),
    input: Path = typer.Option(
        None, "--input", help="Vidéo source, override de paths.input_video pour cette exécution."
    ),
    config: Path = typer.Option(None, "--config", help="Chemin vers config.yaml."),
) -> None:
    # Console Windows en cp1252 : un caractère hors de cette page dans un
    # journal (flèche, guillemet typographique) faisait planter le passage
    # entier au milieu d'une étape.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    durations = run_steps(plan_steps(step, start_from), config, input)
    if len(durations) > 1:
        logger.info("%s", summary(durations))


if __name__ == "__main__":
    app()
