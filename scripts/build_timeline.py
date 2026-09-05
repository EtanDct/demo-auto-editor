"""Étape E : recalage temporel (plan-technique.md, section 3).

Calcule les nouveaux intervalles à partir de la durée réelle de chaque
segment de narration, en parcourant les segments dans l'ordre chronologique
source et en suivant l'ordre de priorité du plan technique :

1. pauses (déjà dans le conducteur, non modifiées ici) ;
2. vitesse audio, très légèrement, dans les bornes de config.yaml ;
3. si la vitesse seule ne suffit pas, accélération au maximum autorisé
   PUIS extension du plan vidéo de ce qui manque encore (le segment est
   marqué `extended`, à traiter par le rendu en figeant la dernière
   image) : tous les segments suivants sont alors décalés d'autant, pour
   que la timeline finale reste strictement séquentielle et sans
   chevauchement. L'extension est calculée sur la durée *accélérée* de la
   narration, sinon le plan reste gelé pour rien.

Détecte aussi les chevauchements entre segments source (avant tout
décalage), condition nécessaire avant de construire la timeline finale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from build_narration import load_edl
from pipeline_config import PipelineConfig, load_config
from schemas import EditDecision, NarrationManifestEntry, TimelineEntry, TimelineReport

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def load_narration_manifest(path: Path) -> dict[str, NarrationManifestEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifeste de narration introuvable : {path}. Lance d'abord "
            "'python run.py --step narrate'."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [NarrationManifestEntry.model_validate(item) for item in raw]
    return {e.segment_id: e for e in entries}


def check_source_overlaps(decisions: list[EditDecision]) -> list[str]:
    warnings = []
    ordered = sorted(decisions, key=lambda d: d.source_start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.source_start < previous.source_end:
            warnings.append(
                f"Chevauchement source : {previous.id} (fin {previous.source_end}s) "
                f"et {current.id} (début {current.source_start}s)"
            )
    return warnings


def build_timeline(
    decisions: list[EditDecision],
    narration_by_id: dict[str, NarrationManifestEntry],
    config: PipelineConfig,
) -> TimelineReport:
    warnings = check_source_overlaps(decisions)
    entries: list[TimelineEntry] = []
    cumulative_shift = 0.0

    for decision in sorted(decisions, key=lambda d: d.source_start):
        narration = narration_by_id.get(decision.id)
        if narration is None:
            warnings.append(f"Aucune narration trouvée pour {decision.id}, segment ignoré.")
            continue

        source_duration = decision.source_end - decision.source_start
        adjusted_start = decision.source_start + cumulative_shift
        adjusted_end = decision.source_end + cumulative_shift
        total_audio_duration = (
            decision.narration.pause_before_ms / 1000
            + narration.duration
            + decision.narration.pause_after_ms / 1000
        )

        if total_audio_duration <= source_duration:
            entries.append(
                TimelineEntry(
                    id=decision.id,
                    source_start=decision.source_start,
                    source_end=decision.source_end,
                    new_start=adjusted_start,
                    new_end=adjusted_end,
                    narration_duration=narration.duration,
                )
            )
            continue

        required_speed = total_audio_duration / source_duration
        if required_speed <= config.retiming.max_speed_factor:
            entries.append(
                TimelineEntry(
                    id=decision.id,
                    source_start=decision.source_start,
                    source_end=decision.source_end,
                    new_start=adjusted_start,
                    new_end=adjusted_end,
                    narration_duration=narration.duration,
                    audio_speed_factor=round(required_speed, 4),
                )
            )
            continue

        # La vitesse seule ne suffit pas dans les bornes acceptables : on
        # accélère au maximum autorisé, PUIS on étend le plan vidéo de ce qui
        # manque encore. La durée à couvrir est celle réellement jouée une fois
        # accélérée (`played_duration`), pas la durée brute de la narration :
        # sinon le plan reste gelé du facteur d'accélération en trop à chaque
        # extension (~0.4s par segment, 6.3s cumulées sur le premier extrait
        # réel de 188s).
        speed = config.retiming.max_speed_factor
        played_duration = total_audio_duration / speed
        new_end = adjusted_start + played_duration
        extension = new_end - adjusted_end
        cumulative_shift += extension
        # Ratio d'étirement réel du plan vidéo (1.0 = plan source inchangé).
        extension_ratio = played_duration / source_duration
        entries.append(
            TimelineEntry(
                id=decision.id,
                source_start=decision.source_start,
                source_end=decision.source_end,
                new_start=adjusted_start,
                new_end=new_end,
                narration_duration=narration.duration,
                audio_speed_factor=speed,
                extended=True,
                needs_review=extension_ratio > 1.2,
            )
        )
        warnings.append(
            f"{decision.id} : narration ({total_audio_duration:.2f}s, {played_duration:.2f}s "
            f"une fois accélérée à x{speed}) dépasse le plan source ({source_duration:.2f}s) ; "
            f"plan étendu de {extension:.2f}s (segments suivants décalés d'autant), "
            f"needs_review={extension_ratio > 1.2}."
        )

    return TimelineReport(entries=entries, warnings=warnings)


def write_timeline(report: TimelineReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    for warning in report.warnings:
        logger.warning(warning)
    logger.info(
        "Timeline écrite dans %s (%d segments, %d avertissements)",
        out_path, len(report.entries), len(report.warnings),
    )


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    data_dir = config.paths.resolve("data_dir")
    decisions = load_edl(data_dir / "edit_decision_list.yaml")
    narration_by_id = load_narration_manifest(data_dir / "narration_manifest.json")
    report = build_timeline(decisions, narration_by_id, config)
    write_timeline(report, data_dir / "timeline.json")


if __name__ == "__main__":
    app()
