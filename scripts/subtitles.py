"""Étape G : sous-titres (plan-technique.md, section 3).

Génère data/subtitles_en.srt à partir du conducteur de montage (text_en) et
des timecodes finaux de data/timeline.json (après recalage, étape E) :

- deux lignes maximum, longueur limitée par ligne ;
- segmentation greedy alignée sur les mots (pas de coupure en milieu de mot) ;
- un texte trop long pour un seul sous-titre est réparti sur plusieurs, le
  temps du segment étant partagé au prorata du nombre de mots. Il n'est jamais
  tronqué : des mots qui disparaissent de la vidéo livrée ne se voient pas au
  contrôle automatique.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from build_narration import load_edl
from pipeline_config import PipelineConfig, load_config
from schemas import EditDecision, TimelineEntry

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def load_timeline(path: Path) -> dict[str, TimelineEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"Timeline introuvable : {path}. Lance d'abord 'python run.py --step retime'."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [TimelineEntry.model_validate(item) for item in raw["entries"]]
    return {e.id: e for e in entries}


def wrap_lines(text: str, max_chars_per_line: int) -> list[str]:
    """Découpe le texte en lignes, sur les mots, sans jamais rien perdre."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars_per_line or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def split_into_cues(text: str, max_chars_per_line: int, max_lines: int) -> list[list[str]]:
    """Répartit le texte en autant de sous-titres que nécessaire.

    Un segment peut désormais couvrir une phrase entière (les segments Whisper
    sont recollés avant traduction), et son texte dépasse souvent ce que deux
    lignes peuvent porter. L'ancienne version tronquait alors le texte, avec un
    simple avertissement : des mots disparaissaient de la vidéo livrée. On
    produit plutôt plusieurs sous-titres successifs.
    """
    lines = wrap_lines(text, max_chars_per_line)
    return [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)] or []


def wrap_text(text: str, max_chars_per_line: int, max_lines: int) -> list[str]:
    """Premier sous-titre seulement. Conservé pour les appels qui n'en veulent qu'un."""
    cues = split_into_cues(text, max_chars_per_line, max_lines)
    return cues[0] if cues else []


def _format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(
    decisions: list[EditDecision],
    timeline_by_id: dict[str, TimelineEntry],
    config: PipelineConfig,
) -> str:
    blocks = []
    index = 1
    for decision in sorted(decisions, key=lambda d: d.source_start):
        entry = timeline_by_id.get(decision.id)
        if entry is None:
            logger.warning("Pas d'entrée timeline pour %s, sous-titre ignoré.", decision.id)
            continue

        cues = split_into_cues(
            decision.text_en, config.subtitles.max_chars_per_line, config.subtitles.max_lines
        )
        if not cues:
            continue

        # Le temps du segment est réparti au prorata du nombre de mots : un
        # sous-titre dense reste affiché plus longtemps qu'un sous-titre court.
        weights = [sum(len(line.split()) for line in cue) or 1 for cue in cues]
        total_weight = sum(weights)
        span = entry.new_end - entry.new_start
        cursor = entry.new_start
        for cue, weight in zip(cues, weights):
            end = cursor + span * weight / total_weight
            blocks.append(
                f"{index}\n"
                f"{_format_timestamp(cursor)} --> {_format_timestamp(end)}\n"
                f"{chr(10).join(cue)}\n"
            )
            index += 1
            cursor = end

    return "\n".join(blocks)


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    data_dir = config.paths.resolve("data_dir")
    decisions = load_edl(data_dir / "edit_decision_list.yaml")
    timeline_by_id = load_timeline(data_dir / "timeline.json")
    srt = build_srt(decisions, timeline_by_id, config)
    out_path = data_dir / "subtitles_en.srt"
    out_path.write_text(srt, encoding="utf-8")
    logger.info("Sous-titres écrits dans %s", out_path)


if __name__ == "__main__":
    app()
