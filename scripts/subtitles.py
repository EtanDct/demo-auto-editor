"""Étape G : sous-titres (plan-technique.md, section 3).

Génère data/subtitles_en.srt à partir du conducteur de montage (text_en) et
des timecodes finaux de data/timeline.json (après recalage, étape E) :

- deux lignes maximum, longueur limitée par ligne ;
- segmentation greedy alignée sur les mots (pas de coupure en milieu de mot) ;
- si le texte ne tient pas dans les lignes autorisées, il est tronqué et un
  avertissement est levé (texte à raccourcir manuellement dans l'EDL).
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


def wrap_text(text: str, max_chars_per_line: int, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    consumed_words = sum(len(line.split()) for line in lines)
    if consumed_words < len(words):
        logger.warning(
            "Sous-titre tronqué (%d/%d mots) : texte trop long pour %d ligne(s) de %d caractères. "
            "Texte complet : %r",
            consumed_words, len(words), max_lines, max_chars_per_line, text,
        )
    return lines


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

        lines = wrap_text(decision.text_en, config.subtitles.max_chars_per_line, config.subtitles.max_lines)
        if not lines:
            continue

        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(entry.new_start)} --> {_format_timestamp(entry.new_end)}\n"
            f"{chr(10).join(lines)}\n"
        )
        index += 1

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
