"""Étape G : sous-titres (plan-technique.md, section 3).

Génère data/subtitles_en.srt à partir du conducteur de montage (text_en) et
des timecodes finaux de data/timeline.json (après recalage, étape E) :

- deux lignes maximum, longueur limitée par ligne ;
- coupure sur les mots, jamais au milieu d'un mot ; entre deux sous-titres,
  de préférence sur une fin de phrase ou une virgule ;
- un texte trop long pour un seul sous-titre est réparti sur plusieurs, le
  temps du segment étant partagé au prorata du nombre de mots. Il n'est jamais
  tronqué : des mots qui disparaissent de la vidéo livrée ne se voient pas au
  contrôle automatique.

Le SRT est le livrable, lisible par n'importe quel lecteur. Pour l'incrustation,
le rendu en dérive un fichier ASS (`build_ass`) : c'est le seul format qui place
un texte au pixel près, ici au centre de la bande basse de la mise en page. Un
SRT incrusté tel quel se pose sur l'image, à une hauteur que libass calcule
dans sa propre résolution de référence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import typer

from build_narration import load_edl
from frame_layout import LINE_HEIGHT, Layout
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

    Où couper entre deux sous-titres : remplis au maximum, ils se coupaient là où
    la place manquait, quitte à laisser « …one for revenues, one for the » en
    suspens. Un sous-titre plein est donc raccourci jusqu'à la dernière fin de
    phrase qu'il contient, à défaut jusqu'à la dernière virgule — pourvu qu'il
    reste au moins à moitié plein, sans quoi on retrouverait l'excès inverse,
    une suite de sous-titres de trois mots.
    """
    words = text.split()
    cues: list[list[str]] = []
    start = 0
    while start < len(words):
        end = start + 1
        while end < len(words) and _fits(words[start : end + 1], max_chars_per_line, max_lines):
            end += 1
        if end < len(words):
            end = _natural_break(words, start, end)
        cues.append(wrap_lines(" ".join(words[start:end]), max_chars_per_line))
        start = end
    return cues


STRONG_PUNCTUATION = (".", "!", "?", "…")
SOFT_PUNCTUATION = (",", ";", ":")
MIN_CUE_FILL = 0.5


def _fits(words: list[str], max_chars_per_line: int, max_lines: int) -> bool:
    return len(wrap_lines(" ".join(words), max_chars_per_line)) <= max_lines


def _natural_break(words: list[str], start: int, end: int) -> int:
    """Fin de sous-titre sur une ponctuation, si elle ne le vide pas trop.

    `end` est la fin maximale (exclue) ; on cherche en reculant une fin de
    phrase, puis une virgule, qui garde au moins `MIN_CUE_FILL` des mots.
    """
    required = max(1, int((end - start) * MIN_CUE_FILL))
    earliest_last = start + required - 1
    for punctuation in (STRONG_PUNCTUATION, SOFT_PUNCTUATION):
        for last in range(end - 1, earliest_last - 1, -1):
            if words[last].endswith(punctuation):
                return last + 1
    return end


def _format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    lines: list[str]


def build_cues(
    decisions: list[EditDecision],
    timeline_by_id: dict[str, TimelineEntry],
    config: PipelineConfig,
) -> list[Cue]:
    """Répliques et horaires, indépendamment du format de fichier."""
    result = []
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
            result.append(Cue(cursor, end, cue))
            cursor = end

    return result


def build_srt(
    decisions: list[EditDecision],
    timeline_by_id: dict[str, TimelineEntry],
    config: PipelineConfig,
) -> str:
    blocks = [
        f"{index}\n"
        f"{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n"
        f"{chr(10).join(cue.lines)}\n"
        for index, cue in enumerate(build_cues(decisions, timeline_by_id, config), 1)
    ]
    return "\n".join(blocks)


# --- ASS, pour l'incrustation ------------------------------------------------

NAMED_COLORS = {
    "white": "FFFFFF", "black": "000000", "yellow": "FFFF00", "cyan": "00FFFF",
    "red": "FF0000", "green": "00FF00", "blue": "0000FF", "gray": "808080", "grey": "808080",
}


def ass_color(value: str, alpha: int = 0) -> str:
    """Couleur FFmpeg (`white`, `0xRRGGBB`, `#RRGGBB`) en notation ASS `&HAABBGGRR`."""
    text = str(value).strip().lower()
    rgb = (NAMED_COLORS.get(text) or text.removeprefix("0x").removeprefix("#")).lower()
    if len(rgb) != 6 or any(c not in "0123456789abcdef" for c in rgb):
        raise ValueError(f"Couleur de sous-titres illisible : {value!r}")
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def _ass_timestamp(seconds: float) -> str:
    total_cs = round(max(0.0, seconds) * 100)
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_text(line: str) -> str:
    """Neutralise ce qu'ASS interpréterait : accolades (balises) et antislash."""
    return line.replace("\\", "/").replace("{", "(").replace("}", ")")


def subtitle_font_size(layout: Layout, config: PipelineConfig) -> int:
    """Taille réglée, réduite si la bande ne peut pas porter toutes les lignes."""
    size = config.layout.subtitle_size
    if layout.subtitles_in_band:
        fitting = int(layout.bottom_height * 0.85 / (config.subtitles.max_lines * LINE_HEIGHT))
        size = min(size, fitting)
    return max(size, 12)


def build_ass(cues: list[Cue], layout: Layout, config: PipelineConfig) -> str:
    """Sous-titres ASS à la résolution du canevas.

    Dans une bande, chaque réplique est centrée sur la bande (`\\an5\\pos`) : une
    ligne ou deux restent au milieu, sans contour puisque le fond est uni. Sans
    bande, elles se posent en bas de l'image avec un contour, comme avant.
    """
    settings = config.layout
    size = subtitle_font_size(layout, config)
    in_band = layout.subtitles_in_band
    outline, alignment = (0, 5) if in_band else (2, 2)
    margin_v = 0 if in_band else round(layout.height * 0.05)
    position = ""
    if in_band:
        position = f"{{\\an5\\pos({layout.width // 2},{layout.bottom_top + layout.bottom_height // 2})}}"

    header = "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {layout.width}",
        f"PlayResY: {layout.height}",
        # Pas de retour à la ligne automatique : les lignes sont déjà coupées,
        # et libass en recouperait d'autres à sa façon.
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{settings.font_name},{size},{ass_color(settings.subtitle_color)},"
        f"&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline},0,{alignment},"
        f"{settings.band_padding},{settings.band_padding},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])
    # Hors de l'f-string : Python 3.11 n'y accepte pas d'antislash.
    line_break = "\\N"
    events = [
        f"Dialogue: 0,{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},Default,,0,0,0,,"
        f"{position}{line_break.join(_ass_text(line) for line in cue.lines)}"
        for cue in cues
    ]
    return header + "\n" + "\n".join(events) + "\n"


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
