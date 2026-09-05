"""Étapes F + H : effets visuels et rendu final (plan-technique.md, section 3).

Construit un unique filter_complex FFmpeg qui :

1. découpe la vidéo source en morceaux séquentiels (segments narrés,
   éventuellement étendus par un gel de la dernière image, et intervalles
   "silencieux" entre deux segments, inchangés) ;
2. applique l'effet visuel (`visual_action`) de chaque segment via
   scripts/overlays.py ;
3. recolle tous les morceaux (`concat`) puis incruste les sous-titres ;
4. construit la piste audio finale en plaçant chaque narration à son
   `new_start` (avec `atempo` si une accélération légère a été retenue à
   l'étape E, et `adelay` pour le décalage) ;
5. exporte `output/final.mp4` (qualité maître) puis `output/preview.mp4`
   (basse résolution, second passage FFmpeg sur le master).

Validé sur un premier extrait réel (2m47s, 33 segments, 17 gels de plan) :
le rendu s'est terminé sans erreur FFmpeg, mais scripts/validate_output.py
a détecté un écart durée vidéo/audio de ~18s — le gel de plan tronquait un
micro-extrait de fin de segment (quelques ms) qui tombait souvent entre
deux frames et ne produisait aucune image, donc aucun gel. Corrigé en
appliquant `tpad` directement sur le clip complet du segment (jamais vide)
plutôt que sur un extrait de quelques millisecondes.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer

from build_narration import load_edl
from build_timeline import load_narration_manifest
from overlays import overlay_filter_for
from pipeline_config import PipelineConfig, load_config
from schemas import EditDecision, NarrationManifestEntry, TimelineEntry

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

@dataclass
class Piece:
    kind: str  # "gap" | "segment"
    start: float
    end: float
    extension: float = 0.0
    decision: EditDecision | None = None


def load_timeline_entries(path: Path) -> list[TimelineEntry]:
    if not path.exists():
        raise FileNotFoundError(
            f"Timeline introuvable : {path}. Lance d'abord 'python run.py --step retime'."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TimelineEntry.model_validate(item) for item in raw["entries"]]


def load_inspection_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Rapport d'inspection introuvable : {path}. Lance d'abord "
            "'python run.py --step inspect'."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_pieces(
    decisions: list[EditDecision], timeline_entries: list[TimelineEntry], source_duration: float
) -> list[Piece]:
    entries_by_id = {e.id: e for e in timeline_entries}
    ordered = [d for d in sorted(decisions, key=lambda d: d.source_start) if d.id in entries_by_id]

    pieces: list[Piece] = []
    cursor = 0.0
    for decision in ordered:
        entry = entries_by_id[decision.id]
        if decision.source_start > cursor:
            pieces.append(Piece(kind="gap", start=cursor, end=decision.source_start))
        extension = max(0.0, (entry.new_end - entry.new_start) - (decision.source_end - decision.source_start))
        pieces.append(
            Piece(
                kind="segment",
                start=decision.source_start,
                end=decision.source_end,
                extension=extension,
                decision=decision,
            )
        )
        cursor = decision.source_end

    if cursor < source_duration:
        pieces.append(Piece(kind="gap", start=cursor, end=source_duration))

    return pieces


def _escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:")


def build_video_filter(
    pieces: list[Piece], config: PipelineConfig, width: int, height: int, srt_path: Path, fps: float
) -> tuple[str, str]:
    chains = []
    labels = []

    for i, piece in enumerate(pieces):
        label = f"v{i}"
        # fps= force un débit constant tôt dans la chaîne : la source est en
        # débit variable (capture d'écran typique), et sans ça `tpad` arrondit
        # chaque clonage à la frame réelle la plus proche de son morceau, avec
        # une dérive cumulée observée en test réel (~1.5s sur 17 gels de plan).
        base = f"[0:v]trim=start={piece.start:.3f}:end={piece.end:.3f},setpts=PTS-STARTPTS,fps={fps}"

        overlay = None
        if piece.kind == "segment" and piece.decision is not None:
            # `setpts=PTS-STARTPTS` ci-dessus ramène le temps du morceau à zéro :
            # les décalages de `visual_action` sont donc bien relatifs au début
            # du segment. La durée transmise exclut l'extension par gel de plan,
            # appliquée plus bas et qui prolongerait l'effet sur l'image figée.
            overlay = overlay_filter_for(
                piece.decision.visual_action, config, width, height, piece.end - piece.start
            )
        if overlay:
            base = f"{base},{overlay}"

        if piece.kind == "segment" and piece.extension > 0:
            # tpad clone la dernière frame DÉCODÉE de son entrée : appliqué
            # directement sur le clip complet du segment (jamais vide, à
            # l'inverse d'un micro-extrait de quelques ms qui peut tomber
            # entre deux frames et ne produire aucune image — cas vécu en
            # test réel, voir historique de ce fichier).
            chains.append(f"{base},tpad=stop_mode=clone:stop_duration={piece.extension:.3f}[{label}]")
        else:
            chains.append(f"{base}[{label}]")

        labels.append(label)

    concat_inputs = "".join(f"[{lbl}]" for lbl in labels)
    chains.append(f"{concat_inputs}concat=n={len(labels)}:v=1:a=0[vconcat]")
    # Marge de sécurité : la quantification aux limites d'image de chaque
    # morceau (trim/fps/tpad) fait dériver légèrement la durée totale vers le
    # bas au fil de la concaténation (~0.6s observé sur 33 segments en test
    # réel). On sur-étend puis on coupe à la durée exacte en sortie (-t dans
    # render()) plutôt que de chercher un arrondi parfait par morceau.
    chains.append("[vconcat]tpad=stop_mode=clone:stop_duration=2.000[vpadded]")
    chains.append(f"[vpadded]subtitles='{_escape_filter_path(srt_path)}'[vout]")

    return ";".join(chains), "[vout]"


def total_output_duration(pieces: list[Piece]) -> float:
    """Durée totale de la vidéo finale, extensions par gel de plan incluses."""
    return sum((p.end - p.start) + p.extension for p in pieces)


def build_audio_filter(
    decisions: list[EditDecision],
    timeline_by_id: dict[str, TimelineEntry],
    narration_by_id: dict[str, NarrationManifestEntry],
    audio_input_start_index: int,
    total_duration: float,
) -> tuple[str, list[Path], str]:
    ordered = [d for d in sorted(decisions, key=lambda d: d.source_start) if d.id in timeline_by_id]
    chains = []
    labels = []
    audio_files: list[Path] = []

    for i, decision in enumerate(ordered):
        entry = timeline_by_id[decision.id]
        narration = narration_by_id[decision.id]
        input_index = audio_input_start_index + i
        audio_files.append(Path(narration.audio_file))

        delay_ms = round((entry.new_start + decision.narration.pause_before_ms / 1000) * 1000)
        label = f"a{i}"
        tempo_clause = f"atempo={entry.audio_speed_factor:.4f}," if entry.audio_speed_factor != 1.0 else ""
        chains.append(f"[{input_index}:a]{tempo_clause}adelay=delays={delay_ms}:all=1[{label}]")
        labels.append(label)

    if not labels:
        raise ValueError("Aucun segment audio à mixer : vérifie edit_decision_list.yaml / timeline.json.")

    mix_inputs = "".join(f"[{lbl}]" for lbl in labels)
    chains.append(f"{mix_inputs}amix=inputs={len(labels)}:duration=longest:dropout_transition=0[amix_pre]")
    chains.append(f"[amix_pre]apad=whole_dur={total_duration:.3f}[aout]")

    return ";".join(chains), audio_files, "[aout]"


def render(config: PipelineConfig, dry_run: bool = False) -> Path:
    data_dir = config.paths.resolve("data_dir")
    audio_dir = config.paths.resolve("audio_dir")
    output_dir = config.paths.resolve("output_dir")
    logs_dir = config.paths.resolve("logs_dir")

    decisions = load_edl(data_dir / "edit_decision_list.yaml")
    timeline_entries = load_timeline_entries(data_dir / "timeline.json")
    timeline_by_id = {e.id: e for e in timeline_entries}
    narration_by_id = load_narration_manifest(data_dir / "narration_manifest.json")
    inspection = load_inspection_report(data_dir / "inspection_report.json")

    width, height = (int(v) for v in inspection["resolution"].split("x"))
    source_duration = inspection["duration_seconds"]
    source_video = Path(inspection["video_path"])
    srt_path = data_dir / "subtitles_en.srt"
    if not srt_path.exists():
        raise FileNotFoundError(f"Sous-titres introuvables : {srt_path}. Lance d'abord --step subtitles.")

    pieces = build_pieces(decisions, timeline_entries, source_duration)
    total_duration = total_output_duration(pieces)
    video_filter, video_label = build_video_filter(pieces, config, width, height, srt_path, inspection["fps"])
    audio_filter, audio_files, audio_label = build_audio_filter(
        decisions,
        timeline_by_id,
        narration_by_id,
        audio_input_start_index=1,
        total_duration=total_duration,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final.mp4"
    preview_path = output_dir / "preview.mp4"

    cmd = ["ffmpeg", "-y", "-i", str(source_video)]
    for f in audio_files:
        cmd += ["-i", str(config.paths.resolve("audio_dir").parent / f)]
    cmd += [
        "-filter_complex", f"{video_filter};{audio_filter}",
        "-map", video_label,
        "-map", audio_label,
        "-c:v", config.export.video_codec,
        "-crf", str(config.export.crf),
        "-c:a", config.export.audio_codec,
        "-ar", str(config.export.audio_sample_rate),
        "-t", f"{total_duration:.3f}",
        str(final_path),
    ]

    logs_dir.mkdir(parents=True, exist_ok=True)
    command_log = logs_dir / "render_command.txt"
    command_log.write_text(" ".join(f'"{c}"' if " " in c else c for c in cmd), encoding="utf-8")
    logger.info("Commande FFmpeg écrite dans %s", command_log)

    if dry_run:
        logger.info("--dry-run : rendu non exécuté.")
        return final_path

    logger.info("Rendu du master vers %s", final_path)
    subprocess.run(cmd, check=True)

    preview_cmd = [
        "ffmpeg", "-y", "-i", str(final_path),
        "-vf", f"scale=-2:{config.export.preview_max_height}",
        "-c:a", "copy",
        str(preview_path),
    ]
    logger.info("Rendu de la prévisualisation vers %s", preview_path)
    subprocess.run(preview_cmd, check=True)

    return final_path


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Chemin vers config.yaml."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Écrire la commande FFmpeg sans l'exécuter."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    render(config, dry_run=dry_run)


if __name__ == "__main__":
    app()
