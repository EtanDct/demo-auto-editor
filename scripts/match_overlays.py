"""Étape F1 : appariement entre ce que le narrateur nomme et ce qui est à l'écran.

Rapproche `ui_reference` (étape C : le narrateur cite un libellé) de
`screen_elements.json` (étape `screen` : ce libellé est-il affiché, où, quand)
et propose une incrustation quand — et seulement quand — la correspondance ne
laisse pas de place au doute.

Sortie : `data/overlay_candidates.json`, un verdict par segment nommé, accepté
ou refusé **avec son motif**. Sans le motif, un rappel faible serait
indiscernable d'un bug.

Le principe est le refus par défaut. Une incrustation au mauvais endroit
décrédibilise toute la vidéo ; une incrustation manquante ne se voit pas. Quatre
règles écartent donc un candidat, dans cet ordre :

1. **score insuffisant** : le libellé lu à l'écran ne ressemble pas assez à
   celui qu'annonce le narrateur ;
2. **ambiguïté** : deux candidats se valent. C'est le cas du libellé affiché à
   deux endroits — on renonce plutôt que de tirer à pile ou face ;
3. **trop fugace** : l'élément n'est affiché qu'une fraction du segment, donc
   probablement pas ce dont on parle ;
4. **boîte aberrante** : trop petite pour être un contrôle, ou si grande que
   l'OCR a capturé un bloc de texte entier.

Une seule chose peut sauver un candidat écarté pour ambiguïté : la position du
pointeur (étape `cursor`). Si un seul des prétendants est sous la souris, c'est
lui. Le pointeur n'intervient que là — il ne fabrique jamais une correspondance
à partir de rien et ne renverse jamais un appariement déjà net.

Rien n'est écrit dans le conducteur de montage sans `--apply` : le rapport se
relit d'abord, planche de contact à l'appui (`--contact-sheet`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

from build_narration import load_edl
from detect_cursor import load_track, position_at
from pipeline_config import PipelineConfig, load_config
from schemas import (
    BoundingBox,
    CursorTrack,
    EditDecision,
    OverlayCandidate,
    OverlayMatchReport,
    ScreenElement,
    ScreenTextIndex,
    VisualAction,
)
from ui_reference import identifying_tokens, match_key, normalize_text

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


@dataclass
class ScoredElement:
    element: ScreenElement
    score: float
    visible_fraction: float


def load_screen_index(path: Path) -> ScreenTextIndex:
    if not path.exists():
        raise FileNotFoundError(
            f"Index du texte à l'écran introuvable : {path}. Lance d'abord "
            "'python run.py --step screen'."
        )
    return ScreenTextIndex.model_validate_json(path.read_text(encoding="utf-8"))


def score_candidate(label: str, screen_text: str) -> float:
    """Ressemblance entre le libellé annoncé et un libellé lu à l'écran (0 à 1).

    La comparaison ignore casse, accents et espaces : l'OCR restitue le français
    sans accents et colle les mots de façon instable. Un libellé dont tous les
    mots identifiants sont présents part de 0.5, complété selon l'écart de
    longueur — c'est ce qui distingue "Repositories" retrouvé dans
    "Top repositories" (proche) de "Repositories" noyé dans
    "Created 13 commits in 2 repositories" (bien plus lâche).

    Une correspondance seulement partielle est plafonnée sous le seuil
    d'acceptation : elle peut départager, jamais décider seule.
    """
    label_key, screen_key = match_key(label), match_key(screen_text)
    if not label_key or not screen_key:
        return 0.0
    if label_key == screen_key:
        return 1.0

    tokens = identifying_tokens(label)
    if not tokens:
        return 0.0

    covered = sum(1 for token in tokens if token in screen_key)
    coverage = covered / len(tokens)
    if coverage < 1.0:
        return round(coverage * 0.5, 4)

    return round(0.5 + 0.5 * min(1.0, len(label_key) / len(screen_key)), 4)


def gather_candidates(
    decision: EditDecision, elements: list[ScreenElement], config: PipelineConfig
) -> list[ScoredElement]:
    """Éléments visibles pendant le segment, notés contre le libellé annoncé."""
    settings = config.overlay_matching
    label = decision.ui_reference.label
    # Le narrateur nomme souvent l'élément juste avant ou après l'avoir montré :
    # la fenêtre déborde du segment des deux côtés.
    start = decision.source_start - settings.time_margin_seconds
    end = decision.source_end + settings.time_margin_seconds

    scored = []
    for element in elements:
        if element.last_seen <= start or element.first_seen >= end:
            continue
        score = score_candidate(label, element.text)
        if score <= 0:
            continue
        scored.append(
            ScoredElement(
                element=element,
                score=score,
                visible_fraction=element.visible_at(decision.source_start, decision.source_end),
            )
        )

    return sorted(scored, key=lambda s: (-s.score, -s.visible_fraction))


def distance_to_box(box: BoundingBox, point: tuple[float, float]) -> float:
    """Distance du pointeur à la boîte, 0 s'il est dedans.

    Calculée en coordonnées normalisées : sur une image 16/9, un même seuil
    couvre donc environ deux fois plus de pixels horizontalement que
    verticalement. C'est plutôt souhaitable ici, les contrôles d'une interface
    étant serrés verticalement et espacés horizontalement.
    """
    x, y = point
    dx = max(box.x - x, 0.0, x - (box.x + box.width))
    dy = max(box.y - y, 0.0, y - (box.y + box.height))
    return (dx * dx + dy * dy) ** 0.5


def cursor_positions_during(
    track: CursorTrack | None, decision: EditDecision, config: PipelineConfig
) -> list[tuple[float, float]]:
    """Positions du pointeur pendant le segment, marge comprise.

    On retient toutes les positions de la fenêtre plutôt qu'un instant précis :
    le narrateur amène la souris sur l'élément à un moment quelconque de sa
    phrase, souvent avant de le nommer.
    """
    if track is None:
        return []
    margin = config.overlay_matching.time_margin_seconds
    start, end = decision.source_start - margin, decision.source_end + margin
    positions = [(s.x, s.y) for s in track.samples if start <= s.timestamp <= end]
    if positions:
        return positions
    # Pointeur immobile pendant tout le segment : sa dernière position connue
    # reste valable, dans la limite de `hold_seconds`.
    held = position_at(track, decision.source_start, config.cursor.hold_seconds)
    return [held] if held else []


def arbitrate_with_cursor(
    contenders: list[ScoredElement], positions: list[tuple[float, float]], config: PipelineConfig
) -> tuple[ScoredElement | None, float | None]:
    """Départage des libellés équivalents par la position du pointeur.

    Le pointeur ne sert qu'ici : il ne crée jamais une correspondance à partir
    de rien et ne renverse jamais un appariement déjà net. Il ne tranche que si
    un seul prétendant est assez près, sinon l'ambiguïté demeure et on renonce.
    """
    max_distance = config.overlay_matching.cursor_max_distance
    if not positions or max_distance <= 0:
        return None, None

    near = []
    for contender in contenders:
        distance = min(distance_to_box(contender.element.box, p) for p in positions)
        if distance <= max_distance:
            near.append((distance, contender))

    if len(near) != 1:
        return None, None
    distance, winner = near[0]
    return winner, round(distance, 4)


def display_window(
    decision: EditDecision, element: ScreenElement
) -> tuple[float | None, float | None]:
    """Fenêtre d'affichage de l'incrustation, bornée à la présence de l'élément.

    Un cadre dessiné alors que l'élément n'est plus à l'écran se voit
    immédiatement. On borne donc l'effet à ce que l'index constate, plutôt que
    de couvrir tout le segment par défaut. Quand l'élément est présent d'un
    bout à l'autre, on ne renseigne rien : le conducteur de montage reste
    lisible et l'effet couvre naturellement le segment.
    """
    duration = decision.source_end - decision.source_start
    start = max(0.0, element.first_seen - decision.source_start)
    end = min(duration, element.last_seen - decision.source_start)
    if start <= 0 and end >= duration:
        return None, None
    return round(start, 3), round(end, 3)


def judge(
    decision: EditDecision,
    scored: list[ScoredElement],
    config: PipelineConfig,
    cursor_positions: list[tuple[float, float]] | None = None,
) -> OverlayCandidate:
    """Applique les règles de refus et rend un verdict motivé."""
    settings = config.overlay_matching
    label = decision.ui_reference.label
    base = {"segment_id": decision.id, "label": label}

    viable = [s for s in scored if s.score >= settings.min_score]
    if not viable:
        best = scored[0] if scored else None
        return OverlayCandidate(
            **base,
            accepted=False,
            reason=(
                f"aucun libellé à l'écran ne correspond (meilleur score {best.score:.2f} "
                f"pour {best.element.text!r})"
                if best
                else "aucun libellé à l'écran ne correspond"
            ),
            score=best.score if best else 0.0,
            element_text=best.element.text if best else None,
        )

    best = viable[0]
    rivals = [s for s in viable[1:] if best.score - s.score <= settings.ambiguity_margin]
    evidence = ["ocr"]
    cursor_distance = None
    beaten: list[str] = []
    if rivals:
        winner, cursor_distance = arbitrate_with_cursor(
            [best, *rivals], cursor_positions or [], config
        )
        if winner is None:
            return OverlayCandidate(
                **base,
                accepted=False,
                reason=f"ambigu : {len(rivals) + 1} libellés équivalents à l'écran",
                score=best.score,
                element_text=best.element.text,
                rivals=[s.element.text for s in rivals[:4]],
                evidence=evidence,
            )
        beaten = [s.element.text for s in [best, *rivals] if s is not winner]
        best = winner
        evidence = ["ocr", "cursor"]

    if best.visible_fraction < settings.min_visible_fraction:
        return OverlayCandidate(
            **base,
            accepted=False,
            reason=(
                f"trop fugace : affiché {best.visible_fraction:.0%} du segment "
                f"(minimum {settings.min_visible_fraction:.0%})"
            ),
            score=best.score,
            element_id=best.element.id,
            element_text=best.element.text,
            visible_fraction=best.visible_fraction,
            evidence=evidence,
        )

    area = best.element.box.area
    if not settings.min_box_area <= area <= settings.max_box_area:
        return OverlayCandidate(
            **base,
            accepted=False,
            reason=(
                f"boîte aberrante : {area:.4%} de l'image (attendu entre "
                f"{settings.min_box_area:.2%} et {settings.max_box_area:.0%})"
            ),
            score=best.score,
            element_id=best.element.id,
            element_text=best.element.text,
            visible_fraction=best.visible_fraction,
            evidence=evidence,
        )

    start_offset, end_offset = display_window(decision, best.element)
    return OverlayCandidate(
        **base,
        accepted=True,
        start_offset=start_offset,
        end_offset=end_offset,
        reason=(
            "départagé par le pointeur" if "cursor" in evidence
            else "correspondance unique et stable"
        ),
        element_id=best.element.id,
        element_text=best.element.text,
        box=best.element.box,
        score=best.score,
        visible_fraction=best.visible_fraction,
        rivals=beaten,
        evidence=evidence,
        cursor_distance=cursor_distance,
    )


def match(
    decisions: list[EditDecision],
    index: ScreenTextIndex,
    config: PipelineConfig,
    cursor_track: CursorTrack | None = None,
) -> OverlayMatchReport:
    named = [
        d for d in decisions if d.ui_reference and d.ui_reference.kind == "named_control"
    ]
    candidates = [
        judge(
            d,
            gather_candidates(d, index.elements, config),
            config,
            cursor_positions_during(cursor_track, d, config),
        )
        for d in named
    ]
    return OverlayMatchReport(
        candidates=candidates,
        segments_total=len(decisions),
        segments_named=len(named),
    )


def apply_to_edl(decisions: list[EditDecision], report: OverlayMatchReport, config: PipelineConfig):
    """Reporte les correspondances retenues dans le conducteur de montage."""
    accepted = {c.segment_id: c for c in report.accepted}
    applied = 0
    for decision in decisions:
        candidate = accepted.get(decision.id)
        if candidate is None or candidate.box is None:
            continue
        decision.visual_action = VisualAction(
            type=config.overlay_matching.action_type,
            target=candidate.element_text or candidate.label,
            x=candidate.box.x,
            y=candidate.box.y,
            width=candidate.box.width,
            height=candidate.box.height,
            start_offset=candidate.start_offset,
            end_offset=candidate.end_offset,
        )
        applied += 1
    return decisions, applied


def draw_contact_sheet(
    report: OverlayMatchReport,
    config: PipelineConfig,
    out_path: Path,
    cursor_track: CursorTrack | None = None,
) -> int:
    """Planche de contact : chaque correspondance retenue, cadre dessiné.

    Valider une correspondance à l'œil prend deux secondes, en saisir les
    coordonnées à la main en prend deux minutes. C'est cette planche qui rend le
    tri praticable, davantage que le taux de détection brut.
    """
    import cv2

    accepted = [c for c in report.accepted if c.box]
    if not accepted:
        return 0

    frames_dir = config.paths.resolve("frames_dir") / "screen_text"
    decisions = {d.id: d for d in load_edl(config.paths.resolve("data_dir") / "edit_decision_list.yaml")}
    sample_fps = config.screen_text.sample_fps

    tiles = []
    for candidate in accepted:
        decision = decisions[candidate.segment_id]
        middle = (decision.source_start + decision.source_end) / 2
        frame_path = frames_dir / f"sample_{int(round(middle * sample_fps)) + 1:05d}.jpg"
        if not frame_path.exists():
            continue
        image = cv2.imread(str(frame_path))
        height, width = image.shape[:2]
        box = candidate.box
        x0, y0 = int(box.x * width), int(box.y * height)
        x1, y1 = int((box.x + box.width) * width), int((box.y + box.height) * height)
        cv2.rectangle(image, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 220, 255), 3)

        # Le pointeur est dessiné quand il a servi à trancher : sans lui, un
        # arbitrage se relit à l'aveugle. On trace toutes les positions de la
        # fenêtre, celles qu'a réellement vues l'arbitrage — la position à
        # l'instant médian n'est pas forcément l'une d'elles.
        if "cursor" in candidate.evidence and cursor_track is not None:
            for px, py in cursor_positions_during(cursor_track, decision, config):
                cv2.circle(image, (int(px * width), int(py * height)), 22, (255, 120, 0), 2)

        tile = cv2.resize(image, (width // 2, height // 2))
        # La légende va sur son propre bandeau : écrite sur l'image, elle se
        # confond avec l'interface capturée et devient illisible.
        tiles.append(_with_caption(tile, candidate))

    if not tiles:
        return 0

    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(tiles))
    return len(tiles)


def _with_caption(tile, candidate: OverlayCandidate):
    import cv2
    import numpy as np

    caption = np.zeros((34, tile.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        caption,
        f"{candidate.segment_id}  \"{candidate.label}\" -> \"{candidate.element_text}\"  "
        f"score {candidate.score:.2f}  visible {candidate.visible_fraction:.0%}  "
        f"[{'+'.join(candidate.evidence)}]",
        (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([caption, tile])


def write_report(report: OverlayMatchReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    for candidate in report.candidates:
        mark = "retenu " if candidate.accepted else "écarté "
        logger.info(
            "%s %-8s %-28s %s", mark, candidate.segment_id, repr(candidate.label)[:28], candidate.reason
        )
    logger.info(
        "%d correspondance(s) retenue(s) sur %d segment(s) nommé(s) (%d segments au total). "
        "Rapport : %s",
        len(report.accepted), report.segments_named, report.segments_total, out_path,
    )


@app.command()
def main(
    config_path: Path = typer.Option(None, help="Chemin vers config.yaml."),
    apply: bool = typer.Option(
        False, "--apply", help="Reporter les correspondances retenues dans edit_decision_list.yaml."
    ),
    contact_sheet: bool = typer.Option(
        False, "--contact-sheet", help="Écrire une planche de contact pour relecture visuelle."
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    data_dir = config.paths.resolve("data_dir")

    decisions = load_edl(data_dir / "edit_decision_list.yaml")
    index = load_screen_index(data_dir / "screen_elements.json")

    cursor_path = data_dir / "cursor_track.json"
    cursor_track = load_track(cursor_path) if cursor_path.exists() else None
    if cursor_track is None:
        logger.info(
            "Pas de trajectoire du pointeur (%s) : les ambiguïtés ne pourront pas être "
            "départagées. Lance 'python run.py --step cursor' pour l'activer.",
            cursor_path,
        )

    report = match(decisions, index, config, cursor_track)
    write_report(report, data_dir / "overlay_candidates.json")

    if contact_sheet:
        out_path = config.paths.resolve("logs_dir") / "overlay_contact_sheet.jpg"
        drawn = draw_contact_sheet(report, config, out_path, cursor_track)
        logger.info("Planche de contact : %d vignette(s) dans %s", drawn, out_path)

    if apply:
        decisions, applied = apply_to_edl(decisions, report, config)
        edl_path = data_dir / "edit_decision_list.yaml"
        edl_path.write_text(
            yaml.safe_dump([d.model_dump() for d in decisions], allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("%d incrustation(s) reportée(s) dans %s", applied, edl_path)
    elif report.accepted:
        logger.info("Relis le rapport, puis relance avec --apply pour les reporter dans l'EDL.")


if __name__ == "__main__":
    app()
