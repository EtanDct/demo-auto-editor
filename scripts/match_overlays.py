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
from pipeline_config import PipelineConfig, load_config
from schemas import (
    BoundingBox,
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


def judge(
    decision: EditDecision, scored: list[ScoredElement], config: PipelineConfig
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
    if rivals:
        return OverlayCandidate(
            **base,
            accepted=False,
            reason=f"ambigu : {len(rivals) + 1} libellés équivalents à l'écran",
            score=best.score,
            element_text=best.element.text,
            rivals=[s.element.text for s in rivals[:4]],
        )

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
        )

    return OverlayCandidate(
        **base,
        accepted=True,
        reason="correspondance unique et stable",
        element_id=best.element.id,
        element_text=best.element.text,
        box=best.element.box,
        score=best.score,
        visible_fraction=best.visible_fraction,
    )


def match(
    decisions: list[EditDecision], index: ScreenTextIndex, config: PipelineConfig
) -> OverlayMatchReport:
    named = [
        d for d in decisions if d.ui_reference and d.ui_reference.kind == "named_control"
    ]
    candidates = [judge(d, gather_candidates(d, index.elements, config), config) for d in named]
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
        )
        applied += 1
    return decisions, applied


def draw_contact_sheet(report: OverlayMatchReport, config: PipelineConfig, out_path: Path) -> int:
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
        f"score {candidate.score:.2f}  visible {candidate.visible_fraction:.0%}",
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
    report = match(decisions, index, config)
    write_report(report, data_dir / "overlay_candidates.json")

    if contact_sheet:
        out_path = config.paths.resolve("logs_dir") / "overlay_contact_sheet.jpg"
        drawn = draw_contact_sheet(report, config, out_path)
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
