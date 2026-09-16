"""Incrustations pilotées par le pointeur, sans aucune correspondance de texte.

Un effet, le **survol** : quand le pointeur se pose sur un libellé et y reste,
ce libellé est encadré. Il est dérivé de la seule trajectoire de la souris
(`data/cursor_track.json`) et de l'index du texte à l'écran.

C'est la voie robuste du montage automatique. L'appariement entre ce que dit le
narrateur et ce que montre l'écran (`match_overlays`) échoue dès que la
narration et l'interface ne sont pas dans la même langue, ou que le narrateur
décrit au lieu de nommer. Ici rien de tout ça n'intervient : ce qui est montré
est déduit de ce que fait la souris, ce qui reste vrai quelle que soit la
langue et quoi que dise le narrateur.

Les positions ne sont interpolées qu'entre deux relevés consécutifs ; sur un
trou plus large, la dernière position est tenue plutôt que glissée vers la
suivante, ce qui inventerait un déplacement qui n'a pas eu lieu.

**Retiré : le marqueur qui suivait le pointeur.** Il produisait plus de faux
positifs qu'il n'aidait, pour deux raisons, la seconde rédhibitoire :
- il était décalé d'une trentaine de pixels, la détection situant le centre de
  la tache en mouvement quand le point actif d'une flèche est sa pointe ;
- il restait affiché sur la dernière position tenue, donc figé là où la souris
  n'était plus. Tenir la position vaut pour déduire un survol, corroboré par
  l'élément qui se trouve dessous ; ça ne vaut pas pour un marqueur qui affirme
  où est la souris.
Le rétablir supposerait de ne le dessiner que sur les intervalles réellement
détectés, pas sur les positions tenues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from overlays import enable_clause, stepped_fade
from pipeline_config import PipelineConfig
from schemas import BoundingBox, CursorSample, CursorTrack, ScreenElement

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Intervalle sur lequel la position du pointeur est décrite continûment."""

    start: float
    end: float
    from_x: float
    from_y: float
    to_x: float
    to_y: float

    @property
    def is_still(self) -> bool:
        return self.from_x == self.to_x and self.from_y == self.to_y


@dataclass
class Hover:
    """Le pointeur s'est posé sur un libellé et y est resté."""

    element: ScreenElement
    start: float
    end: float


def build_spans(samples: list[CursorSample], sample_fps: float, max_hold: float) -> list[Span]:
    """Découpe la trajectoire en intervalles décrits continûment.

    La détection ne voit le pointeur que lorsqu'il *bouge* : un pointeur posé
    sur un bouton ne produit plus aucun relevé. Un silence ne veut donc pas dire
    « position inconnue » mais « il n'a pas bougé », et la dernière position
    tient jusqu'au relevé suivant — plafonnée à `max_hold`, au-delà duquel la
    souris a pu quitter la fenêtre sans qu'on le sache.

    Deux relevés consécutifs sont en revanche reliés par une interpolation :
    là, le pointeur se déplaçait vraiment.
    """
    contiguous = 2.5 / sample_fps
    spans: list[Span] = []
    for current, following in zip(samples, samples[1:]):
        gap = following.timestamp - current.timestamp
        if gap <= contiguous:
            spans.append(
                Span(current.timestamp, following.timestamp,
                     current.x, current.y, following.x, following.y)
            )
        else:
            spans.append(
                Span(current.timestamp, min(following.timestamp, current.timestamp + max_hold),
                     current.x, current.y, current.x, current.y)
            )
    if samples:
        last = samples[-1]
        spans.append(
            Span(last.timestamp, last.timestamp + max_hold, last.x, last.y, last.x, last.y)
        )
    return spans


def held_positions(spans: list[Span], step: float) -> list[tuple[float, float, float]]:
    """Position du pointeur sur une grille régulière, trous tenus compris.

    Les survols se cherchent ici et non dans les relevés bruts : c'est
    justement quand la souris s'arrête qu'elle désigne quelque chose, et
    c'est aussi là qu'elle cesse d'être détectée.
    """
    grid: list[tuple[float, float, float]] = []
    for span in spans:
        steps = max(1, int((span.end - span.start) / step))
        for i in range(steps):
            t = span.start + i * step
            if span.is_still or span.end <= span.start:
                grid.append((t, span.from_x, span.from_y))
            else:
                ratio = (t - span.start) / (span.end - span.start)
                grid.append((
                    t,
                    span.from_x + (span.to_x - span.from_x) * ratio,
                    span.from_y + (span.to_y - span.from_y) * ratio,
                ))
    return grid


def find_hovers(
    track: CursorTrack, elements: list[ScreenElement], config: PipelineConfig
) -> list[Hover]:
    """Moments où le pointeur se pose sur un libellé et y reste.

    Aucune narration n'entre en jeu : c'est la souris qui désigne. Un survol
    trop bref est écarté — la souris ne fait que passer, et un cadre qui
    clignote au passage est pire que pas de cadre.
    """
    settings = config.cursor_overlay
    # Un contrôle porte un libellé court et étroit. Sans cette borne, l'OCR
    # fournit aussi des lignes de texte courant, qu'encadrer ferait amateur.
    usable = [
        e for e in elements
        if e.box.area <= settings.hover_max_box_area
        and e.box.width <= settings.hover_max_box_width
        and len(e.text) <= settings.hover_max_chars
    ]
    spans = build_spans(track.samples, track.sample_fps, settings.max_hold_seconds)
    grid = held_positions(spans, settings.hover_step_seconds)

    hovers: list[Hover] = []
    for timestamp, x, y in grid:
        visible = [e for e in usable if e.first_seen <= timestamp <= e.last_seen]
        near = [(_distance(e.box, x, y), e) for e in visible]
        near = [(d, e) for d, e in near if d <= settings.hover_max_distance]
        if not near:
            continue
        _, element = min(near, key=lambda pair: pair[0])

        if (
            hovers
            and hovers[-1].element.id == element.id
            and timestamp - hovers[-1].end <= settings.hover_join_seconds
        ):
            hovers[-1].end = timestamp
        else:
            hovers.append(Hover(element=element, start=timestamp, end=timestamp))

    return [
        Hover(h.element, h.start, h.end + settings.hover_tail_seconds)
        for h in hovers
        if h.end - h.start >= settings.min_hover_seconds
    ]


def _distance(box: BoundingBox, x: float, y: float) -> float:
    dx = max(box.x - x, 0.0, x - (box.x + box.width))
    dy = max(box.y - y, 0.0, y - (box.y + box.height))
    return (dx * dx + dy * dy) ** 0.5


def hover_filter(hover: Hover, config: PipelineConfig) -> str:
    settings = config.cursor_overlay
    box = hover.element.box
    pad = settings.hover_padding

    def draw(alpha: float, start: float, end: float | None) -> str:
        return (
            f"drawbox=x=(iw*{max(0.0, box.x - pad):.5f}):y=(ih*{max(0.0, box.y - pad):.5f}):"
            f"w=(iw*{min(1.0, box.width + 2 * pad):.5f}):h=(ih*{min(1.0, box.height + 2 * pad):.5f}):"
            f"color={settings.hover_color}@{alpha:.3f}:t={settings.hover_thickness}"
            f"{enable_clause(start, end)}"
        )

    return stepped_fade(config, (hover.start, hover.end), settings.hover_opacity, draw)


def _shift(value: float, offset: float) -> float:
    return round(value - offset, 3)


def cursor_filter_for(
    track: CursorTrack | None,
    elements: list[ScreenElement],
    piece_start: float,
    piece_end: float,
    config: PipelineConfig,
) -> str | None:
    """Fragment de filtre pour un morceau de vidéo, temps ramenés à son origine.

    `render_video` applique `setpts=PTS-STARTPTS` à chaque morceau : les temps
    doivent donc être exprimés depuis le début du morceau, pas depuis celui de
    la vidéo source.
    """
    settings = config.cursor_overlay
    if track is None or not settings.enabled:
        return None

    fragments: list[str] = []

    if settings.hover_enabled:
        for hover in find_hovers(track, elements, config):
            if hover.end <= piece_start or hover.start >= piece_end:
                continue
            clipped = Hover(
                hover.element,
                max(_shift(hover.start, piece_start), 0.0),
                min(_shift(hover.end, piece_start), piece_end - piece_start),
            )
            if clipped.end - clipped.start >= settings.min_hover_seconds:
                fragments.append(hover_filter(clipped, config))

    return ",".join(fragments) if fragments else None
