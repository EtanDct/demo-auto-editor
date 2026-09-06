"""Incrustations pilotées par le pointeur, sans aucune correspondance de texte.

Deux effets, tous deux dérivés de la seule trajectoire de la souris
(`data/cursor_track.json`) et de l'index du texte à l'écran :

- **le marqueur de suivi** : un cadre qui accompagne le pointeur, pour que
  l'œil sache où regarder ;
- **le survol** : quand le pointeur se pose sur un libellé et y reste, ce
  libellé est encadré.

C'est la voie robuste du montage automatique. L'appariement entre ce que dit le
narrateur et ce que montre l'écran (`match_overlays`) échoue dès que la
narration et l'interface ne sont pas dans la même langue, ou que le narrateur
décrit au lieu de nommer. Ici rien de tout ça n'intervient : ce qui est montré
est déduit de ce que fait la souris, ce qui reste vrai quelle que soit la
langue et quoi que dise le narrateur.

Mécanique FFmpeg : `drawbox` évalue bien `t` dans `x` et `y` (à l'inverse de
son épaisseur), donc un seul `drawbox` suffit à faire suivre le pointeur, avec
une expression affine par morceaux. Les positions ne sont interpolées qu'entre
deux relevés consécutifs ; sur un trou plus large, la dernière position est
tenue plutôt que glissée vers la suivante, ce qui inventerait un déplacement
qui n'a pas eu lieu.
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


def group_runs(spans: list[Span]) -> list[list[Span]]:
    """Regroupe les intervalles jointifs : un `drawbox` par groupe."""
    runs: list[list[Span]] = []
    for span in spans:
        if runs and abs(runs[-1][-1].end - span.start) < 1e-6:
            runs[-1].append(span)
        else:
            runs.append([span])
    return runs


def _axis_expression(run: list[Span], axis: str, dimension: str) -> str:
    """Expression affine par morceaux de la position du pointeur sur un axe."""
    def value(span: Span) -> str:
        start_value = getattr(span, f"from_{axis}")
        end_value = getattr(span, f"to_{axis}")
        if span.is_still:
            return f"({dimension}*{start_value:.5f})"
        slope = (end_value - start_value) / (span.end - span.start)
        return f"({dimension}*({start_value:.5f}+{slope:.5f}*(t-{span.start:.3f})))"

    expression = value(run[-1])
    for span in reversed(run[:-1]):
        expression = f"if(lt(t,{span.end:.3f}),{value(span)},{expression})"
    return expression


def follow_filter(run: list[Span], config: PipelineConfig) -> str:
    """Un `drawbox` centré sur le pointeur, actif sur toute la durée du groupe."""
    settings = config.cursor_overlay
    half = settings.marker_size / 2
    x = f"({_axis_expression(run, 'x', 'iw')})-(iw*{half})"
    y = f"({_axis_expression(run, 'y', 'ih')})-(ih*{half})"
    return (
        f"drawbox=x='{x}':y='{y}':w=(iw*{settings.marker_size}):h=(ih*{settings.marker_size}):"
        f"color={settings.marker_color}@{settings.marker_opacity}:"
        f"t={settings.marker_thickness}"
        f"{enable_clause(run[0].start, run[-1].end)}"
    )


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

    if settings.follow_enabled:
        inside = [
            s for s in track.samples if piece_start - 1.0 <= s.timestamp <= piece_end + 1.0
        ]
        spans = build_spans(inside, track.sample_fps, settings.max_hold_seconds)
        for run in group_runs(spans):
            clipped = [
                Span(
                    max(_shift(s.start, piece_start), 0.0),
                    min(_shift(s.end, piece_start), piece_end - piece_start),
                    s.from_x, s.from_y, s.to_x, s.to_y,
                )
                for s in run
                if s.end > piece_start and s.start < piece_end
            ]
            clipped = [s for s in clipped if s.end > s.start]
            if clipped:
                fragments.append(follow_filter(clipped, config))

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
