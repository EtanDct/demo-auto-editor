"""Mise en page de la vidéo livrée : un canevas fixe, et deux bandes.

Le recadrage retire le bandeau du navigateur, dont la hauteur change d'une
capture à l'autre (avec ou sans favoris, un navigateur ou un autre). Livrer
cette hauteur telle quelle donnerait des vidéos de formats différents ; on pose
donc l'image sur un canevas fixe (1920x1080 par défaut), et la place libérée
devient deux bandes :

- **en haut**, le chapitre en cours, le rappel de l'élément encadré, le logo, et
  un filet de progression qui sépare la bande de l'application ;
- **en bas**, les sous-titres, qui ne masquent ainsi plus l'interface.

Rien n'est fixé en dur. La hauteur des bandes est ce qui reste une fois l'image
posée, et la mise en page se dégrade proprement quand la place manque : la
bande du haut disparaît d'abord, puis les sous-titres reviennent sur l'image.
Une capture qui fait déjà la taille du canevas est livrée telle quelle.

Ordre dans le graphe de filtres : les cadres (`overlays`, `cursor_overlays`)
sont dessinés sur l'image rognée, en coordonnées normalisées ; ils le sont
AVANT la mise en page, sans quoi le décalage de la bande du haut les ferait
tomber à côté. Les bandes s'ajoutent après la concaténation des morceaux, et
leurs horaires sont donc ceux de la vidéo finale.

Deux limites de FFmpeg ont dicté la construction, vérifiées à part :
- `drawbox` n'évalue pas sa largeur image par image : une barre de progression
  en `drawbox` est pleine dès la première image. Elle est donc faite d'une bande
  de couleur qui glisse sous un `overlay`, dont la position, elle, s'anime ;
- l'alpha de `drawtext` s'applique aussi à son fond (`box`) : la pastille du
  rappel fond d'un bloc, sans calque séparé.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from overlays import _fontfile_arg, _text_alpha, escape_path, resolve_window, text_source
from pipeline_config import PipelineConfig
from schemas import ChapterPlan

# Hauteur de ligne de la police rapportée à sa taille, à l'échelle d'Arial : sert
# à centrer un texte dans une bande, et à borner la taille des sous-titres.
LINE_HEIGHT = 1.2


def _even(value: float) -> int:
    """Dimension compatible avec le sous-échantillonnage 4:2:0 du H.264."""
    number = int(value)
    return number - number % 2


@dataclass(frozen=True)
class Layout:
    """Placement de l'image rognée sur le canevas, et bandes qui en résultent."""

    width: int
    height: int
    content_width: int
    content_height: int
    content_x: int
    content_y: int
    top_height: int
    bottom_height: int
    scaled: bool = False

    @property
    def subtitles_in_band(self) -> bool:
        return self.bottom_height > 0

    @property
    def needs_pad(self) -> bool:
        return (self.width, self.height) != (self.content_width, self.content_height)

    @property
    def bottom_top(self) -> int:
        """Ordonnée du haut de la bande basse."""
        return self.content_y + self.content_height


def compute_layout(content_width: int, content_height: int, config: PipelineConfig) -> Layout:
    """Pose l'image sur le canevas et répartit la place libre entre les bandes."""
    settings = config.layout
    if not settings.enabled:
        return Layout(content_width, content_height, content_width, content_height, 0, 0, 0, 0)

    width, height = _even(settings.width), _even(settings.height)
    # Mise à l'échelle, sans déformation, selon deux règles :
    # - une image qui déborde du canevas est réduite juste assez pour y tenir ;
    # - une image plus petite est agrandie, mais seulement tant que les bandes
    #   tiennent encore : agrandir jusqu'à remplir le canevas ferait disparaître
    #   les sous-titres de leur bande.
    # On ne rétrécit jamais une image pour faire de la place aux bandes : une
    # capture pleine à la taille du canevas y perdrait en lisibilité.
    fit = min(width / content_width, height / content_height)
    if fit < 1:
        ratio = fit
    else:
        with_bands = (height - _even(settings.top_height) - settings.min_bottom_height) / content_height
        ratio = min(fit, max(1.0, with_bands))
    if abs(ratio - 1) < 1e-9:
        new_width, new_height = content_width, content_height
    else:
        new_width, new_height = _even(content_width * ratio), _even(content_height * ratio)

    slack = height - new_height
    top_wanted = _even(settings.top_height)
    if slack >= top_wanted + settings.min_bottom_height:
        top, bottom, y = top_wanted, slack - top_wanted, top_wanted
    elif slack >= settings.min_bottom_height:
        # Les sous-titres passent avant le chapitre : ils portent le propos.
        top, bottom, y = 0, slack, 0
    else:
        # Trop peu de place pour une bande lisible : on centre, sans bande.
        top, bottom, y = 0, 0, _even(slack / 2)

    return Layout(
        width=width,
        height=height,
        content_width=new_width,
        content_height=new_height,
        content_x=_even((width - new_width) / 2),
        content_y=y,
        top_height=top,
        bottom_height=bottom,
        scaled=(new_width, new_height) != (content_width, content_height),
    )


# --- contenu des bandes ------------------------------------------------------

@dataclass(frozen=True)
class TimedText:
    """Un texte affiché dans une bande, aux horaires de la vidéo finale."""

    text: str
    start: float
    end: float
    index: str = ""


@dataclass
class BandContent:
    """Ce qui s'affiche dans la bande du haut au fil de la vidéo."""

    total_duration: float
    chapters: list[TimedText] = field(default_factory=list)
    title: str | None = None
    echoes: list[TimedText] = field(default_factory=list)


def piece_output_starts(pieces) -> list[float]:
    """Instant, dans la vidéo finale, où commence chaque morceau."""
    starts, cursor = [], 0.0
    for piece in pieces:
        starts.append(cursor)
        cursor += (piece.end - piece.start) + piece.extension
    return starts


def chapter_windows(plan: ChapterPlan | None, pieces, total_duration: float) -> list[TimedText]:
    """Horaires de chaque chapitre dans la vidéo finale.

    Un chapitre commence avec le plan de sa première phrase, et l'intervalle
    muet qui précède appartient encore au chapitre d'avant : on ne change pas
    d'étape avant que le narrateur l'ait annoncée. Le premier commence à zéro.
    """
    if plan is None or not plan.chapters:
        return []
    start_by_segment = {
        piece.decision.id: start
        for piece, start in zip(pieces, piece_output_starts(pieces))
        if piece.kind == "segment" and piece.decision is not None
    }
    placed = [
        (chapter.title, start_by_segment[chapter.first_segment])
        for chapter in plan.chapters
        if chapter.first_segment in start_by_segment
    ]
    if not placed:
        return []
    placed[0] = (placed[0][0], 0.0)

    ends = [start for _, start in placed[1:]] + [total_duration]
    return [
        TimedText(title, start, end, index=f"{index}/{len(placed)}")
        for index, ((title, start), end) in enumerate(zip(placed, ends), 1)
        if end > start
    ]


def echo_windows(pieces) -> list[TimedText]:
    """Horaires du rappel de chaque élément encadré par l'appariement.

    Seuls les cadres du conducteur de montage sont rappelés : ils viennent de
    l'appariement narrateur/écran, qui a vérifié que le libellé cité est unique
    et visible. Les survols de la souris ne passent pas par là.
    """
    echoes = []
    for piece, start in zip(pieces, piece_output_starts(pieces)):
        action = piece.decision.visual_action if piece.decision is not None else None
        if piece.kind != "segment" or action is None or not action.target.strip():
            continue
        length = piece.end - piece.start
        try:
            window_start, window_end = resolve_window(action, length)
        except ValueError:
            continue
        echoes.append(
            TimedText(action.target.strip(), start + window_start, start + (window_end or length))
        )
    return echoes


# --- graphe de filtres -------------------------------------------------------

def _between(start: float, end: float) -> str:
    return f":enable='between(t,{start:.3f},{end:.3f})'"


def _band_text(
    text: str,
    config: PipelineConfig,
    window: TimedText,
    x: str,
    band_height: int,
    size: int,
    color: str,
    extra: str = "",
) -> str:
    fade = min(config.layout.fade_seconds, (window.end - window.start) / 2)
    alpha = _text_alpha((window.start, window.end), fade, 1.0)
    # `y_align=font` aligne sur la hauteur de la police et non sur celle des
    # glyphes rendus : « Filters » et « Key figures » tombent à la même hauteur,
    # sans sautiller d'un chapitre à l'autre.
    y = round((band_height - size * LINE_HEIGHT) / 2)
    return (
        f"drawtext={text_source(text, config)}{_fontfile_arg(config, 'layout')}:"
        f"fontsize={size}:fontcolor={color}:x={x}:y={y}:y_align=font{extra}:"
        f"alpha={alpha}{_between(window.start, window.end)}"
    )


def top_band_filters(layout: Layout, content: BandContent, config: PipelineConfig) -> list[str]:
    """Textes de la bande du haut : chapitre à gauche, rappel au centre."""
    if layout.top_height <= 0:
        return []
    settings = config.layout
    size, pad = settings.top_text_size, settings.band_padding
    fragments = []

    if content.chapters:
        # L'index dans une colonne de largeur fixe : les titres s'alignent d'un
        # chapitre à l'autre, sans mesurer le texte.
        title_x = pad + round(size * 2.6)
        for chapter in content.chapters:
            fragments.append(
                _band_text(
                    chapter.index, config, chapter, str(pad), layout.top_height, size,
                    settings.muted_color,
                )
            )
            fragments.append(
                _band_text(
                    chapter.text, config, chapter, str(title_x), layout.top_height, size,
                    settings.text_color,
                )
            )
    elif content.title:
        whole = TimedText(content.title, 0.0, content.total_duration)
        fragments.append(
            _band_text(content.title, config, whole, str(pad), layout.top_height, size, settings.text_color)
        )

    if settings.echo_enabled:
        for echo in content.echoes:
            fragments.append(
                _band_text(
                    echo.text, config, echo, "(w-text_w)/2", layout.top_height, size,
                    settings.echo_text_color,
                    extra=f":box=1:boxcolor={settings.accent_color}:boxborderw=10",
                )
            )
    return fragments


def progress_y(layout: Layout, config: PipelineConfig) -> int | None:
    """Le filet sépare la bande du haut de l'application ; à défaut, borde le bas."""
    thickness = config.layout.progress_thickness
    if layout.top_height > 0:
        return layout.top_height - thickness
    if layout.bottom_height > 0:
        return layout.height - thickness
    return None


def build_layout_chains(
    input_label: str,
    layout: Layout,
    content: BandContent,
    config: PipelineConfig,
    fps: float,
    subtitles_path,
    logo_input: int | None = None,
) -> tuple[list[str], str]:
    """Chaînes qui posent l'image sur le canevas, habillent les bandes, sous-titrent."""
    settings = config.layout
    chains: list[str] = []
    steps: list[str] = []

    if layout.scaled:
        steps.append(f"scale={layout.content_width}:{layout.content_height}")
    if layout.needs_pad:
        steps.append(
            f"pad={layout.width}:{layout.height}:{layout.content_x}:{layout.content_y}:"
            f"color={settings.background}"
        )
    steps += top_band_filters(layout, content, config)

    y = progress_y(layout, config) if settings.progress_enabled else None
    if y is not None and content.total_duration > 0:
        steps.append(
            f"drawbox=x=0:y={y}:w={layout.width}:h={settings.progress_thickness}:"
            f"color={settings.track_color}:t=fill"
        )

    label = input_label
    if steps:
        chains.append(f"[{label}]{','.join(steps)}[vlayout]")
        label = "vlayout"

    if y is not None and content.total_duration > 0:
        chains.append(
            f"color=c={settings.accent_color}:s={layout.width}x{settings.progress_thickness}:r={fps}[pbar]"
        )
        ticks = "".join(
            f",drawbox=x={round(layout.width * c.start / content.total_duration) - 2}:y={y}:"
            f"w=4:h={settings.progress_thickness}:color={settings.track_color}:t=fill"
            for c in content.chapters[1:]
        )
        chains.append(
            f"[{label}][pbar]overlay=x='-w+w*t/{content.total_duration:.3f}':y={y}:"
            f"shortest=1{ticks}[vprogress]"
        )
        label = "vprogress"

    if logo_input is not None and layout.top_height > 0:
        chains.append(f"[{logo_input}:v]scale=-2:{settings.logo_height}[logo]")
        chains.append(
            f"[{label}][logo]overlay=x=W-w-{settings.band_padding}:"
            f"y={round((layout.top_height - settings.logo_height) / 2)}[vlogo]"
        )
        label = "vlogo"

    chains.append(f"[{label}]subtitles='{escape_path(subtitles_path)}'[vout]")
    return chains, "[vout]"
