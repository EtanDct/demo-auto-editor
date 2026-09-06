"""Étape F : génération des effets visuels (plan-technique.md, section 3).

Construit un fragment de filtre FFmpeg par type d'effet (`VisualActionType`),
appliqué au clip du segment concerné avant le montage final. Les coordonnées
de `VisualAction` sont normalisées (0-1) et traduites en expressions FFmpeg
relatives à l'image, pour rester valides quelle que soit la résolution.

Attention, les variables de dimension diffèrent d'un filtre à l'autre :
`drawbox` ne connaît que `iw` / `ih` (dans `drawbox`, `w` et `h` désignent
la boîte, pas l'image), alors que `drawtext` expose `main_w` / `main_h`.
Utiliser `main_w` dans un `drawbox` fait échouer tout le filter_complex
("Undefined constant"), donc tout le rendu.

Affichage temporel
------------------

Un effet peut n'occuper qu'une partie de son segment (`start_offset` /
`end_offset`), ce qui évite d'encadrer un élément pendant qu'on parle d'autre
chose. Le repère de temps est celui du clip : `setpts=PTS-STARTPTS` ramène
chaque morceau à zéro dans `render_video`, donc les décalages sont bien
relatifs au début du segment.

Ce que FFmpeg permet, vérifié filtre par filtre :

- `drawbox` et `drawtext` acceptent `enable` ;
- `drawtext` accepte une expression pour `alpha`, donc un vrai fondu ;
- `drawbox` n'anime ni son alpha ni son épaisseur : son fondu est approché
  par paliers, plusieurs `drawbox` d'opacités croissantes sur des fenêtres
  successives ;
- `crop` ne supporte pas du tout `enable` ("Timeline not supported"), donc un
  `zoom` ne peut pas être minuté et couvre forcément tout son segment.

Limite connue : `zoom` reste un cadrage fixe (pas de zoom progressif façon
Ken Burns), ce qui demanderait `zoompan` et une refonte de la chaîne.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

from schemas import VisualAction

from pipeline_config import PipelineConfig

BOX_PEAK_ALPHA = 0.9
TEXT_PEAK_ALPHA = 1.0


def escape_path(path: Path | str) -> str:
    """Chemin utilisable dans un argument de filtre FFmpeg."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def text_source(text: str, config: PipelineConfig) -> str:
    """Argument FFmpeg portant un libellé, via un fichier plutôt qu'en ligne.

    L'apostrophe est inéchappable dans un argument de filtre. FFmpeg ne traite
    aucun échappement à l'intérieur d'une section entre apostrophes, et hors
    quotes aucune des formes essayées ne passe : `\\'`, `\\\\'`, et l'idiome
    `'\\''` des shells POSIX font échouer le filter_complex ou font carrément
    planter drawtext (0xC0000005). Or un libellé venu de l'OCR ou du LLM
    contient couramment « What's new » ou « d'accueil ».

    `textfile=` lit le libellé dans un fichier et ne demande aucun
    échappement. Le nom du fichier dérive du contenu : deux libellés
    identiques partagent le même, et rejouer un rendu produit les mêmes
    fichiers.
    """
    directory = config.paths.resolve("work_dir") / "text"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}.txt"
    path.write_text(text, encoding="utf-8")
    return f"textfile='{escape_path(path)}'"


def _fontfile_arg(config: PipelineConfig, action_type: str) -> str:
    """Argument `fontfile` de drawtext, obligatoire sous Windows.

    Sans `fontfile`, drawtext s'en remet à fontconfig, absent des builds
    FFmpeg Windows courants ("Cannot load default config file") : le filtre
    échoue et emporte tout le rendu. Autant refuser tout de suite, avant des
    minutes d'encodage, en pointant le réglage à renseigner.
    """
    if not config.overlays.font_path:
        if platform.system() == "Windows":
            raise ValueError(
                f"L'effet '{action_type}' affiche du texte, ce qui exige une police explicite "
                "sous Windows (FFmpeg y est compilé sans fontconfig). Renseigne "
                'overlays.font_path dans config.yaml, par exemple "C:/Windows/Fonts/arialbd.ttf".'
            )
        return ""
    return f":fontfile='{escape_path(config.overlays.font_path)}'"


def resolve_window(
    visual_action: VisualAction, segment_duration: float | None
) -> tuple[float, float | None]:
    """Fenêtre d'affichage de l'effet, en secondes depuis le début du segment."""
    start = max(0.0, visual_action.start_offset or 0.0)
    end = visual_action.end_offset
    if end is None:
        end = segment_duration
    if end is not None:
        end = min(end, segment_duration) if segment_duration is not None else end
        if end <= start:
            raise ValueError(
                f"Fenêtre d'affichage vide pour '{visual_action.target}' : "
                f"début {start}s, fin {end}s."
            )
    return start, end


def _is_always_on(window: tuple[float, float | None], segment_duration: float | None) -> bool:
    start, end = window
    return start <= 0 and (end is None or (segment_duration is not None and end >= segment_duration))


def enable_clause(start: float, end: float | None) -> str:
    if end is None:
        return f":enable='gte(t,{start:.3f})'" if start > 0 else ""
    return f":enable='between(t,{start:.3f},{end:.3f})'"


def _fade_span(window: tuple[float, float | None], fade: float) -> float:
    """Durée de fondu réellement applicable, réduite si la fenêtre est courte."""
    start, end = window
    if end is None or fade <= 0:
        return 0.0
    return min(fade, (end - start) / 2)


def _box(va: VisualAction, config: PipelineConfig, alpha: float, start: float, end: float | None) -> str:
    return (
        f"drawbox=x=(iw*{va.x}):y=(ih*{va.y}):"
        f"w=(iw*{va.width}):h=(ih*{va.height}):"
        f"color={config.overlays.highlight_color}@{alpha:.3f}:"
        f"t={config.overlays.line_width}{enable_clause(start, end)}"
    )


def stepped_fade(
    config: PipelineConfig,
    window: tuple[float, float | None],
    peak: float,
    draw,
) -> str:
    """Fondu d'un `drawbox` approché par paliers d'opacité.

    `drawbox` n'évalue pas d'expression pour son alpha : un vrai fondu
    demanderait de composer un calque à part et de le fondre avant `overlay`,
    ce qui ferait éclater la chaîne de filtres pour un gain modeste. On empile
    donc quelques `drawbox` d'opacités croissantes sur des fenêtres qui ne se
    recouvrent pas — un seul est actif à la fois.
    """
    start, end = window
    fade = _fade_span(window, config.overlays.fade_seconds)
    steps = config.overlays.fade_steps
    if fade <= 0 or steps < 2:
        return draw(peak, start, end)

    # Le dernier palier de la montée vaudrait déjà `peak` : on le laisse au
    # cœur, qui s'étend d'autant. Idem en descente. Deux `drawbox` de moins
    # pour un rendu identique.
    step = fade / steps
    ramp_slots = steps - 1
    core_start, core_end = start + ramp_slots * step, end - ramp_slots * step

    fragments = [
        draw(peak * i / steps, start + (i - 1) * step, start + i * step)
        for i in range(1, steps)
    ]
    fragments.append(draw(peak, core_start, core_end))
    fragments += [
        draw(peak * i / steps, end - i * step, end - (i - 1) * step)
        for i in range(ramp_slots, 0, -1)
    ]
    return ",".join(fragments)


def _text_alpha(window: tuple[float, float | None], fade: float, peak: float) -> str:
    """Expression d'alpha pour drawtext : un vrai fondu, lui, est possible."""
    start, end = window
    if fade <= 0 or end is None:
        return f"{peak:.3f}"
    return (
        f"'if(lt(t,{start + fade:.3f}),{peak:.3f}*(t-{start:.3f})/{fade:.3f},"
        f"if(lt(t,{end - fade:.3f}),{peak:.3f},"
        f"{peak:.3f}*({end:.3f}-t)/{fade:.3f}))'"
    )


def _text(
    va: VisualAction,
    config: PipelineConfig,
    window: tuple[float, float | None],
    font_size: int,
    x_expr: str,
    y_expr: str,
    box_opacity: float,
) -> str:
    start, end = window
    alpha = _text_alpha(window, _fade_span(window, config.overlays.fade_seconds), TEXT_PEAK_ALPHA)
    return (
        f"drawtext={text_source(va.target, config)}{_fontfile_arg(config, va.type)}:"
        f"fontsize={font_size}:fontcolor={config.overlays.callout_color}:"
        f"x={x_expr}:y={y_expr}:"
        f"box=1:boxcolor=black@{box_opacity}:boxborderw=8:"
        f"alpha={alpha}{enable_clause(start, end)}"
    )


def _highlight_box(va: VisualAction, config: PipelineConfig, window) -> str:
    return stepped_fade(
        config, window, BOX_PEAK_ALPHA, lambda a, s, e: _box(va, config, a, s, e)
    )


def _callout(va: VisualAction, config: PipelineConfig, window) -> str:
    box = _highlight_box(va, config, window)
    text = _text(
        va, config, window, font_size=28,
        x_expr=f"(main_w*{va.x})-text_w/2", y_expr=f"(main_h*{va.y})-text_h-14",
        box_opacity=0.6,
    )
    return f"{box},{text}"


def _popup(va: VisualAction, config: PipelineConfig, window) -> str:
    return _text(
        va, config, window, font_size=24,
        x_expr=f"(main_w*{va.x})+(main_w*{va.width})+16", y_expr=f"(main_h*{va.y})",
        box_opacity=0.7,
    )


def _cursor_emphasis(va: VisualAction, config: PipelineConfig, window) -> str:
    def draw(alpha: float, start: float, end: float | None) -> str:
        return (
            f"drawbox=x=(iw*{va.x}-15):y=(ih*{va.y}-15):w=30:h=30:"
            f"color={config.overlays.highlight_color}@{alpha:.3f}:t=3{enable_clause(start, end)}"
        )

    return stepped_fade(config, window, BOX_PEAK_ALPHA, draw)


def _zoom(va: VisualAction, config: PipelineConfig, window, width: int, height: int) -> str:
    return (
        f"crop=w=(iw*{va.width}):h=(ih*{va.height}):x=(iw*{va.x}):y=(ih*{va.y}),"
        f"scale={width}:{height}"
    )


def overlay_filter_for(
    visual_action: VisualAction | None,
    config: PipelineConfig,
    width: int,
    height: int,
    segment_duration: float | None = None,
) -> str | None:
    """Retourne un fragment de filtre FFmpeg (sans crochets d'E/S) ou None si aucun effet."""
    if visual_action is None:
        return None

    window = resolve_window(visual_action, segment_duration)

    if visual_action.type == "zoom":
        if not _is_always_on(window, segment_duration):
            raise ValueError(
                "Un 'zoom' ne peut pas être minuté : il repose sur `crop`, que FFmpeg "
                "n'expose pas à la timeline ('Timeline not supported'). Retire "
                "start_offset / end_offset, ou choisis un autre effet."
            )
        return _zoom(visual_action, config, window, width, height)

    builders = {
        "highlight": _highlight_box,
        "callout": _callout,
        "popup": _popup,
        "cursor_emphasis": _cursor_emphasis,
    }
    builder = builders.get(visual_action.type)
    if builder is None:
        raise ValueError(f"Type d'effet inconnu : {visual_action.type}")
    return builder(visual_action, config, window)
