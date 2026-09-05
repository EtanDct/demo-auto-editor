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

Limites connues (v1, à affiner une fois un extrait vidéo réel disponible) :
- pas d'animation d'entrée/sortie progressive (affichage statique pendant
  toute la durée du segment) ;
- `zoom` est un cadrage fixe (pas de zoom progressif façon Ken Burns).
"""

from __future__ import annotations

import platform

from schemas import VisualAction

from pipeline_config import PipelineConfig


def _escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


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
    escaped = config.overlays.font_path.replace("\\", "/").replace(":", "\\:")
    return f":fontfile='{escaped}'"


def _highlight_box(va: VisualAction, config: PipelineConfig) -> str:
    return (
        f"drawbox=x=(iw*{va.x}):y=(ih*{va.y}):"
        f"w=(iw*{va.width}):h=(ih*{va.height}):"
        f"color={config.overlays.highlight_color}@0.9:t={config.overlays.line_width}"
    )


def _callout(va: VisualAction, config: PipelineConfig) -> str:
    box = _highlight_box(va, config)
    label = _escape_text(va.target)
    text = (
        f"drawtext=text='{label}'{_fontfile_arg(config, va.type)}:fontsize=28:"
        f"fontcolor={config.overlays.callout_color}:"
        f"x=(main_w*{va.x})-text_w/2:y=(main_h*{va.y})-text_h-14:"
        f"box=1:boxcolor=black@0.6:boxborderw=8"
    )
    return f"{box},{text}"


def _popup(va: VisualAction, config: PipelineConfig) -> str:
    label = _escape_text(va.target)
    return (
        f"drawtext=text='{label}'{_fontfile_arg(config, va.type)}:fontsize=24:"
        f"fontcolor={config.overlays.callout_color}:"
        f"x=(main_w*{va.x})+(main_w*{va.width})+16:y=(main_h*{va.y}):"
        f"box=1:boxcolor=black@0.7:boxborderw=8"
    )


def _cursor_emphasis(va: VisualAction, config: PipelineConfig) -> str:
    return (
        f"drawbox=x=(iw*{va.x}-15):y=(ih*{va.y}-15):w=30:h=30:"
        f"color={config.overlays.highlight_color}@0.9:t=3"
    )


def _zoom(va: VisualAction, config: PipelineConfig, width: int, height: int) -> str:
    return (
        f"crop=w=(iw*{va.width}):h=(ih*{va.height}):x=(iw*{va.x}):y=(ih*{va.y}),"
        f"scale={width}:{height}"
    )


def overlay_filter_for(
    visual_action: VisualAction | None, config: PipelineConfig, width: int, height: int
) -> str | None:
    """Retourne un fragment de filtre FFmpeg (sans crochets d'E/S) ou None si aucun effet."""
    if visual_action is None:
        return None

    builders = {
        "highlight": lambda: _highlight_box(visual_action, config),
        "callout": lambda: _callout(visual_action, config),
        "popup": lambda: _popup(visual_action, config),
        "cursor_emphasis": lambda: _cursor_emphasis(visual_action, config),
        "zoom": lambda: _zoom(visual_action, config, width, height),
    }
    builder = builders.get(visual_action.type)
    if builder is None:
        raise ValueError(f"Type d'effet inconnu : {visual_action.type}")
    return builder()
