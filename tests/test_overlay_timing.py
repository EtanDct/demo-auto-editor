"""Tests de l'affichage temporel des incrustations (étape F).

Ces filtres finissent dans un `filter_complex` unique : une expression invalide
n'échoue pas à la construction, elle fait tomber tout le rendu après plusieurs
minutes d'encodage. Les assertions sur les chaînes ne suffisent donc pas — les
fragments sont aussi passés à FFmpeg.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from match_overlays import display_window
from overlays import overlay_filter_for, resolve_window
from schemas import ScreenElement, VisualAction

from test_match_overlays import decision, element

FFMPEG = shutil.which("ffmpeg")


@pytest.fixture
def config_with_font(config):
    """drawtext exige une police explicite sous Windows."""
    patched = config.model_copy(deep=True)
    patched.overlays.font_path = "C:/Windows/Fonts/arial.ttf"
    return patched


def action(action_type: str = "highlight", **kwargs) -> VisualAction:
    return VisualAction(
        type=action_type, target="Pull requests",
        **{"x": 0.25, "y": 0.35, "width": 0.3, "height": 0.18, **kwargs},
    )


# --- fenêtre --------------------------------------------------------------

def test_sans_decalage_l_effet_couvre_tout_le_segment():
    assert resolve_window(action(), 5.0) == (0.0, 5.0)


def test_la_fenetre_est_bornee_a_la_duree_du_segment():
    """Un `end_offset` trop grand déborderait sur le morceau suivant."""
    assert resolve_window(action(end_offset=9.0), 5.0) == (0.0, 5.0)


def test_une_fenetre_vide_est_refusee():
    with pytest.raises(ValueError, match="Fenêtre d'affichage vide"):
        resolve_window(action(start_offset=4.0), 3.0)


def test_le_schema_refuse_une_fin_avant_le_debut():
    with pytest.raises(ValueError):
        action(start_offset=3.0, end_offset=1.0)


# --- génération -----------------------------------------------------------

def test_un_effet_minute_porte_une_condition_de_temps(config):
    fragment = overlay_filter_for(action(start_offset=1.0, end_offset=3.5), config, 640, 360, 5.0)

    assert "enable='between(t,1.000," in fragment
    assert fragment.rstrip().endswith("3.500)'")


def test_le_fondu_produit_des_paliers_d_opacite_croissants(config):
    fragment = overlay_filter_for(action(start_offset=1.0, end_offset=3.5), config, 640, 360, 5.0)
    alphas = [float(part.split("@")[1].split(":")[0]) for part in fragment.split("drawbox=")[1:]]

    montee = alphas[: config.overlays.fade_steps - 1]
    descente = alphas[-(config.overlays.fade_steps - 1) :]
    assert montee == sorted(montee)
    assert descente == sorted(descente, reverse=True)
    assert max(alphas) == pytest.approx(0.9)


def test_sans_fondu_un_seul_cadre_est_dessine(config):
    patched = config.model_copy(deep=True)
    patched.overlays.fade_seconds = 0.0

    fragment = overlay_filter_for(action(start_offset=1.0, end_offset=3.5), patched, 640, 360, 5.0)

    assert fragment.count("drawbox=") == 1


def test_une_fenetre_courte_raccourcit_le_fondu_au_lieu_de_deborder(config):
    """Sinon la montée empiéterait sur la descente et l'effet clignoterait."""
    fragment = overlay_filter_for(action(start_offset=1.0, end_offset=1.2), config, 640, 360, 5.0)
    bornes = [part.split("between(t,")[1].split(")")[0] for part in fragment.split("enable='")[1:]]
    debuts = [float(b.split(",")[0]) for b in bornes]

    assert debuts == sorted(debuts)
    assert all(1.0 <= d <= 1.2 for d in debuts)


def test_un_zoom_minute_est_refuse_avec_l_explication(config):
    """`crop` n'est pas exposé à la timeline par FFmpeg."""
    with pytest.raises(ValueError, match="Timeline not supported"):
        overlay_filter_for(action("zoom", start_offset=1.0, end_offset=2.0), config, 640, 360, 5.0)


def test_un_zoom_pleine_duree_reste_possible(config):
    assert overlay_filter_for(action("zoom"), config, 640, 360, 5.0).startswith("crop=")


def test_le_texte_fond_par_une_expression_d_alpha(config_with_font):
    """drawtext, lui, accepte une expression : vrai fondu plutôt que paliers."""
    fragment = overlay_filter_for(
        action("popup", start_offset=1.0, end_offset=3.5), config_with_font, 640, 360, 5.0
    )

    assert "alpha='if(lt(t," in fragment
    assert fragment.count("drawtext=") == 1


# --- FFmpeg accepte-t-il vraiment ces fragments ? -------------------------

@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg absent du PATH")
@pytest.mark.parametrize(
    "action_type,kwargs",
    [
        ("highlight", {}),
        ("highlight", {"start_offset": 1.0, "end_offset": 3.5}),
        ("highlight", {"start_offset": 1.0, "end_offset": 1.2}),
        ("callout", {"start_offset": 1.0, "end_offset": 3.5}),
        ("popup", {"start_offset": 0.5, "end_offset": 4.0}),
        ("cursor_emphasis", {"start_offset": 1.0, "end_offset": 2.0}),
        ("zoom", {}),
    ],
)
def test_ffmpeg_accepte_le_fragment(action_type, kwargs, config_with_font):
    fragment = overlay_filter_for(action(action_type, **kwargs), config_with_font, 640, 360, 5.0)

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x180:duration=5", "-vf", fragment, "-f", "null", "-"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr[:300]


# --- fenêtre proposée par l'appariement -----------------------------------

def test_un_element_present_tout_le_segment_ne_borne_rien(config):
    """Le conducteur de montage reste lisible : pas de bornes inutiles."""
    assert display_window(decision("Post", 10.0, 20.0), element("Post", 5.0, 40.0)) == (None, None)


def test_l_effet_est_borne_a_la_presence_de_l_element():
    """Un cadre dessiné alors que l'élément a disparu se voit immédiatement."""
    start, end = display_window(decision("Post", 10.0, 20.0), element("Post", 12.0, 17.0))

    assert (start, end) == (2.0, 7.0)


def test_les_bornes_ne_sortent_jamais_du_segment():
    start, end = display_window(decision("Post", 10.0, 20.0), element("Post", 5.0, 17.0))

    assert start == 0.0 and end == 7.0
