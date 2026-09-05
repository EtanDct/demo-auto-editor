"""Tests des incrustations pilotées par la souris.

C'est la voie robuste du montage automatique : elle ne dépend d'aucune
correspondance de texte, donc d'aucune hypothèse sur la langue de l'interface
ni sur ce que dit le narrateur.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from cursor_overlays import (
    Span,
    build_spans,
    cursor_filter_for,
    find_hovers,
    follow_filter,
    group_runs,
    held_positions,
)
from schemas import BoundingBox, CursorSample, CursorTrack, ScreenElement

FFMPEG = shutil.which("ffmpeg")
FPS = 8.0


def track(*samples) -> CursorTrack:
    return CursorTrack(
        sample_fps=FPS,
        samples=[CursorSample(timestamp=t, x=x, y=y) for t, x, y in samples],
    )


def element(text: str, x: float = 0.2, y: float = 0.3, first: float = 0.0, last: float = 100.0):
    return ScreenElement(
        id=f"scr-{abs(hash((text, x))) % 10000:04d}",
        text=text,
        box=BoundingBox(x=x, y=y, width=0.08, height=0.03),
        first_seen=first, last_seen=last, confidence=0.95, occurrences=10,
    )


# --- trajectoire ----------------------------------------------------------

def test_deux_releves_consecutifs_sont_relies_par_une_interpolation():
    spans = build_spans(track((0.0, 0.1, 0.1), (0.125, 0.2, 0.2)).samples, FPS, max_hold=6.0)

    assert not spans[0].is_still
    assert (spans[0].to_x, spans[0].to_y) == (0.2, 0.2)


def test_un_silence_signifie_que_le_pointeur_n_a_pas_bouge():
    """La détection ne voit le pointeur que lorsqu'il se déplace : un trou veut
    dire immobile, pas inconnu. Glisser jusqu'au relevé suivant inventerait un
    déplacement qui n'a pas eu lieu."""
    spans = build_spans(track((0.0, 0.1, 0.1), (3.0, 0.8, 0.8)).samples, FPS, max_hold=6.0)

    assert spans[0].is_still
    assert (spans[0].start, spans[0].end) == (0.0, 3.0)


def test_une_position_tenue_finit_par_expirer():
    """Au-delà, la souris a pu quitter la fenêtre sans qu'on le sache."""
    spans = build_spans(track((0.0, 0.1, 0.1), (30.0, 0.8, 0.8)).samples, FPS, max_hold=6.0)

    assert spans[0].end == 6.0


def test_les_intervalles_jointifs_forment_un_seul_groupe():
    spans = build_spans(
        track((0.0, 0.1, 0.1), (0.125, 0.2, 0.2), (0.25, 0.3, 0.3)).samples, FPS, max_hold=6.0
    )

    assert len(group_runs(spans)) == 1


def test_un_trou_expire_coupe_le_groupe():
    spans = build_spans(track((0.0, 0.1, 0.1), (30.0, 0.8, 0.8)).samples, FPS, max_hold=6.0)

    assert len(group_runs(spans)) == 2


def test_la_grille_tient_la_position_pendant_un_arret():
    spans = [Span(0.0, 1.0, 0.4, 0.5, 0.4, 0.5)]

    grid = held_positions(spans, step=0.25)

    assert [(round(t, 2), x, y) for t, x, y in grid] == [
        (0.0, 0.4, 0.5), (0.25, 0.4, 0.5), (0.5, 0.4, 0.5), (0.75, 0.4, 0.5)
    ]


# --- survols --------------------------------------------------------------

def test_un_pointeur_pose_sur_un_libelle_le_fait_encadrer(config):
    """Le cas qui compte : la souris s'arrête sur un bouton et ne bouge plus,
    donc n'est plus détectée. Sans position tenue, aucun survol ne sortirait."""
    hovers = find_hovers(track((0.0, 0.24, 0.31), (3.0, 0.9, 0.9)), [element("Home")], config)

    assert len(hovers) == 1
    assert hovers[0].element.text == "Home"
    assert hovers[0].end - hovers[0].start >= config.cursor_overlay.min_hover_seconds


def test_une_souris_qui_ne_fait_que_passer_n_encadre_rien(config):
    """Un cadre qui clignote au passage est pire que pas de cadre."""
    samples = [(0.0, 0.24, 0.31), (0.125, 0.5, 0.5), (0.25, 0.8, 0.8)]

    assert find_hovers(track(*samples), [element("Home")], config) == []


def test_un_pointeur_loin_de_tout_libelle_n_encadre_rien(config):
    assert find_hovers(track((0.0, 0.9, 0.9), (3.0, 0.9, 0.9)), [element("Home")], config) == []


def test_un_element_absent_de_l_ecran_a_ce_moment_est_ignore(config):
    hovers = find_hovers(
        track((10.0, 0.24, 0.31), (13.0, 0.9, 0.9)), [element("Home", last=5.0)], config
    )

    assert hovers == []


def test_un_bloc_de_texte_courant_n_est_pas_encadre(config):
    """Un contrôle porte un libellé court ; encadrer une phrase ferait amateur."""
    prose = element("Look-ahead and point-in-time fixes across the data layers")

    assert find_hovers(track((0.0, 0.24, 0.31), (3.0, 0.9, 0.9)), [prose], config) == []


# --- fragments FFmpeg -----------------------------------------------------

def test_le_marqueur_suit_le_pointeur_dans_le_temps(config):
    run = [Span(0.0, 0.5, 0.1, 0.1, 0.6, 0.6)]

    fragment = follow_filter(run, config)

    assert fragment.startswith("drawbox=")
    assert "t-0.000" in fragment  # position affine en fonction du temps


def test_un_pointeur_immobile_donne_une_position_constante(config):
    run = [Span(0.0, 1.0, 0.4, 0.5, 0.4, 0.5)]

    fragment = follow_filter(run, config)

    assert "(iw*0.40000)" in fragment
    assert "t-" not in fragment.split("enable")[0]


def test_les_temps_sont_ramenes_a_l_origine_du_morceau(config):
    """`render_video` applique `setpts=PTS-STARTPTS` : un temps absolu décalerait
    toutes les incrustations du morceau."""
    fragment = cursor_filter_for(
        track((20.0, 0.24, 0.31), (23.0, 0.9, 0.9)), [element("Home")], 20.0, 26.0, config
    )

    assert fragment is not None
    assert "between(t,0." in fragment or "between(t,1." in fragment
    assert "between(t,2" not in fragment.replace("between(t,2.", "X")  # pas de temps absolu


def test_sans_trajectoire_aucune_incrustation(config):
    assert cursor_filter_for(None, [element("Home")], 0.0, 10.0, config) is None


def test_les_incrustations_du_pointeur_sont_desactivables(config):
    patched = config.model_copy(deep=True)
    patched.cursor_overlay.enabled = False

    assert cursor_filter_for(track((0.0, 0.24, 0.31)), [element("Home")], 0.0, 10.0, patched) is None


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg absent du PATH")
def test_ffmpeg_accepte_le_fragment_complet(config):
    """Une expression invalide ne se voit pas dans une chaîne : elle fait tomber
    tout le rendu après plusieurs minutes d'encodage."""
    fragment = cursor_filter_for(
        track((0.0, 0.1, 0.1), (0.125, 0.24, 0.31), (0.25, 0.24, 0.31), (3.0, 0.9, 0.9)),
        [element("Home")], 0.0, 5.0, config,
    )

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x180:duration=5", "-vf", fragment, "-f", "null", "-"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr[:300]
