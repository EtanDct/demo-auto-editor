"""Tests du suivi du pointeur (étape `cursor`).

La détection ne reconnaît aucune forme de curseur : elle isole la position à
l'instant *i* comme la seule zone mobile commune à `diff(i-1, i)` et
`diff(i, i+1)`. Les tests portent sur cette intersection, sur des images de
synthèse — c'est la logique qui décide, pas l'apparence du pointeur.
"""

from __future__ import annotations

import numpy as np
import pytest

from detect_cursor import position_at, track_cursor
from schemas import CursorSample, CursorTrack

SIZE = 200
FRAME_INTERVAL = 0.5


def frame(*dots, size: int = SIZE):
    """Fond uni avec de petits carrés clairs aux positions données (en pixels)."""
    image = np.zeros((size, size), dtype=np.uint8)
    for x, y in dots:
        image[y - 3 : y + 3, x - 3 : x + 3] = 255
    return image


def run(frames, config):
    return track_cursor(len(frames), frames.__getitem__, config, FRAME_INTERVAL)


@pytest.fixture
def cursor_config(config):
    """Seuils adaptés aux images de synthèse (200x200, curseur de 6x6)."""
    patched = config.model_copy(deep=True)
    patched.cursor.min_area_fraction = 0.0002   # ~8 px
    patched.cursor.max_area_fraction = 0.02     # ~800 px
    patched.cursor.match_tolerance_fraction = 0.05
    return patched


def test_un_pointeur_qui_se_deplace_est_localise(cursor_config):
    """Sa position à l'instant i est la seule commune aux deux écarts."""
    frames = [frame((30, 30)), frame((80, 80)), frame((130, 130))]

    samples = run(frames, cursor_config)

    assert len(samples) == 1
    assert samples[0].x == pytest.approx(80 / SIZE, abs=0.02)
    assert samples[0].y == pytest.approx(80 / SIZE, abs=0.02)
    assert samples[0].timestamp == pytest.approx(FRAME_INTERVAL)


def test_un_pointeur_immobile_ne_produit_aucune_position(cursor_config):
    """Il ne bouge pas, donc il n'apparaît dans aucun écart. C'est assumé :
    `position_at` reconduit la dernière position connue."""
    frames = [frame((80, 80))] * 4

    assert run(frames, cursor_config) == []


def test_la_trajectoire_complete_est_suivie(cursor_config):
    frames = [frame((20, 20)), frame((60, 60)), frame((100, 100)), frame((140, 140))]

    samples = run(frames, cursor_config)

    assert [round(s.timestamp, 2) for s in samples] == [0.5, 1.0]
    assert [round(s.x * SIZE) for s in samples] == [60, 100]


def test_deux_zones_mobiles_au_meme_endroit_ne_donnent_aucune_position(cursor_config):
    """Une animation en parallèle rend l'appariement indécidable : mieux vaut
    ne rien dire qu'une position douteuse."""
    frames = [
        frame((30, 30), (30, 150)),
        frame((80, 80), (80, 150)),
        frame((130, 130), (130, 150)),
    ]

    assert run(frames, cursor_config) == []


def test_une_zone_trop_grande_n_est_pas_un_pointeur(cursor_config):
    """Un changement de page fait tout bouger : ce n'est pas la souris."""
    big = np.zeros((SIZE, SIZE), dtype=np.uint8)
    big[20:180, 20:180] = 255
    frames = [np.zeros((SIZE, SIZE), dtype=np.uint8), big, np.zeros((SIZE, SIZE), dtype=np.uint8)]

    assert run(frames, cursor_config) == []


def test_moins_de_trois_frames_ne_permet_aucune_intersection(cursor_config):
    frames = [frame((30, 30)), frame((80, 80))]

    assert run(frames, cursor_config) == []


# --- reconduction de la dernière position ---------------------------------

def track(*samples) -> CursorTrack:
    return CursorTrack(
        sample_fps=2.0,
        samples=[CursorSample(timestamp=t, x=x, y=y) for t, x, y in samples],
    )


def test_la_derniere_position_connue_est_reconduite():
    """Un pointeur posé sur un bouton reste dessus même s'il ne bouge plus."""
    assert position_at(track((10.0, 0.4, 0.5)), 11.0, hold_seconds=2.0) == (0.4, 0.5)


def test_une_position_trop_ancienne_est_abandonnee():
    assert position_at(track((10.0, 0.4, 0.5)), 20.0, hold_seconds=2.0) is None


def test_la_position_la_plus_recente_l_emporte():
    known = track((10.0, 0.1, 0.1), (12.0, 0.8, 0.8))

    assert position_at(known, 12.5, hold_seconds=2.0) == (0.8, 0.8)


def test_aucune_position_avant_la_premiere_detection():
    assert position_at(track((10.0, 0.4, 0.5)), 5.0, hold_seconds=2.0) is None


def test_une_trajectoire_vide_ne_donne_aucune_position():
    assert position_at(track(), 10.0, hold_seconds=2.0) is None
