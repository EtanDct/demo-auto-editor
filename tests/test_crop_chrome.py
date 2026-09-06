"""Tests de la détection du bandeau de navigateur (étape `crop`).

Le détecteur n'encode aucune mise en page : il ne sait pas ce qu'est une barre
de favoris ni quel navigateur est utilisé. Les cas ci-dessous fabriquent donc
les autres configurations à partir d'une même matière, et vérifient que la
frontière suit — c'est la seule garantie que ça marchera ailleurs.
"""

from __future__ import annotations

import numpy as np
import pytest

from crop_chrome import detect_chrome_height, resolve_crop_top

HEIGHT, WIDTH = 400, 600


def frames(chrome_height: int, count: int = 40):
    """Images où le haut est figé et le bas change à chaque image."""
    rng = np.random.default_rng(0)
    chrome = rng.integers(0, 255, (chrome_height, WIDTH), dtype=np.uint8)
    return [
        np.vstack([chrome, rng.integers(0, 255, (HEIGHT - chrome_height, WIDTH), dtype=np.uint8)])
        for _ in range(count)
    ]


def detect(images, max_fraction: float = 0.25, run_rows: int = 20) -> int:
    return detect_chrome_height(images, max_fraction, run_rows)[0]


def test_un_bandeau_fige_est_situe_a_sa_hauteur_exacte():
    assert detect(frames(80)) == 80


def test_un_bandeau_plus_fin_est_suivi():
    """L'utilisateur n'a pas de barre de favoris : le bandeau est plus court."""
    assert detect(frames(40)) == 40


def test_un_bandeau_plus_epais_est_suivi():
    """Un navigateur avec une barre d'extensions en plus."""
    assert detect(frames(95)) == 95


def test_une_capture_plein_ecran_ne_donne_aucun_recadrage():
    """Aucun bandeau : la page commence au bord supérieur."""
    assert detect(frames(0)) == 0


def test_un_entete_applicatif_trop_haut_est_refuse():
    """Une bande figée qui occupe la moitié de l'écran n'est pas un navigateur :
    c'est plus probablement un en-tête applicatif, et la barre supérieure de
    SAP Fiori fait partie du produit montré."""
    assert detect(frames(200)) == 0


def test_le_motif_du_refus_est_explicite():
    _, reason = detect_chrome_height(frames(200), max_fraction=0.25, run_rows=20)

    assert "plafond" in reason


def test_une_image_entierement_figee_ne_donne_rien():
    """Rien ne distingue le bandeau du contenu."""
    still = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)

    assert detect([still] * 20) == 0


def test_le_motif_de_la_detection_porte_les_chiffres():
    _, reason = detect_chrome_height(frames(80), max_fraction=0.25, run_rows=20)

    assert "80px" in reason and "variation" in reason


# --- réglage --------------------------------------------------------------

def test_le_reglage_off_desactive_le_recadrage(config):
    patched = config.model_copy(deep=True)
    patched.crop.top = "off"

    height, reason = resolve_crop_top(patched, frames(80))

    assert height == 0
    assert "désactivé" in reason


def test_une_hauteur_imposee_court_circuite_la_detection(config):
    """Sur une série tournée dans les mêmes conditions, on mesure une fois."""
    patched = config.model_copy(deep=True)
    patched.crop.top = 120

    height, reason = resolve_crop_top(patched, frames(80))

    assert height == 120
    assert "imposée" in reason


def test_le_mode_auto_mesure(config):
    patched = config.model_copy(deep=True)
    patched.crop.top = "auto"

    assert resolve_crop_top(patched, frames(80))[0] == 80
