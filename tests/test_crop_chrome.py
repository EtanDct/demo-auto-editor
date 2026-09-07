"""Tests de la détection du bandeau de navigateur (étape `crop`).

Le détecteur n'encode aucune mise en page : il ne sait pas ce qu'est une barre
de favoris ni quel navigateur est utilisé. Les cas ci-dessous fabriquent donc
les autres configurations à partir d'une même matière, et vérifient que la
frontière suit — c'est la seule garantie que ça marchera ailleurs.
"""

from __future__ import annotations

import numpy as np
import pytest

from crop_chrome import (
    detect_anchor_height,
    detect_chrome_height,
    first_band_row,
    parse_color,
    resolve_crop_top,
)

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


# --- ancrage par la couleur de la barre applicative -----------------------
#
# La variation temporelle ne sait pas séparer deux zones figées empilées : un
# bandeau de navigateur suivi d'un en-tête applicatif qui ne bouge jamais lui
# apparaissent comme une seule bande, et le plafond la fait renoncer. La teinte
# de la barre Fiori, elle, dit exactement où commence l'application.

NAVY = (0x35, 0x4A, 0x5F)  # #354a5f, barre supérieure du thème Fiori Belize


def app_frames(
    chrome_height: int = 80,
    bar_height: int = 44,
    frozen_header: int = 0,
    count: int = 40,
    navy=NAVY,
    jitter: int = 0,
):
    """Bandeau clair figé, puis la barre Fiori, puis la page.

    `frozen_header` insère sous la barre un en-tête applicatif qui ne change
    jamais — le cas qui met la détection temporelle en échec.
    """
    rng = np.random.default_rng(0)
    images = []
    for _ in range(count):
        frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:chrome_height] = 235
        bar = np.array(navy, dtype=np.int16) + (
            rng.integers(-jitter, jitter + 1, (bar_height, WIDTH, 3)) if jitter else 0
        )
        frame[chrome_height : chrome_height + bar_height] = bar.clip(0, 255)
        top = chrome_height + bar_height
        frame[top : top + frozen_header] = 248
        frame[top + frozen_header :] = rng.integers(
            0, 255, (HEIGHT - top - frozen_header, WIDTH, 3), dtype=np.uint8
        )
        images.append(frame)
    return images


def anchor(images, config):
    return detect_anchor_height(images, config.crop)[0]


def test_la_barre_fiori_situe_la_frontiere(config):
    assert anchor(app_frames(chrome_height=80), config) == 80


def test_un_bandeau_plus_fin_est_suivi_par_la_couleur(config):
    """Utilisateur sans barre de favoris : le bandeau est plus court."""
    assert anchor(app_frames(chrome_height=40), config) == 40


def test_un_bandeau_plus_epais_est_suivi_par_la_couleur(config):
    """Navigateur avec une barre d'extensions en plus."""
    assert anchor(app_frames(chrome_height=95), config) == 95


def test_une_capture_sans_navigateur_ne_donne_aucun_recadrage(config):
    """L'application occupe l'écran : la barre commence au bord supérieur."""
    assert anchor(app_frames(chrome_height=0), config) == 0


def test_la_compression_delave_la_teinte_sans_perdre_la_barre(config):
    """La couleur n'arrive jamais exacte : le JPEG et le H.264 la font bouger."""
    assert anchor(app_frames(chrome_height=80, jitter=12), config) == 80


def test_une_teinte_absente_ne_donne_aucune_frontiere(config):
    """Une interface dont on ne connaît pas la couleur — c'est le secours qui
    prend le relais, pas l'ancrage qui invente une valeur."""
    images = app_frames(chrome_height=80, navy=(0xC0, 0xC0, 0xC0))

    height, reason = detect_anchor_height(images, config.crop)

    assert height is None
    assert "354a5f" in reason


def test_un_entete_applicatif_fige_ne_trompe_plus_la_detection(config):
    """Régression : c'est le cas qui a fait échouer la vidéo SAP. Sous le
    bandeau du navigateur, un en-tête applicatif ne bouge jamais non plus ; les
    deux zones figées s'empilent, la variation temporelle place la frontière
    au-delà du plafond, donc renonce. La couleur, elle, tranche."""
    images = app_frames(chrome_height=80, frozen_header=150)

    grey = [f.mean(axis=2) for f in images]
    assert detect_chrome_height(grey, 0.25, 20)[0] == 0  # l'ancienne méthode renonce

    assert resolve_crop_top(config, images)[0] == 80


def test_quelques_pixels_de_la_meme_teinte_ne_font_pas_une_barre(config):
    """Une icône bleu marine dans la barre d'outils du navigateur : la couleur
    est là, la largeur n'y est pas."""
    images = app_frames(chrome_height=80)
    for frame in images:
        frame[20:40, 100:180] = NAVY  # ~13 % de la largeur

    assert anchor(images, config) == 80


def test_un_lisere_trop_fin_ne_fait_pas_une_barre(config):
    """Une bordure pleine largeur de deux pixels n'est pas une barre applicative."""
    images = app_frames(chrome_height=80)
    for frame in images:
        frame[30:32, :] = NAVY

    assert anchor(images, config) == 80


def test_une_barre_vue_sur_trop_peu_d_images_est_refusee(config):
    """Un reflet ou une transition sur trois images ne fixe pas la frontière."""
    images = app_frames(chrome_height=80, navy=(0xC0, 0xC0, 0xC0))
    for frame in images[:3]:
        frame[80:124] = NAVY

    assert anchor(images, config) is None


def test_une_image_de_transition_ne_deplace_pas_la_frontiere(config):
    """La médiane protège des images prises pendant un défilement."""
    images = app_frames(chrome_height=80)
    for frame in images[:6]:
        frame[80:124] = 235          # barre masquée par une transition
        frame[200:244] = NAVY        # et retrouvée plus bas

    assert anchor(images, config) == 80


def test_le_motif_de_l_ancrage_porte_les_chiffres(config):
    _, reason = detect_anchor_height(app_frames(chrome_height=80), config.crop)

    assert "80px" in reason and "354a5f" in reason


def test_une_frontiere_trop_basse_reste_refusee(config):
    """Le plafond vaut aussi pour l'ancrage : une barre à la moitié de l'écran
    n'est pas un bandeau de navigateur, quelle que soit sa couleur."""
    height, reason = resolve_crop_top(config, app_frames(chrome_height=200))

    assert height == 0
    assert "plafond" in reason


def test_l_ancrage_passe_avant_la_variation(config):
    """Les deux méthodes s'accordent ici ; c'est l'ancrage qui doit répondre,
    et sa trace doit le dire."""
    _, reason = resolve_crop_top(config, app_frames(chrome_height=80))

    assert "couleur" in reason


def test_plusieurs_teintes_peuvent_etre_configurees(config):
    """Un autre thème Fiori : la liste s'allonge, le code ne change pas."""
    patched = config.model_copy(deep=True)
    patched.crop.anchor_colors = ["#354a5f", "#1c2228"]
    images = app_frames(chrome_height=60, navy=(0x1C, 0x22, 0x28))

    assert detect_anchor_height(images, patched.crop)[0] == 60


def test_la_teinte_la_plus_haute_l_emporte(config):
    """Deux couleurs configurées présentes toutes les deux : c'est la première
    rencontrée en descendant qui marque le début de l'application."""
    patched = config.model_copy(deep=True)
    patched.crop.anchor_colors = ["#354a5f", "#1c2228"]
    images = app_frames(chrome_height=80)
    for frame in images:
        frame[300:344] = (0x1C, 0x22, 0x28)

    assert detect_anchor_height(images, patched.crop)[0] == 80


# --- lecture des couleurs -------------------------------------------------

@pytest.mark.parametrize("written", ["#354a5f", "354a5f", "#354A5F", "  #354a5f  "])
def test_la_couleur_se_lit_en_hexadecimal(written):
    assert parse_color(written) == (0x35, 0x4A, 0x5F)


@pytest.mark.parametrize("written", ["#354a5", "bleu", "", "#354a5f00"])
def test_une_couleur_illisible_est_signalee_tot(written):
    """Une faute de frappe dans config.yaml doit se voir avant l'encodage."""
    with pytest.raises(ValueError, match="[Cc]ouleur"):
        parse_color(written)


def test_une_image_en_niveaux_de_gris_ne_porte_aucune_teinte():
    """Garde-fou : l'ancrage ne doit pas inventer une frontière sur du gris."""
    grey = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)

    assert first_band_row(grey, [NAVY], 24, 0.6, 6) is None
