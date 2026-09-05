"""Tests des fragments de filtres d'incrustation (étape F).

Ces filtres n'ont encore jamais tourné sur une vraie vidéo (aucun
`visual_action` renseigné à ce jour) : les tests figent au moins le contrat
— coordonnées normalisées traduites en expressions relatives à l'image, et
échappement des libellés pour le parseur de filtres FFmpeg.
"""

from __future__ import annotations

import pytest

from overlays import overlay_filter_for
from schemas import VisualAction


def make_action(action_type: str, target: str = "Post", **kwargs) -> VisualAction:
    return VisualAction(
        type=action_type,
        target=target,
        **{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, **kwargs},
    )


def config_with_font(config):
    """Config avec une police explicite : requise par drawtext sous Windows."""
    patched = config.model_copy(deep=True)
    patched.overlays.font_path = "C:/Windows/Fonts/arial.ttf"
    return patched


def test_aucune_action_ne_produit_aucun_filtre(config):
    assert overlay_filter_for(None, config, 1920, 1080) is None


def test_highlight_utilise_des_coordonnees_relatives_a_l_image(config):
    """Le filtre doit rester valide quelle que soit la résolution de sortie."""
    result = overlay_filter_for(make_action("highlight"), config, 1920, 1080)

    assert "drawbox=" in result
    assert "iw*0.1" in result and "ih*0.2" in result
    assert "1920" not in result and "1080" not in result


def test_drawbox_n_utilise_jamais_main_w(config):
    """Régression : `main_w` n'existe pas dans drawbox (il y désigne la boîte,
    pas l'image). FFmpeg répond "Undefined constant" et tout le rendu tombe."""
    for action_type in ("highlight", "callout", "cursor_emphasis"):
        result = overlay_filter_for(make_action(action_type), config_with_font(config), 1920, 1080)
        for fragment in result.split(","):
            if fragment.startswith("drawbox="):
                assert "main_w" not in fragment and "main_h" not in fragment


def test_zoom_recadre_puis_reetire_a_la_resolution_de_sortie(config):
    result = overlay_filter_for(make_action("zoom"), config, 1920, 1080)

    assert result.startswith("crop=")
    assert result.endswith("scale=1920:1080")


def test_callout_combine_un_cadre_et_un_libelle(config):
    result = overlay_filter_for(
        make_action("callout", target="Create"), config_with_font(config), 1920, 1080
    )

    assert "drawbox=" in result and "drawtext=" in result
    assert "text='Create'" in result


def test_le_libelle_est_echappe_pour_le_parseur_ffmpeg(config):
    """Un ':' ou une apostrophe non échappés cassent tout le filter_complex."""
    result = overlay_filter_for(
        make_action("popup", target="Note : l'écran"), config_with_font(config), 1920, 1080
    )

    assert r"Note \: l\'écran" in result


def test_type_d_action_inconnu_leve_une_erreur(config):
    action = make_action("highlight")
    object.__setattr__(action, "type", "wipe")

    with pytest.raises(ValueError, match="Type d'effet inconnu"):
        overlay_filter_for(action, config, 1920, 1080)


@pytest.mark.parametrize("action_type", ["zoom", "highlight", "callout", "popup", "cursor_emphasis"])
def test_tous_les_types_declares_produisent_un_filtre(action_type, config):
    assert overlay_filter_for(make_action(action_type), config_with_font(config), 1920, 1080)


def test_coordonnees_hors_image_refusees_par_le_schema():
    with pytest.raises(ValueError):
        make_action("highlight", x=1.5)


def test_largeur_nulle_refusee_par_le_schema():
    with pytest.raises(ValueError):
        make_action("highlight", width=0.0)


def test_effet_texte_sans_police_refuse_avant_le_rendu(config, monkeypatch):
    """Sans fontfile, drawtext échoue sous Windows (pas de fontconfig) : il
    vaut mieux échouer ici qu'après plusieurs minutes d'encodage."""
    monkeypatch.setattr("overlays.platform.system", lambda: "Windows")
    assert config.overlays.font_path is None

    with pytest.raises(ValueError, match="overlays.font_path"):
        overlay_filter_for(make_action("callout"), config, 1920, 1080)


def test_effet_sans_texte_ne_reclame_aucune_police(config, monkeypatch):
    monkeypatch.setattr("overlays.platform.system", lambda: "Windows")

    assert overlay_filter_for(make_action("highlight"), config, 1920, 1080)
    assert overlay_filter_for(make_action("zoom"), config, 1920, 1080)
