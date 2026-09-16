"""Tests des sous-titres incrustés au format ASS, placés dans la bande basse.

Le SRT reste le livrable ; l'ASS n'existe que pour l'incrustation, parce que
c'est le seul format qui place un texte au pixel près sur le canevas.
"""

from __future__ import annotations

import re

import pytest

from frame_layout import Layout, compute_layout
from subtitles import Cue, _ass_timestamp, ass_color, build_ass, build_cues, subtitle_font_size

from test_build_timeline import make_decision
from test_subtitles import make_entry


def dialogues(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def style(ass: str) -> list[str]:
    line = next(line for line in ass.splitlines() if line.startswith("Style: Default"))
    return line.removeprefix("Style: ").split(",")


# --- couleurs et horaires ---------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("white", "&H00FFFFFF"),
        ("0x354a5f", "&H005F4A35"),   # ASS range les composantes en BGR
        ("#F5B800", "&H0000B8F5"),
        ("Yellow", "&H0000FFFF"),
    ],
)
def test_les_couleurs_ffmpeg_sont_traduites_en_notation_ass(value, expected):
    assert ass_color(value) == expected


@pytest.mark.parametrize("value", ["blanc", "0x354a5", "#GGGGGG", ""])
def test_une_couleur_illisible_est_signalee(value):
    with pytest.raises(ValueError, match="illisible"):
        ass_color(value)


def test_l_horodatage_ass_est_au_centieme():
    assert _ass_timestamp(3725.456) == "1:02:05.46"
    assert _ass_timestamp(0) == "0:00:00.00"


# --- placement --------------------------------------------------------------

def test_la_resolution_de_reference_est_celle_du_canevas(config):
    ass = build_ass([], compute_layout(1920, 868, config), config)

    assert "PlayResX: 1920" in ass and "PlayResY: 1080" in ass


def test_dans_la_bande_chaque_replique_est_centree_sur_la_bande(config):
    """Bande basse de 148 px sous une image posée à y=64 : centre à 64+868+74."""
    layout = compute_layout(1920, 868, config)

    ass = build_ass([Cue(1.0, 3.0, ["Click Clear."])], layout, config)

    (line,) = dialogues(ass)
    assert "{\\an5\\pos(960,1006)}Click Clear." in line
    assert style(ass)[16] == "0"  # pas de contour sur un fond uni


def test_sans_bande_les_repliques_reviennent_sur_l_image_avec_un_contour(config):
    layout = compute_layout(1920, 1020, config)  # trop peu de place pour une bande

    ass = build_ass([Cue(1.0, 3.0, ["Click Clear."])], layout, config)

    (line,) = dialogues(ass)
    assert "\\pos" not in line
    fields = style(ass)
    assert fields[16] == "2"   # contour
    assert fields[18] == "2"   # en bas, centré


def test_deux_lignes_sont_separees_par_un_saut_ass(config):
    ass = build_ass(
        [Cue(0.0, 2.0, ["Welcome to this presentation", "of the sales report."])],
        compute_layout(1920, 868, config), config,
    )

    assert dialogues(ass)[0].endswith("Welcome to this presentation\\Nof the sales report.")


def test_les_accolades_du_texte_ne_deviennent_pas_des_balises(config):
    """Une accolade ouvre une balise ASS : le texte qui suit disparaîtrait."""
    ass = build_ass([Cue(0.0, 2.0, ["Use {filter} here"])], compute_layout(1920, 868, config), config)

    assert dialogues(ass)[0].endswith("Use (filter) here")


def test_la_taille_est_reduite_quand_la_bande_ne_porte_pas_deux_lignes(config):
    roomy = compute_layout(1920, 868, config)
    tight = Layout(1920, 1080, 1920, 1000, 0, 0, 0, 80)

    assert subtitle_font_size(roomy, config) == config.layout.subtitle_size
    assert subtitle_font_size(tight, config) < config.layout.subtitle_size
    # Deux lignes à la taille réduite tiennent dans la bande.
    assert 2 * subtitle_font_size(tight, config) * 1.2 <= 80


# --- mêmes répliques que le SRT ---------------------------------------------

def test_l_ass_reprend_exactement_les_repliques_et_horaires_du_srt(config):
    """Le livrable et l'incrustation ne doivent jamais diverger."""
    long_text = (
        "Here we filter by region and category, then clear the filters to see "
        "every order again, and finally search for a customer by name."
    )
    decision = make_decision("seg-001", 0.0, 12.0)
    decision.text_en = long_text
    timeline = {"seg-001": make_entry("seg-001", 2.0, 14.0)}

    cues = build_cues([decision], timeline, config)
    ass = build_ass(cues, compute_layout(1920, 868, config), config)

    assert len(cues) > 1  # texte réparti sur plusieurs répliques
    assert len(dialogues(ass)) == len(cues)
    words = " ".join(
        re.sub(r"\{[^}]*\}", "", line.split(",", 9)[9]).replace("\\N", " ")
        for line in dialogues(ass)
    ).split()
    assert words == long_text.split()
