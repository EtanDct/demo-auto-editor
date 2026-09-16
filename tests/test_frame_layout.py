"""Tests de la mise en page de livraison : canevas fixe, bandes, habillage.

Le recadrage produit une hauteur qui change d'une capture à l'autre. Les cas
ci-dessous la font varier et vérifient que la mise en page suit, et qu'elle se
dégrade proprement quand la place manque — c'est la garantie que ça marchera
sur une autre vidéo que celle de référence.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from frame_layout import (
    BandContent,
    Layout,
    TimedText,
    build_layout_chains,
    chapter_windows,
    compute_layout,
    echo_windows,
    piece_output_starts,
    progress_y,
    top_band_filters,
)
from render_video import Piece, build_video_filter
from schemas import Chapter, ChapterPlan, VisualAction
from subtitles import Cue, build_ass

from test_build_timeline import make_decision

FFMPEG = shutil.which("ffmpeg")


def layout_of(config, width, height):
    return compute_layout(width, height, config)


# --- placement sur le canevas ---------------------------------------------

def test_la_capture_rognee_de_reference_donne_deux_bandes(config):
    """1920x1020 moins 151 px de navigateur, arrondi au pair par FFmpeg."""
    layout = layout_of(config, 1920, 868)

    assert (layout.width, layout.height) == (1920, 1080)
    assert (layout.top_height, layout.bottom_height) == (64, 148)
    assert (layout.content_x, layout.content_y) == (0, 64)
    assert not layout.scaled


def test_un_bandeau_plus_fin_donne_des_bandes_plus_basses(config):
    """Utilisateur sans barre de favoris : moins rogné, moins à combler."""
    layout = layout_of(config, 1920, 900)

    assert layout.top_height == 64
    assert layout.bottom_height == 1080 - 900 - 64


def test_la_bande_du_haut_cede_d_abord_quand_la_place_manque(config):
    """Les sous-titres portent le propos : ils gardent leur bande."""
    layout = layout_of(config, 1920, 950)  # 130 px libres, moins que 64 + 110

    assert layout.top_height == 0
    assert layout.bottom_height == 130
    assert layout.content_y == 0


def test_sans_place_pour_une_bande_l_image_est_centree(config):
    layout = layout_of(config, 1920, 1020)  # 60 px libres

    assert (layout.top_height, layout.bottom_height) == (0, 0)
    assert layout.content_y == 30
    assert not layout.subtitles_in_band


def test_une_capture_a_la_taille_du_canevas_est_livree_telle_quelle(config):
    layout = layout_of(config, 1920, 1080)

    assert not layout.scaled and not layout.needs_pad
    assert (layout.top_height, layout.bottom_height) == (0, 0)


def test_une_capture_plus_grande_est_reduite_sans_deformation(config):
    layout = layout_of(config, 2560, 1440)

    assert layout.scaled
    assert (layout.content_width, layout.content_height) == (1920, 1080)


def test_une_capture_plus_etroite_est_agrandie_sans_perdre_ses_bandes(config):
    """Agrandie jusqu'à remplir la hauteur, elle chasserait les sous-titres."""
    layout = layout_of(config, 1440, 868)

    assert layout.scaled
    assert (layout.top_height, layout.bottom_height) == (64, config.layout.min_bottom_height)
    assert abs(2 * layout.content_x + layout.content_width - 1920) <= 2  # centrée
    assert layout.content_height == 1080 - 64 - config.layout.min_bottom_height


def test_une_petite_capture_est_agrandie_en_gardant_ses_bandes(config):
    layout = layout_of(config, 1280, 720)

    assert layout.scaled
    assert layout.content_height == 1080 - 64 - config.layout.min_bottom_height
    assert layout.subtitles_in_band


def test_une_capture_reduite_n_est_pas_retrecie_pour_faire_place_aux_bandes(config):
    """Une capture 16:9 trop grande tient exactement dans le canevas : pas de
    bande, plutôt qu'une interface rapetissée pour y loger les sous-titres."""
    layout = layout_of(config, 2560, 1440)

    assert (layout.top_height, layout.bottom_height) == (0, 0)


def test_toutes_les_dimensions_restent_paires(config):
    """Le H.264 en 4:2:0 refuse les dimensions impaires."""
    for width, height in [(1917, 861), (1366, 767), (2559, 1439), (1000, 777)]:
        layout = layout_of(config, width, height)
        for value in (
            layout.width, layout.height, layout.content_width, layout.content_height,
            layout.content_x, layout.content_y, layout.top_height,
        ):
            assert value % 2 == 0, (width, height, layout)


def test_la_mise_en_page_desactivee_livre_la_taille_rognee(config):
    patched = config.model_copy(deep=True)
    patched.layout.enabled = False

    layout = layout_of(patched, 1920, 868)

    assert not layout.scaled and not layout.needs_pad
    assert (layout.width, layout.height) == (1920, 868)


# --- horaires dans la vidéo finale ------------------------------------------

def pieces_with_gap_and_freeze():
    """Intervalle muet de 2 s, segment gelé de 3 s, puis second segment."""
    first = make_decision("seg-001", 2.0, 10.0)
    second = make_decision("seg-002", 10.0, 16.0)
    return [
        Piece(kind="gap", start=0.0, end=2.0),
        Piece(kind="segment", start=2.0, end=10.0, extension=3.0, decision=first),
        Piece(kind="segment", start=10.0, end=16.0, decision=second),
    ]


def test_les_debuts_de_morceaux_comptent_les_gels_de_plan():
    assert piece_output_starts(pieces_with_gap_and_freeze()) == [0.0, 2.0, 13.0]


def test_un_chapitre_commence_avec_le_plan_de_sa_premiere_phrase():
    """Le gel de plan du chapitre précédent décale le suivant d'autant."""
    plan = ChapterPlan(chapters=[
        Chapter(title="Key figures", first_segment="seg-001"),
        Chapter(title="Filters", first_segment="seg-002"),
    ])

    windows = chapter_windows(plan, pieces_with_gap_and_freeze(), total_duration=19.0)

    assert [(w.text, w.index, w.start, w.end) for w in windows] == [
        ("Key figures", "1/2", 0.0, 13.0),
        ("Filters", "2/2", 13.0, 19.0),
    ]


def test_le_premier_chapitre_couvre_l_intervalle_muet_du_debut():
    plan = ChapterPlan(chapters=[Chapter(title="Filters", first_segment="seg-002"),
                                 Chapter(title="Search", first_segment="seg-001")])

    windows = chapter_windows(plan, pieces_with_gap_and_freeze(), 19.0)

    assert windows[0].start == 0.0


def test_un_chapitre_sur_une_phrase_absente_de_la_timeline_est_ignore():
    plan = ChapterPlan(chapters=[
        Chapter(title="Key figures", first_segment="seg-001"),
        Chapter(title="Ghost", first_segment="seg-099"),
    ])

    windows = chapter_windows(plan, pieces_with_gap_and_freeze(), 19.0)

    assert [w.text for w in windows] == ["Key figures"]
    assert windows[0].end == 19.0


def test_sans_chapitres_aucune_fenetre():
    assert chapter_windows(None, pieces_with_gap_and_freeze(), 19.0) == []
    assert chapter_windows(ChapterPlan(), pieces_with_gap_and_freeze(), 19.0) == []


def test_le_rappel_suit_la_fenetre_du_cadre_dans_la_video_finale():
    pieces = pieces_with_gap_and_freeze()
    pieces[2].decision.visual_action = VisualAction(
        type="highlight", target="Clear", x=0.9, y=0.4, width=0.03, height=0.03,
        start_offset=1.0, end_offset=2.5,
    )

    (echo,) = echo_windows(pieces)

    assert (echo.text, echo.start, echo.end) == ("Clear", 14.0, 15.5)


def test_un_cadre_sans_fin_est_rappele_jusqu_a_la_fin_du_plan_hors_gel():
    """Le gel de plan prolonge l'image, pas la désignation."""
    pieces = pieces_with_gap_and_freeze()
    pieces[1].decision.visual_action = VisualAction(
        type="highlight", target="Revenue", x=0.1, y=0.1, width=0.1, height=0.1
    )

    (echo,) = echo_windows(pieces)

    assert (echo.start, echo.end) == (2.0, 10.0)


def test_seuls_les_segments_encadres_sont_rappeles():
    assert echo_windows(pieces_with_gap_and_freeze()) == []


# --- graphe de filtres ------------------------------------------------------

def band(total=19.0, **kwargs):
    return BandContent(total_duration=total, **kwargs)


def test_l_image_est_posee_sur_le_canevas_a_la_hauteur_de_la_bande(config):
    layout = layout_of(config, 1920, 868)

    chains, label = build_layout_chains("vpadded", layout, band(), config, 30.0, "subs.ass")

    assert label == "[vout]"
    assert f"pad=1920:1080:0:64:color={config.layout.background}" in chains[0]
    assert chains[-1].startswith("[vprogress]subtitles=")


def test_les_cadres_sont_dessines_avant_la_mise_en_page(config, tmp_path):
    """Régression à éviter : dessinés après, les cadres tomberaient 64 px trop
    haut, décalés de la hauteur de la bande."""
    decision = make_decision("seg-001", 0.0, 10.0)
    decision.visual_action = VisualAction(
        type="highlight", target="Clear", x=0.5, y=0.5, width=0.1, height=0.05
    )
    pieces = [Piece(kind="segment", start=0.0, end=10.0, decision=decision)]
    layout = layout_of(config, 1920, 868)

    graph, _ = build_video_filter(
        pieces, config, 1920, 868, tmp_path / "subs.ass", 30.0, layout=layout,
        band=band(10.0),
    )

    assert graph.index("drawbox=x=(iw*0.5)") < graph.index("pad=1920:1080")


def test_sans_mise_en_page_ni_pad_ni_filet(config):
    layout = Layout(1920, 1080, 1920, 1080, 0, 0, 0, 0)

    chains, _ = build_layout_chains("vpadded", layout, band(), config, 30.0, "subs.ass")

    assert chains == ["[vpadded]subtitles='subs.ass'[vout]"]


def test_une_image_reduite_est_mise_a_l_echelle_avant_d_etre_posee(config):
    layout = layout_of(config, 2560, 1440)

    chains, _ = build_layout_chains("vpadded", layout, band(), config, 30.0, "subs.ass")

    assert "scale=1920:1080,pad=" not in chains[0]  # rien à combler : pas de pad
    assert chains[0].startswith("[vpadded]scale=1920:1080")


def test_le_filet_separe_la_bande_du_haut_de_l_application(config):
    assert progress_y(layout_of(config, 1920, 868), config) == 64 - config.layout.progress_thickness


def test_sans_bande_du_haut_le_filet_borde_le_bas(config):
    layout = layout_of(config, 1920, 950)

    assert progress_y(layout, config) == 1080 - config.layout.progress_thickness


def test_sans_bande_aucun_filet(config):
    assert progress_y(layout_of(config, 1920, 1020), config) is None


def test_le_filet_avance_par_un_overlay_et_non_un_drawbox(config):
    """`drawbox` n'évalue pas sa largeur image par image : une barre en drawbox
    est pleine dès la première image (vérifié sur FFmpeg 9)."""
    chains, _ = build_layout_chains(
        "vpadded", layout_of(config, 1920, 868), band(total=102.4), config, 30.0, "subs.ass"
    )
    graph = ";".join(chains)

    assert "overlay=x='-w+w*t/102.400'" in graph
    assert "w=iw*t" not in graph


def test_chaque_changement_de_chapitre_marque_le_filet(config):
    chapters = [TimedText("A", 0.0, 25.0, "1/3"), TimedText("B", 25.0, 50.0, "2/3"),
                TimedText("C", 50.0, 100.0, "3/3")]
    chains, _ = build_layout_chains(
        "vpadded", layout_of(config, 1920, 868), band(total=100.0, chapters=chapters),
        config, 30.0, "subs.ass",
    )
    graph = ";".join(chains)

    y = 64 - config.layout.progress_thickness
    assert f"drawbox=x={480 - 2}:y={y}:w=4" in graph
    assert f"drawbox=x={960 - 2}:y={y}:w=4" in graph
    assert graph.count(":w=4:") == 2  # pas de marque au début


def test_le_logo_entre_par_son_index_et_s_aligne_a_droite(config):
    chains, label = build_layout_chains(
        "vpadded", layout_of(config, 1920, 868), band(), config, 30.0, "subs.ass", logo_input=16
    )
    graph = ";".join(chains)

    assert f"[16:v]scale=-2:{config.layout.logo_height}[logo]" in graph
    assert "overlay=x=W-w-" in graph
    assert label == "[vout]"


def test_sans_bande_du_haut_le_logo_n_est_pas_pose(config):
    chains, _ = build_layout_chains(
        "vpadded", layout_of(config, 1920, 950), band(), config, 30.0, "subs.ass", logo_input=16
    )

    assert "[16:v]" not in ";".join(chains)


def test_la_bande_affiche_le_titre_quand_il_n_y_a_pas_de_chapitres(config):
    fragments = top_band_filters(
        layout_of(config, 1920, 868), band(title="Sales Report Overview"), config
    )

    assert len(fragments) == 1
    assert "between(t,0.000,19.000)" in fragments[0]


def test_un_chapitre_affiche_son_index_et_son_titre_en_colonnes(config):
    fragments = top_band_filters(
        layout_of(config, 1920, 868),
        band(chapters=[TimedText("Filters", 0.0, 19.0, "1/1")], title="ignored"),
        config,
    )

    assert len(fragments) == 2
    assert f"fontcolor={config.layout.muted_color}" in fragments[0]
    assert "y_align=font" in fragments[1]


def test_le_rappel_est_une_pastille_centree_aux_couleurs_des_cadres(config):
    fragments = top_band_filters(
        layout_of(config, 1920, 868), band(echoes=[TimedText("Clear", 4.0, 6.0)]), config
    )

    (echo,) = fragments
    assert "x=(w-text_w)/2" in echo
    assert f"boxcolor={config.layout.accent_color}" in echo
    assert "between(t,4.000,6.000)" in echo


def test_le_rappel_peut_etre_desactive(config):
    patched = config.model_copy(deep=True)
    patched.layout.echo_enabled = False

    fragments = top_band_filters(
        layout_of(patched, 1920, 868), band(echoes=[TimedText("Clear", 4.0, 6.0)]), patched
    )

    assert fragments == []


# --- FFmpeg réel ------------------------------------------------------------

@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg absent du PATH")
def test_ffmpeg_produit_le_canevas_et_ses_bandes(config, tmp_path):
    """Le graphe complet passe dans FFmpeg, et chaque zone a la bonne couleur."""
    patched = config.model_copy(deep=True)
    settings = patched.layout
    settings.width, settings.height = 320, 240
    settings.top_height, settings.min_bottom_height = 24, 30
    settings.top_text_size, settings.subtitle_size = 12, 14
    settings.band_padding, settings.progress_thickness = 8, 2

    layout = compute_layout(320, 180, patched)
    assert (layout.top_height, layout.bottom_height) == (24, 36)

    subs = tmp_path / "subs.ass"
    subs.write_text(build_ass([Cue(0.0, 2.0, ["Click Clear."])], layout, patched), encoding="utf-8")
    content = band(
        total=2.0,
        chapters=[TimedText("Filters", 0.0, 1.0, "1/2"), TimedText("Search", 1.0, 2.0, "2/2")],
        echoes=[TimedText("Clear", 0.5, 1.5)],
    )
    chains, label = build_layout_chains("0:v", layout, content, patched, 10.0, subs)
    frame = tmp_path / "frame.png"

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=white:s=320x180:r=10:d=2",
         "-filter_complex", ";".join(chains), "-map", label,
         "-frames:v", "1", str(frame)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-400:]

    import cv2

    image = cv2.imread(str(frame))
    assert image.shape[:2] == (240, 320)
    navy_bgr = np.array([0x5F, 0x4A, 0x35])
    assert np.abs(image[4, 316].astype(int) - navy_bgr).max() <= 6       # bande du haut
    assert image[24 + 90, 160].min() >= 245                              # image posée
    assert np.abs(image[236, 4].astype(int) - navy_bgr).max() <= 6       # bande du bas


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg absent du PATH")
def test_ffmpeg_pose_le_logo_a_droite_de_la_bande_du_haut(config, tmp_path):
    import cv2

    patched = config.model_copy(deep=True)
    settings = patched.layout
    settings.width, settings.height = 320, 240
    settings.top_height, settings.min_bottom_height = 24, 30
    settings.band_padding, settings.logo_height, settings.progress_enabled = 8, 16, False

    logo = tmp_path / "logo.png"
    cv2.imwrite(str(logo), np.full((40, 40, 3), (0, 0, 255), dtype=np.uint8))  # carré rouge
    subs = tmp_path / "subs.ass"
    layout = compute_layout(320, 180, patched)
    subs.write_text(build_ass([], layout, patched), encoding="utf-8")
    chains, label = build_layout_chains("0:v", layout, band(total=1.0), patched, 10.0, subs, logo_input=1)
    frame = tmp_path / "frame.png"

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=white:s=320x180:r=10:d=1", "-i", str(logo),
         "-filter_complex", ";".join(chains), "-map", label, "-frames:v", "1", str(frame)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-400:]

    image = cv2.imread(str(frame))
    red = (image[:, :, 2] > 200) & (image[:, :, 1] < 60)
    ys, xs = np.nonzero(red)
    # 16 px de haut, centré dans la bande de 24 px, à 8 px du bord droit.
    assert (ys.min(), ys.max()) == (4, 19)
    assert xs.max() == 320 - 8 - 1
