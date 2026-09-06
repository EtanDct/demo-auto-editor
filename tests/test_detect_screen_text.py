"""Tests de l'index du texte à l'écran (étape `screen`).

L'OCR lui-même n'est pas testé ici (c'est un modèle tiers) : on teste le
regroupement des détections en éléments stables, qui est la partie où une
erreur passe inaperçue et fausse silencieusement tout l'appariement en aval.
"""

from __future__ import annotations

import pytest

from detect_screen_text import (
    Detection,
    changed_pixel_ratio,
    expand_to_frames,
    group_into_elements,
    is_usable_label,
    load_detections,
    select_frames_to_analyse,
    write_detections,
)
from schemas import BoundingBox, ScreenElement
from ui_reference import match_key, normalize_text

FRAME_INTERVAL = 0.5


def box(x: float, y: float, width: float = 0.1, height: float = 0.05) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def detection(text: str, x: float, y: float, timestamp: float, confidence: float = 0.9) -> Detection:
    return Detection(text=text, box=box(x, y), confidence=confidence, timestamp=timestamp)


def group(frames, merge_iou: float = 0.6, max_gap_seconds: float = 1.0):
    return group_into_elements(frames, merge_iou, max_gap_seconds, FRAME_INTERVAL)


def test_normalisation_ignore_casse_accents_et_espaces():
    assert normalize_text("  Créer   un  Poste ") == "creer un poste"
    assert normalize_text("MENU") == normalize_text("menu")


def test_un_element_stable_sur_plusieurs_frames_devient_une_seule_entree():
    frames = [[detection("Post", 0.2, 0.3, t / 2)] for t in range(6)]

    (element,) = group(frames)

    assert element.text == "Post"
    assert element.occurrences == 6
    assert (element.first_seen, element.last_seen) == (0.0, 2.5)


def test_le_meme_libelle_a_deux_endroits_reste_deux_elements():
    """C'est le cas qui doit rester ambigu : encadrer le mauvais des deux est
    pire que ne rien encadrer, donc l'aval doit pouvoir voir les deux."""
    frames = [[detection("Menu", 0.1, 0.2, 0.0), detection("Menu", 0.7, 0.8, 0.0)]]

    elements = group(frames)

    assert len(elements) == 2
    assert {e.text for e in elements} == {"Menu"}
    assert {round(e.box.x, 2) for e in elements} == {0.1, 0.7}


def test_un_element_qui_se_deplace_franchement_est_un_nouvel_element():
    frames = [[detection("Post", 0.1, 0.1, 0.0)], [detection("Post", 0.8, 0.8, 0.5)]]

    elements = group(frames)

    assert len(elements) == 2


def test_un_leger_tremblement_de_boite_ne_casse_pas_l_element():
    frames = [
        [detection("Post", 0.100, 0.200, 0.0)],
        [detection("Post", 0.104, 0.202, 0.5)],
    ]

    (element,) = group(frames)

    assert element.occurrences == 2


def test_un_element_qui_disparait_puis_revient_apres_le_delai_est_dedouble():
    frames = [
        [detection("Post", 0.1, 0.2, 0.0)],
        [detection("Post", 0.1, 0.2, 5.0)],
    ]

    elements = group(frames, max_gap_seconds=1.0)

    assert len(elements) == 2


def test_un_element_masque_brievement_reste_le_meme():
    frames = [
        [detection("Post", 0.1, 0.2, 0.0)],
        [],  # frame sans détection : ignorée, pas de rupture de continuité
        [detection("Post", 0.1, 0.2, 0.5)],
    ]

    (element,) = group(frames)

    assert element.occurrences == 2


def test_la_graphie_la_mieux_reconnue_est_retenue():
    """Casse et accents varient d'une frame à l'autre au gré du rendu ; on
    garde la graphie la mieux reconnue, c'est elle qu'on comparera à ce
    qu'annonce le narrateur."""
    frames = [
        [detection("POST", 0.1, 0.2, 0.0, confidence=0.62)],
        [detection("Post", 0.1, 0.2, 0.5, confidence=0.98)],
    ]

    (element,) = group(frames)

    assert element.text == "Post"
    assert element.confidence == pytest.approx((0.62 + 0.98) / 2)


def test_un_caractere_mal_lu_scinde_l_element():
    """Limite connue : le regroupement exige un texte normalisé identique, donc
    une coquille OCR ('P0st') crée un élément parasite. Ces éléments sont peu
    fréquents et se distinguent par leur faible nombre d'occurrences ; c'est à
    l'appariement de les écarter, pas au regroupement de les deviner."""
    frames = [
        [detection("Post", 0.1, 0.2, 0.0, confidence=0.98)],
        [detection("P0st", 0.1, 0.2, 0.5, confidence=0.62)],
        [detection("Post", 0.1, 0.2, 1.0, confidence=0.97)],
    ]

    elements = group(frames)

    assert len(elements) == 2
    assert sorted(e.occurrences for e in elements) == [1, 2]


def test_un_element_vu_sur_une_seule_frame_a_une_plage_non_vide():
    """Sinon il ne serait visible pendant aucun segment et serait inexploitable."""
    frames = [[detection("Post", 0.1, 0.2, 3.0)]]

    (element,) = group(frames)

    assert element.last_seen == pytest.approx(3.0 + FRAME_INTERVAL)
    assert element.visible_at(3.0, 3.5) == pytest.approx(1.0)


def test_les_elements_sont_ordonnes_par_apparition():
    frames = [
        [detection("Deux", 0.5, 0.5, 0.0)],
        [detection("Un", 0.1, 0.1, 0.5)],
    ]

    elements = group(frames)

    assert [e.id for e in elements] == ["scr-0001", "scr-0002"]
    assert [e.text for e in elements] == ["Deux", "Un"]


def test_deux_detections_d_une_meme_frame_ne_prolongent_pas_le_meme_element():
    """Sans ce garde-fou, un libellé dupliqué à l'écran collapse en un élément
    dont la boîte saute d'un endroit à l'autre au fil des frames."""
    frames = [
        [detection("Menu", 0.10, 0.20, 0.0)],
        [detection("Menu", 0.10, 0.20, 0.5), detection("Menu", 0.11, 0.21, 0.5)],
    ]

    elements = group(frames)

    assert len(elements) == 2
    assert sorted(e.occurrences for e in elements) == [1, 2]


def test_visible_at_mesure_le_recouvrement_avec_un_segment():
    element = ScreenElement(
        id="scr-0001", text="Post", box=box(0.1, 0.2),
        first_seen=10.0, last_seen=14.0, confidence=0.9, occurrences=8,
    )

    assert element.visible_at(10.0, 14.0) == pytest.approx(1.0)
    assert element.visible_at(12.0, 16.0) == pytest.approx(0.5)
    assert element.visible_at(20.0, 24.0) == 0.0
    assert element.visible_at(5.0, 5.0) == 0.0


def test_iou_de_boites_disjointes_est_nul():
    assert box(0.0, 0.0).iou(box(0.5, 0.5)) == 0.0


def test_iou_de_boites_identiques_vaut_un():
    assert box(0.1, 0.2).iou(box(0.1, 0.2)) == pytest.approx(1.0)


def test_les_pictogrammes_isoles_sont_ecartes():
    """L'OCR rend les icônes d'interface en symboles isolés : ils ne seront
    jamais le libellé prononcé par un narrateur et polluent l'appariement."""
    for noise in ("口", "←", "★", "N", "+", " ", ""):
        assert not is_usable_label(noise, min_length=2)


def test_les_libelles_courts_mais_reels_sont_conserves():
    for label in ("OK", "Go", "18", "Pull requests"):
        assert is_usable_label(label, min_length=2)


def test_le_changement_se_mesure_en_proportion_de_pixels_touches():
    """Un changement localisé (un menu qui s'ouvre) doit être détecté, alors que
    son écart moyen sur toute l'image reste noyé dans le bruit de compression."""
    import numpy as np

    base = np.zeros((200, 200), dtype=np.uint8)
    localised = base.copy()
    localised[0:20, 0:20] = 255  # 1 % de l'image, franchement modifié

    ratio = changed_pixel_ratio(base, localised, pixel_delta=25)

    assert ratio == pytest.approx(0.01)
    assert float(np.mean(np.abs(localised.astype(int) - base.astype(int)))) < 3


def test_le_bruit_de_compression_ne_declenche_pas_l_ocr():
    import numpy as np

    rng = np.random.default_rng(0)
    base = np.full((200, 200), 128, dtype=np.uint8)
    noisy = np.clip(base + rng.integers(-8, 9, base.shape), 0, 255).astype(np.uint8)

    assert changed_pixel_ratio(base, noisy, pixel_delta=25) == 0.0


def test_le_cache_ocr_permet_de_rejouer_le_regroupement_sans_ocr(tmp_path):
    """L'OCR coûte des minutes, le regroupement se règle par essais : le cache
    doit restituer exactement la séquence frame par frame, frames héritées
    comprises."""
    analysed = [
        (0.0, [detection("Post", 0.1, 0.2, 0.0)]),
        (1.0, [detection("Draft", 0.3, 0.4, 1.0)]),
    ]

    cache = tmp_path / "screen_detections.json"
    write_detections(analysed, frames_sampled=4, out_path=cache)
    frames, analysed_count = load_detections(cache, frame_interval=FRAME_INTERVAL)

    assert analysed_count == 2
    assert [d[0].text for d in frames] == ["Post", "Post", "Draft", "Draft"]
    # Chaque frame porte son propre horodatage, pas celui de la frame analysée.
    assert [d[0].timestamp for d in frames] == [0.0, 0.5, 1.0, 1.5]


def test_le_regroupement_depuis_le_cache_donne_le_meme_resultat(tmp_path):
    analysed = [(0.0, [detection("Post", 0.1, 0.2, 0.0)])]
    cache = tmp_path / "screen_detections.json"
    write_detections(analysed, frames_sampled=3, out_path=cache)

    frames, _ = load_detections(cache, frame_interval=FRAME_INTERVAL)
    (element,) = group(frames)

    assert element.occurrences == 3
    assert (element.first_seen, element.last_seen) == (0.0, 1.0)


def test_une_frame_analysee_sans_detection_est_conservee(tmp_path):
    """Un écran vidé de son texte doit bien vider l'index, pas hériter du précédent."""
    analysed = [(0.0, [detection("Post", 0.1, 0.2, 0.0)]), (1.0, [])]
    cache = tmp_path / "screen_detections.json"
    write_detections(analysed, frames_sampled=4, out_path=cache)

    frames, _ = load_detections(cache, frame_interval=FRAME_INTERVAL)

    assert [len(f) for f in frames] == [1, 1, 0, 0]


def test_cache_absent_donne_une_erreur_actionnable(tmp_path):
    with pytest.raises(FileNotFoundError, match="--step screen"):
        load_detections(tmp_path / "absent.json", frame_interval=FRAME_INTERVAL)


def test_les_espaces_manquants_de_l_ocr_ne_scindent_pas_l_element():
    """Le crénage fait osciller l'OCR entre 'Top repositories' et
    'Toprepositories' sur l'extrait de référence."""
    assert match_key("Top repositories") == match_key("Toprepositories")

    frames = [
        [detection("Top repositories", 0.1, 0.2, 0.0, confidence=0.95)],
        [detection("Toprepositories", 0.1, 0.2, 0.5, confidence=0.88)],
    ]

    (element,) = group(frames)

    assert element.occurrences == 2
    assert element.text == "Top repositories"


def test_deux_libelles_reellement_differents_ne_fusionnent_pas():
    assert match_key("Pull requests") != match_key("Pull request")


def _gray(value: int, changed_fraction: float = 0.0):
    """Image de test : `changed_fraction` de la surface portée à 255."""
    import numpy as np

    img = np.full((100, 100), value, dtype=np.uint8)
    rows = int(round(changed_fraction * 100))
    if rows:
        img[:rows, :] = 255
    return img


def test_seules_les_frames_qui_changent_sont_analysees():
    frames = [_gray(0), _gray(0), _gray(0, 0.5), _gray(0, 0.5), _gray(0, 0.9)]

    selected, threshold = select_frames_to_analyse(
        len(frames), frames.__getitem__, pixel_delta=25, change_ratio=0.05, max_frames=100
    )

    assert selected == [0, 2, 4]
    assert threshold == 0.05


def test_la_premiere_frame_est_toujours_analysee():
    frames = [_gray(0)] * 5

    selected, _ = select_frames_to_analyse(
        len(frames), frames.__getitem__, pixel_delta=25, change_ratio=0.05, max_frames=100
    )

    assert selected == [0]


def test_un_defilement_lent_finit_par_declencher_un_ocr():
    """Chaque pas est sous le seuil, mais l'écart cumulé à la dernière frame
    analysée finit par le franchir : la comparaison ne doit donc pas porter sur
    la frame précédente."""
    frames = [_gray(0, f / 100) for f in range(0, 40, 4)]

    selected, _ = select_frames_to_analyse(
        len(frames), frames.__getitem__, pixel_delta=25, change_ratio=0.1, max_frames=100
    )

    assert len(selected) > 1


def test_une_source_tres_animee_reste_sous_le_plafond_de_cout():
    """Sans plafond, une vidéo au mouvement permanent enverrait toutes ses
    frames à l'OCR : des dizaines de minutes au lieu de quelques-unes."""
    frames = [_gray(0, (f % 2) * 0.9) for f in range(60)]

    selected, threshold = select_frames_to_analyse(
        len(frames), frames.__getitem__, pixel_delta=25, change_ratio=0.001, max_frames=10
    )

    assert len(selected) <= 10
    assert threshold > 0.001  # le seuil a bien été relevé, pas la liste tronquée à l'aveugle


def test_le_plafond_ne_penalise_pas_une_source_calme():
    frames = [_gray(0), _gray(0, 0.5)]

    selected, threshold = select_frames_to_analyse(
        len(frames), frames.__getitem__, pixel_delta=25, change_ratio=0.05, max_frames=10
    )

    assert selected == [0, 1]
    assert threshold == 0.05


def test_les_frames_non_analysees_heritent_de_la_precedente():
    analysed = [(0.0, [detection("Post", 0.1, 0.2, 0.0)]), (1.0, [detection("Draft", 0.3, 0.4, 1.0)])]

    frames = expand_to_frames(analysed, frames_sampled=4, frame_interval=FRAME_INTERVAL)

    assert [f[0].text for f in frames] == ["Post", "Post", "Draft", "Draft"]
    assert [f[0].timestamp for f in frames] == [0.0, 0.5, 1.0, 1.5]
