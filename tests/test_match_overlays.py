"""Tests de l'appariement narrateur / écran (étape `match`).

Ce sont les règles de refus qui comptent : le système est réglé pour la
précision, donc l'essentiel du travail consiste à ne rien proposer. Chaque
motif de refus est testé séparément, sinon un refus pour la mauvaise raison
passerait pour un bon comportement.
"""

from __future__ import annotations

import pytest

from match_overlays import apply_to_edl, gather_candidates, judge, match, score_candidate
from schemas import (
    BoundingBox,
    EditDecision,
    NarrationSpec,
    ScreenElement,
    ScreenTextIndex,
    UiReference,
)


def element(text: str, first_seen: float = 0.0, last_seen: float = 100.0, **box) -> ScreenElement:
    return ScreenElement(
        id=f"scr-{abs(hash(text)) % 10000:04d}",
        text=text,
        box=BoundingBox(**{"x": 0.2, "y": 0.3, "width": 0.08, "height": 0.03, **box}),
        first_seen=first_seen,
        last_seen=last_seen,
        confidence=0.95,
        occurrences=10,
    )


def decision(label: str | None, start: float = 10.0, end: float = 20.0, kind: str = "named_control"):
    return EditDecision(
        id="seg-001",
        source_start=start,
        source_end=end,
        text_fr="texte",
        text_en="text",
        ui_reference=UiReference(kind=kind, label=label) if kind else None,
        narration=NarrationSpec(voice="en_US-amy-medium"),
    )


# --- notation -------------------------------------------------------------

def test_un_libelle_identique_obtient_le_score_maximal():
    assert score_candidate("Pull requests", "Pull requests") == 1.0


def test_la_notation_ignore_casse_accents_et_espaces():
    """L'OCR rend le français sans accents et colle les mots."""
    assert score_candidate("Écritures à contrôler", "Ecrituresacontroler") == 1.0


def test_un_libelle_noye_dans_un_texte_long_score_moins_qu_un_libelle_proche():
    proche = score_candidate("Repositories", "Top repositories")
    noye = score_candidate("Repositories", "Created 13 commits in 2 repositories")

    assert proche > noye
    assert noye < 0.75  # sous le seuil d'acceptation par défaut


def test_une_correspondance_partielle_ne_peut_jamais_suffire(config):
    """Elle peut départager deux candidats, jamais décider seule."""
    partiel = score_candidate("change logs", "Latest from our changelog")

    assert 0 < partiel < config.overlay_matching.min_score


def test_un_libelle_sans_rapport_ne_score_rien():
    assert score_candidate("Enregistrer", "Pull requests") == 0.0


def test_un_libelle_sans_mot_identifiant_ne_score_rien():
    assert score_candidate("le bouton", "Bouton") == 0.0


# --- fenêtre temporelle ---------------------------------------------------

def test_seuls_les_elements_affiches_pendant_le_segment_sont_candidats(config):
    elements = [element("Post", 10.0, 20.0), element("Draft", 60.0, 70.0)]

    scored = gather_candidates(decision("Post", 10.0, 20.0), elements, config)

    assert [s.element.text for s in scored] == ["Post"]


def test_la_fenetre_deborde_du_segment(config):
    """Le narrateur nomme souvent l'élément juste avant de le montrer."""
    margin = config.overlay_matching.time_margin_seconds
    elements = [element("Post", 20.0 + margin / 2, 30.0)]

    scored = gather_candidates(decision("Post", 10.0, 20.0), elements, config)

    assert len(scored) == 1


def test_un_element_hors_fenetre_meme_de_peu_est_ecarte(config):
    margin = config.overlay_matching.time_margin_seconds
    elements = [element("Post", 20.0 + margin + 1.0, 30.0)]

    assert gather_candidates(decision("Post", 10.0, 20.0), elements, config) == []


# --- verdicts -------------------------------------------------------------

def test_une_correspondance_unique_et_stable_est_retenue(config):
    elements = [element("Pull requests", 5.0, 40.0)]
    target = decision("Pull requests")

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert verdict.accepted
    assert verdict.element_text == "Pull requests"
    assert verdict.box == elements[0].box
    assert verdict.score == 1.0


def test_deux_libelles_equivalents_font_renoncer(config):
    """Le cas central : le même libellé affiché à deux endroits. Encadrer le
    mauvais des deux est pire que ne rien encadrer."""
    elements = [element("Menu", 5.0, 40.0, x=0.1), element("Menu", 5.0, 40.0, x=0.8)]
    target = decision("Menu")

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert not verdict.accepted
    assert "ambigu" in verdict.reason
    assert verdict.rivals == ["Menu"]


def test_un_candidat_nettement_meilleur_l_emporte_malgre_un_homonyme_lointain(config):
    """L'ambiguïté se juge sur l'écart de score, pas sur le nombre de candidats."""
    elements = [
        element("Repositories", 5.0, 40.0, x=0.1),
        element("Created 13 commits in 2 repositories", 5.0, 40.0, x=0.8),
    ]
    target = decision("Repositories")

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert verdict.accepted
    assert verdict.element_text == "Repositories"


def test_aucun_candidat_credible_donne_un_refus_motive(config):
    elements = [element("Pull requests", 5.0, 40.0)]
    target = decision("Enregistrer")

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert not verdict.accepted
    assert "aucun libellé" in verdict.reason


def test_un_element_trop_fugace_est_ecarte(config):
    """Affiché un instant pendant le segment : probablement pas ce dont on parle."""
    elements = [element("Post", 10.0, 11.0)]
    target = decision("Post", 10.0, 20.0)

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert not verdict.accepted
    assert "fugace" in verdict.reason


def test_une_boite_couvrant_l_ecran_est_ecartee(config):
    """L'OCR a capturé un bloc de texte entier, pas un contrôle."""
    elements = [element("Post", 5.0, 40.0, width=0.9, height=0.6)]
    target = decision("Post")

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert not verdict.accepted
    assert "aberrante" in verdict.reason


def test_une_boite_minuscule_est_ecartee(config):
    elements = [element("Post", 5.0, 40.0, width=0.004, height=0.004)]
    target = decision("Post")

    verdict = judge(target, gather_candidates(target, elements, config), config)

    assert not verdict.accepted
    assert "aberrante" in verdict.reason


# --- vue d'ensemble -------------------------------------------------------

def test_seuls_les_segments_nommes_sont_apparies(config):
    decisions = [decision("Post"), decision(None, kind="spatial"), decision(None, kind="none")]
    index = ScreenTextIndex(
        sample_fps=2.0, frames_sampled=10, frames_analysed=2,
        elements=[element("Post", 5.0, 40.0)],
    )

    report = match(decisions, index, config)

    assert report.segments_total == 3
    assert report.segments_named == 1
    assert len(report.accepted) == 1


def test_un_segment_sans_designation_ne_produit_aucun_verdict(config):
    decisions = [decision(None, kind="none")]
    index = ScreenTextIndex(sample_fps=2.0, frames_sampled=1, frames_analysed=1, elements=[])

    report = match(decisions, index, config)

    assert report.candidates == []


def test_seules_les_correspondances_retenues_sont_reportees_dans_l_edl(config):
    decisions = [decision("Post"), decision(None, kind="spatial")]
    decisions[1].id = "seg-002"
    index = ScreenTextIndex(
        sample_fps=2.0, frames_sampled=10, frames_analysed=2,
        elements=[element("Post", 5.0, 40.0)],
    )
    report = match(decisions, index, config)

    updated, applied = apply_to_edl(decisions, report, config)

    assert applied == 1
    assert updated[0].visual_action is not None
    assert updated[0].visual_action.type == config.overlay_matching.action_type
    assert updated[0].visual_action.target == "Post"
    assert updated[1].visual_action is None


def test_un_refus_ne_touche_pas_au_conducteur_de_montage(config):
    """Rien d'incertain ne doit atterrir dans le rendu sans relecture."""
    decisions = [decision("Menu")]
    index = ScreenTextIndex(
        sample_fps=2.0, frames_sampled=10, frames_analysed=2,
        elements=[element("Menu", 5.0, 40.0, x=0.1), element("Menu", 5.0, 40.0, x=0.8)],
    )
    report = match(decisions, index, config)

    updated, applied = apply_to_edl(decisions, report, config)

    assert applied == 0
    assert updated[0].visual_action is None


def test_la_boite_retenue_alimente_un_filtre_ffmpeg_valide(config):
    """Bout en bout : la boîte issue de l'OCR doit produire une expression que
    le parseur de filtres FFmpeg accepte."""
    from overlays import overlay_filter_for

    elements = [element("Post", 5.0, 40.0, x=0.6182291666666667, y=0.5186274509803922)]
    decisions = [decision("Post")]
    index = ScreenTextIndex(
        sample_fps=2.0, frames_sampled=10, frames_analysed=2, elements=elements
    )
    updated, _ = apply_to_edl(decisions, match(decisions, index, config), config)

    fragment = overlay_filter_for(updated[0].visual_action, config, 1920, 1080)

    assert fragment.startswith("drawbox=")
    assert "e-0" not in fragment  # notation scientifique : refusée par FFmpeg
