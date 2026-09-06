"""Tests du recollage des phrases avant traduction (étape C).

Whisper découpe sur les silences, pas sur la syntaxe : sur l'extrait de
référence, 25 segments sur 34 s'arrêtaient en milieu de phrase et étaient
traduits isolément, ce qui donnait des bouts sans rapport.
"""

from __future__ import annotations

import pytest

from schemas import TranscriptSegment
from translate import merge_into_sentences


def segment(seg_id: str, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=seg_id, start=start, end=end, text_fr=text)


def test_une_phrase_coupee_en_deux_est_recollee():
    segments = [
        segment("seg-001", 0.0, 11.0, "on se retrouve sur la page"),
        segment("seg-002", 11.0, 16.0, "d'accueil et voilà."),
    ]

    merged = merge_into_sentences(segments, max_seconds=30.0)

    assert len(merged) == 1
    assert merged[0].text_fr == "on se retrouve sur la page d'accueil et voilà."
    assert (merged[0].start, merged[0].end) == (0.0, 16.0)


def test_une_phrase_complete_n_est_pas_recollee():
    segments = [
        segment("seg-001", 0.0, 5.0, "Bienvenue à tous."),
        segment("seg-002", 5.0, 10.0, "Voici la page d'accueil."),
    ]

    assert len(merge_into_sentences(segments, max_seconds=30.0)) == 2


@pytest.mark.parametrize("ponctuation", [".", "!", "?", "…", ":"])
def test_toute_ponctuation_forte_termine_une_phrase(ponctuation):
    segments = [
        segment("seg-001", 0.0, 5.0, f"Voici la page{ponctuation}"),
        segment("seg-002", 5.0, 10.0, "Et ensuite."),
    ]

    assert len(merge_into_sentences(segments, max_seconds=30.0)) == 2


def test_le_recollage_s_arrete_a_la_duree_maximale():
    """Au-delà, le sous-titre devient illisible et le recalage perd sa marge."""
    segments = [
        segment("seg-001", 0.0, 10.0, "une phrase qui continue"),
        segment("seg-002", 10.0, 20.0, "et qui continue encore"),
    ]

    merged = merge_into_sentences(segments, max_seconds=15.0)

    assert len(merged) == 2


def test_plusieurs_fragments_de_suite_sont_tous_recolles():
    segments = [
        segment("seg-001", 0.0, 3.0, "alors on va"),
        segment("seg-002", 3.0, 6.0, "regarder ce que"),
        segment("seg-003", 6.0, 9.0, "contient cette page."),
    ]

    merged = merge_into_sentences(segments, max_seconds=30.0)

    assert len(merged) == 1
    assert merged[0].text_fr == "alors on va regarder ce que contient cette page."


def test_les_identifiants_sont_renumerotes_sans_trou():
    segments = [
        segment("seg-001", 0.0, 3.0, "un fragment"),
        segment("seg-002", 3.0, 6.0, "recollé."),
        segment("seg-003", 6.0, 9.0, "Une autre phrase."),
    ]

    merged = merge_into_sentences(segments, max_seconds=30.0)

    assert [s.id for s in merged] == ["seg-001", "seg-002"]


def test_les_segments_sont_recolles_dans_l_ordre_chronologique():
    segments = [
        segment("seg-002", 5.0, 10.0, "la suite de la phrase."),
        segment("seg-001", 0.0, 5.0, "le début de"),
    ]

    merged = merge_into_sentences(segments, max_seconds=30.0)

    assert len(merged) == 1
    assert merged[0].text_fr == "le début de la suite de la phrase."


def test_une_transcription_vide_ne_pose_pas_de_probleme():
    assert merge_into_sentences([], max_seconds=30.0) == []
