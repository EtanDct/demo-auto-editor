"""Tests du recalage temporel (étape E).

Le cas critique est l'extension de plan : un segment dont la narration ne
tient pas dans le plan source, même accéléré au maximum autorisé. C'est là
qu'un écart de calcul se traduit directement par de l'image gelée inutile
dans le rendu final.
"""

from __future__ import annotations

import pytest

from build_timeline import build_timeline, check_source_overlaps
from schemas import EditDecision, NarrationManifestEntry, NarrationSpec


def make_decision(
    seg_id: str, start: float, end: float, pause_before_ms: int = 0, pause_after_ms: int = 0
) -> EditDecision:
    return EditDecision(
        id=seg_id,
        source_start=start,
        source_end=end,
        text_fr="texte source",
        text_en="source text",
        narration=NarrationSpec(
            voice="en_US-amy-medium",
            pause_before_ms=pause_before_ms,
            pause_after_ms=pause_after_ms,
        ),
    )


def make_narration(seg_id: str, duration: float) -> NarrationManifestEntry:
    return NarrationManifestEntry(
        segment_id=seg_id,
        audio_file=f"audio/narration/{seg_id}.wav",
        duration=duration,
        provider="piper",
        voice="en_US-amy-medium",
    )


def run(decisions, narrations, config):
    return build_timeline(decisions, {n.segment_id: n for n in narrations}, config)


def test_un_leger_blanc_en_fin_de_plan_est_tolere(config):
    """Raccourcir pour quelques dixièmes de seconde hacherait le montage."""
    narration = 10.0 - config.retiming.max_slack_seconds / 2
    report = run([make_decision("seg-001", 0.0, 10.0)], [make_narration("seg-001", narration)], config)

    (entry,) = report.entries
    assert (entry.new_start, entry.new_end) == (0.0, 10.0)
    assert entry.audio_speed_factor == 1.0
    assert not entry.extended
    assert report.warnings == []


def test_un_plan_qui_tourne_a_vide_est_raccourci(config):
    """Le premier plan de l'extrait de référence durait 11s pour 4.8s de voix :
    six secondes de vidéo sans rien à dire."""
    report = run([make_decision("seg-001", 0.0, 10.0)], [make_narration("seg-001", 4.0)], config)

    (entry,) = report.entries
    assert entry.new_end == pytest.approx(4.0 + config.retiming.max_slack_seconds)
    assert len(report.warnings) == 1


def test_le_raccourcissement_avance_les_segments_suivants(config):
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]
    narrations = [make_narration("seg-001", 4.0), make_narration("seg-002", 9.5)]

    first, second = run(decisions, narrations, config).entries

    assert first.new_end == pytest.approx(4.8)
    assert second.new_start == pytest.approx(4.8)
    assert second.new_end - second.new_start == pytest.approx(10.0)


def test_un_plan_n_est_jamais_reduit_sous_le_minimum(config):
    """Un plan d'une fraction de seconde serait illisible."""
    report = run([make_decision("seg-001", 0.0, 10.0)], [make_narration("seg-001", 0.1)], config)

    (entry,) = report.entries
    assert entry.new_end - entry.new_start == pytest.approx(config.retiming.min_shot_seconds)


def test_les_pauses_comptent_dans_la_duree_a_caser(config):
    """5.0s de voix + 1.3s de pauses > 6.0s de plan : il faut accélérer."""
    decision = make_decision("seg-001", 0.0, 6.0, pause_before_ms=300, pause_after_ms=1000)
    report = run([decision], [make_narration("seg-001", 5.0)], config)

    (entry,) = report.entries
    assert entry.audio_speed_factor == pytest.approx(6.3 / 6.0, rel=1e-4)
    assert not entry.extended


def test_depassement_leger_absorbe_par_la_vitesse_seule(config):
    """Un dépassement dans les bornes n'étend pas le plan et ne décale rien."""
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]
    narrations = [make_narration("seg-001", 10.5), make_narration("seg-002", 9.5)]

    report = run(decisions, narrations, config)

    first, second = report.entries
    assert first.audio_speed_factor == pytest.approx(1.05)
    assert not first.extended
    assert (first.new_start, first.new_end) == (0.0, 10.0)
    assert (second.new_start, second.new_end) == (10.0, 20.0)


def test_extension_dimensionnee_sur_la_narration_acceleree(config):
    """Régression : le plan ne doit être étendu que de ce qui manque *après*
    accélération. Le dimensionner sur la durée brute gèle l'image du facteur
    d'accélération en trop (~8% du segment) pour rien."""
    report = run([make_decision("seg-001", 0.0, 10.0)], [make_narration("seg-001", 20.0)], config)

    (entry,) = report.entries
    speed = config.retiming.max_speed_factor
    played_duration = 20.0 / speed

    assert entry.extended
    assert entry.audio_speed_factor == speed
    assert entry.new_end - entry.new_start == pytest.approx(played_duration)
    # L'audio joué remplit exactement le plan : aucune image gelée en trop.
    assert entry.new_end - entry.new_start < 20.0


def test_extension_decale_les_segments_suivants_sans_chevauchement(config):
    decisions = [
        make_decision("seg-001", 0.0, 10.0),
        make_decision("seg-002", 10.0, 20.0),
        make_decision("seg-003", 20.0, 30.0),
    ]
    narrations = [
        make_narration("seg-001", 20.0),
        make_narration("seg-002", 9.5),
        make_narration("seg-003", 9.5),
    ]

    report = run(decisions, narrations, config)

    first, second, third = report.entries
    extension = first.new_end - 10.0
    assert extension > 0
    assert (second.new_start, second.new_end) == pytest.approx((10.0 + extension, 20.0 + extension))
    assert (third.new_start, third.new_end) == pytest.approx((20.0 + extension, 30.0 + extension))
    # Timeline strictement séquentielle : c'est ce que valide validate_output.
    assert first.new_end == pytest.approx(second.new_start)
    assert second.new_end == pytest.approx(third.new_start)


def test_extension_modeste_ne_demande_pas_de_relecture(config):
    """needs_review mesure l'étirement réel du plan, pas le dépassement brut."""
    report = run([make_decision("seg-001", 0.0, 10.0)], [make_narration("seg-001", 12.0)], config)

    (entry,) = report.entries
    assert entry.extended
    assert not entry.needs_review


def test_extension_importante_demande_une_relecture(config):
    report = run([make_decision("seg-001", 0.0, 10.0)], [make_narration("seg-001", 20.0)], config)

    (entry,) = report.entries
    assert entry.needs_review
    assert len(report.warnings) == 1


def test_segment_sans_narration_est_ignore_avec_avertissement(config):
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]

    report = run(decisions, [make_narration("seg-001", 9.5)], config)

    assert [e.id for e in report.entries] == ["seg-001"]
    assert any("seg-002" in w for w in report.warnings)


def test_segments_traites_dans_l_ordre_source_pas_l_ordre_du_fichier(config):
    decisions = [make_decision("seg-002", 10.0, 20.0), make_decision("seg-001", 0.0, 10.0)]
    narrations = [make_narration("seg-001", 9.5), make_narration("seg-002", 9.5)]

    report = run(decisions, narrations, config)

    assert [e.id for e in report.entries] == ["seg-001", "seg-002"]


def test_chevauchement_source_detecte():
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 8.0, 20.0)]

    warnings = check_source_overlaps(decisions)

    assert len(warnings) == 1
    assert "seg-001" in warnings[0] and "seg-002" in warnings[0]


def test_segments_jointifs_ne_sont_pas_un_chevauchement():
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]

    assert check_source_overlaps(decisions) == []
