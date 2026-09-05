"""Tests des sous-titres (étape G) : découpage en lignes et formatage SRT."""

from __future__ import annotations

from subtitles import _format_timestamp, build_srt, wrap_text
from schemas import TimelineEntry

from test_build_timeline import make_decision


def make_entry(seg_id: str, new_start: float, new_end: float) -> TimelineEntry:
    return TimelineEntry(
        id=seg_id,
        source_start=new_start,
        source_end=new_end,
        new_start=new_start,
        new_end=new_end,
        narration_duration=new_end - new_start,
    )


def test_texte_court_tient_sur_une_ligne():
    assert wrap_text("Click the button.", 42, 2) == ["Click the button."]


def test_coupure_sur_les_mots_jamais_au_milieu_d_un_mot():
    lines = wrap_text("Select the company code in the corresponding field.", 30, 2)

    assert len(lines) == 2
    assert all(len(line) <= 30 for line in lines)
    assert " ".join(lines) == "Select the company code in the corresponding field."


def test_texte_trop_long_est_tronque_avec_avertissement(caplog):
    lines = wrap_text("one two three four five six seven eight nine ten", 12, 2)

    assert len(lines) == 2
    assert "tronqué" in caplog.text


def test_mot_plus_long_que_la_ligne_est_conserve_tel_quel():
    """Tronquer au milieu d'un identifiant SAP le rendrait illisible."""
    lines = wrap_text("ManageJournalEntriesApp", 10, 2)

    assert lines == ["ManageJournalEntriesApp"]


def test_texte_vide_ne_produit_aucune_ligne():
    assert wrap_text("", 42, 2) == []


def test_format_timestamp_srt():
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(3723.456) == "01:02:03,456"
    assert _format_timestamp(9.9999) == "00:00:10,000"


def test_srt_numerote_et_ordonne_selon_la_timeline(config):
    decisions = [make_decision("seg-002", 10.0, 20.0), make_decision("seg-001", 0.0, 10.0)]
    timeline = {"seg-001": make_entry("seg-001", 0.0, 10.0), "seg-002": make_entry("seg-002", 12.0, 22.0)}

    srt = build_srt(decisions, timeline, config)

    assert srt.startswith("1\n00:00:00,000 --> 00:00:10,000\n")
    assert "2\n00:00:12,000 --> 00:00:22,000\n" in srt


def test_segment_absent_de_la_timeline_est_saute_sans_trou_de_numerotation(config):
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]
    timeline = {"seg-002": make_entry("seg-002", 10.0, 20.0)}

    srt = build_srt(decisions, timeline, config)

    assert srt.startswith("1\n")
    assert "2\n" not in srt
