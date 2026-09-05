"""Tests du découpage source et de la construction des filtres FFmpeg (étapes F + H).

On ne lance jamais FFmpeg ici : on vérifie que le graphe de filtres décrit
bien la timeline (morceaux jointifs, gels de plan au bon endroit, durée
totale cohérente), ce qui est la partie où une erreur coûte un rendu complet.
"""

from __future__ import annotations

import pytest

from render_video import build_audio_filter, build_pieces, build_video_filter, total_output_duration
from schemas import NarrationManifestEntry

from test_build_timeline import make_decision, make_narration
from test_subtitles import make_entry


def test_les_morceaux_couvrent_la_source_sans_trou_ni_chevauchement():
    decisions = [make_decision("seg-001", 2.0, 10.0), make_decision("seg-002", 15.0, 20.0)]
    entries = [make_entry("seg-001", 2.0, 10.0), make_entry("seg-002", 15.0, 20.0)]

    pieces = build_pieces(decisions, entries, source_duration=30.0)

    assert [p.kind for p in pieces] == ["gap", "segment", "gap", "segment", "gap"]
    assert pieces[0].start == 0.0
    assert pieces[-1].end == 30.0
    for previous, current in zip(pieces, pieces[1:]):
        assert previous.end == pytest.approx(current.start)


def test_pas_de_morceau_vide_quand_les_segments_sont_jointifs():
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]
    entries = [make_entry("seg-001", 0.0, 10.0), make_entry("seg-002", 10.0, 20.0)]

    pieces = build_pieces(decisions, entries, source_duration=20.0)

    assert [p.kind for p in pieces] == ["segment", "segment"]


def test_l_extension_de_plan_est_reportee_sur_le_morceau():
    decisions = [make_decision("seg-001", 0.0, 10.0)]
    entries = [make_entry("seg-001", 0.0, 10.0)]
    entries[0].new_end = 14.0  # plan étendu de 4s par le recalage

    (piece,) = build_pieces(decisions, entries, source_duration=10.0)

    assert piece.extension == pytest.approx(4.0)
    assert total_output_duration([piece]) == pytest.approx(14.0)


def test_segment_sans_entree_timeline_est_ignore():
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]

    pieces = build_pieces(decisions, [make_entry("seg-001", 0.0, 10.0)], source_duration=20.0)

    assert [p.kind for p in pieces] == ["segment", "gap"]


def test_le_filtre_video_gele_uniquement_les_morceaux_etendus(config, tmp_path):
    decisions = [make_decision("seg-001", 0.0, 10.0), make_decision("seg-002", 10.0, 20.0)]
    entries = [make_entry("seg-001", 0.0, 10.0), make_entry("seg-002", 10.0, 20.0)]
    entries[0].new_end = 12.0
    pieces = build_pieces(decisions, entries, source_duration=20.0)

    graph, label = build_video_filter(pieces, config, 1920, 1080, tmp_path / "subs.srt", fps=30.0)

    assert label == "[vout]"
    assert "tpad=stop_mode=clone:stop_duration=2.000[v0]" in graph
    assert "[v1]" in graph and "stop_duration" not in graph.split("[v0]")[1].split("[v1]")[0]
    assert "concat=n=2:v=1:a=0[vconcat]" in graph


def test_le_chemin_des_sous_titres_est_echappe_pour_ffmpeg(config, tmp_path):
    pieces = build_pieces([], [], source_duration=10.0)

    graph, _ = build_video_filter(pieces, config, 1920, 1080, tmp_path / "subs.srt", fps=30.0)

    subtitles_arg = graph.split("subtitles='")[1].split("'")[0]
    assert subtitles_arg.endswith("subs.srt")
    # Séparateurs en "/" et seul le ":" du lecteur Windows reste échappé :
    # tout autre antislash serait interprété par le parseur de filtres FFmpeg.
    assert subtitles_arg.count("\\") == subtitles_arg.count(r"\:")


def test_le_filtre_audio_place_chaque_narration_a_son_timecode():
    decisions = [make_decision("seg-001", 0.0, 10.0, pause_before_ms=150)]
    timeline = {"seg-001": make_entry("seg-001", 5.0, 15.0)}
    narrations = {"seg-001": make_narration("seg-001", 8.0)}

    graph, files, label = build_audio_filter(decisions, timeline, narrations, 1, total_duration=20.0)

    assert label == "[aout]"
    assert "[1:a]adelay=delays=5150:all=1[a0]" in graph
    assert "apad=whole_dur=20.000[aout]" in graph
    assert [f.name for f in files] == ["seg-001.wav"]


def test_l_acceleration_est_appliquee_avant_le_decalage():
    """atempo après adelay décalerait aussi le silence d'amorce."""
    decisions = [make_decision("seg-001", 0.0, 10.0)]
    timeline = {"seg-001": make_entry("seg-001", 0.0, 10.0)}
    timeline["seg-001"].audio_speed_factor = 1.08
    narrations = {"seg-001": make_narration("seg-001", 10.5)}

    graph, _, _ = build_audio_filter(decisions, timeline, narrations, 1, total_duration=10.0)

    assert "[1:a]atempo=1.0800,adelay=delays=0:all=1[a0]" in graph


def test_aucun_segment_audio_est_une_erreur_explicite():
    with pytest.raises(ValueError, match="Aucun segment audio"):
        build_audio_filter([], {}, {}, 1, total_duration=10.0)
