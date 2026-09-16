"""Tests des sous-titres (étape G) : découpage en lignes et formatage SRT."""

from __future__ import annotations

from subtitles import _format_timestamp, build_srt, split_into_cues, wrap_lines
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
    assert wrap_lines("Click the button.", 42) == ["Click the button."]


def test_coupure_sur_les_mots_jamais_au_milieu_d_un_mot():
    lines = wrap_lines("Select the company code in the corresponding field.", 30)

    assert len(lines) == 2
    assert all(len(line) <= 30 for line in lines)
    assert " ".join(lines) == "Select the company code in the corresponding field."


def test_un_texte_trop_long_est_reparti_sur_plusieurs_sous_titres():
    """Régression : il était tronqué, et des mots disparaissaient de la vidéo
    livrée sans que rien ne le signale au contrôle automatique."""
    cues = split_into_cues("one two three four five six seven eight nine ten", 12, 2)

    assert len(cues) > 1
    assert " ".join(l for cue in cues for l in cue) == "one two three four five six seven eight nine ten"


def test_chaque_sous_titre_respecte_les_limites_de_lignes():
    cues = split_into_cues("one two three four five six seven eight nine ten", 12, 2)

    assert all(len(cue) <= 2 for cue in cues)
    assert all(len(line) <= 12 for cue in cues for line in cue)


def test_mot_plus_long_que_la_ligne_est_conserve_tel_quel():
    """Tronquer au milieu d'un identifiant SAP le rendrait illisible."""
    lines = wrap_lines("ManageJournalEntriesApp", 10)

    assert lines == ["ManageJournalEntriesApp"]


def test_texte_vide_ne_produit_aucune_ligne():
    assert wrap_lines("", 42) == []
    assert split_into_cues("", 42, 2) == []


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


def test_le_temps_du_segment_est_partage_entre_les_sous_titres(config):
    """Un segment couvre désormais une phrase entière : son texte tient rarement
    en un seul sous-titre, et les afficher tous en même temps serait illisible."""
    long_text = " ".join(["word"] * 40)
    decisions = [make_decision("seg-001", 0.0, 20.0)]
    decisions[0].text_en = long_text
    timeline = {"seg-001": make_entry("seg-001", 0.0, 20.0)}

    srt = build_srt(decisions, timeline, config)
    blocks = srt.strip().split("\n\n")

    assert len(blocks) > 1
    assert blocks[0].splitlines()[1].startswith("00:00:00,000")
    assert blocks[-1].splitlines()[1].endswith("00:00:20,000")


def test_les_sous_titres_ne_se_chevauchent_pas(config):
    decisions = [make_decision("seg-001", 0.0, 20.0)]
    decisions[0].text_en = " ".join(["word"] * 40)
    timeline = {"seg-001": make_entry("seg-001", 0.0, 20.0)}

    srt = build_srt(decisions, timeline, config)
    bornes = [b.splitlines()[1] for b in srt.strip().split("\n\n")]
    fins = [b.split(" --> ")[1] for b in bornes]
    debuts = [b.split(" --> ")[0] for b in bornes]

    assert debuts[1:] == fins[:-1]


# --- coupure entre deux sous-titres -------------------------------------------

def test_un_sous_titre_se_termine_sur_une_fin_de_phrase_plutot_qu_en_plein_groupe():
    """Régression : « …one for revenues, one for the » restait en suspens."""
    text = (
        "Let's see how it behaves. Here are three tiles: one for revenues, "
        "one for the number of orders, and one for the average price."
    )

    cues = split_into_cues(text, 42, 2)

    # Plein, le premier sous-titre s'arrêtait sur « …one for the » ; il s'arrête
    # désormais sur la virgule qui précède.
    assert cues[0][-1].endswith("revenues,")
    assert all(not cue[-1].endswith(("the", "for", "of")) for cue in cues)
    assert " ".join(l for cue in cues for l in cue).split() == text.split()


def test_a_defaut_de_point_la_coupure_tombe_sur_une_virgule():
    text = (
        "Filter by region and category, then clear every filter "
        "to see all the orders again from the start"
    )

    cues = split_into_cues(text, 30, 2)

    assert cues[0][-1].endswith(",")


def test_une_ponctuation_trop_precoce_ne_vide_pas_le_sous_titre():
    """Couper après « Yes, » laisserait un sous-titre d'un mot."""
    text = "Yes, then we open the sales report and look at every order in the table below it"

    cues = split_into_cues(text, 30, 2)

    assert len(" ".join(cues[0]).split()) > 2


def test_sans_ponctuation_le_remplissage_reste_maximal():
    cues = split_into_cues("one two three four five six seven eight nine ten", 12, 2)

    assert cues[0] == ["one two", "three four"]
