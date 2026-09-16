"""Tests du découpage en chapitres (étape C) et du carton de fin.

Les frontières sont calculées, le LLM ne fait que nommer : ce choix vient d'une
sonde sur la démo Sales Report, où un 3B prié de couper lui-même faisait un
chapitre par phrase. Les tests fixent donc surtout la part déterministe, et ce
qui se passe quand le modèle répond mal.
"""

from __future__ import annotations

import json

import pytest

from intro_card import build_card_filter, load_outro_text
from render_video import resolve_logo
from schemas import ChapterPlan, IntroText
from translate import (
    assemble_chapters,
    build_chapters,
    chapters_prompt,
    generate_chapter_titles,
    slice_into_chapters,
)

from test_build_timeline import make_decision


def narration(durations: list[float]):
    decisions, start = [], 0.0
    for index, duration in enumerate(durations, 1):
        decisions.append(make_decision(f"seg-{index:03d}", start, start + duration))
        start += duration
    return decisions


class FakeLlm:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


# --- découpage --------------------------------------------------------------

def test_la_narration_est_decoupee_en_tranches_de_durees_voisines():
    decisions = narration([10.0] * 14)  # 140 s, un chapitre pour 35 s

    slices = slice_into_chapters(decisions, chapter_seconds=35.0, max_chapters=6)

    # Milieux à 5, 15… 135 s : la 4e phrase (35 s) ouvre pile la 2e tranche.
    assert [len(s) for s in slices] == [3, 4, 3, 4]
    assert [d.id for part in slices for d in part] == [d.id for d in decisions]


def test_une_phrase_va_a_la_tranche_ou_tombe_son_milieu():
    """Une phrase longue à cheval sur deux tranches ne fausse pas le partage."""
    decisions = narration([30.0, 5.0, 5.0, 30.0])  # milieu de la 2e à 32.5 s sur 70

    slices = slice_into_chapters(decisions, chapter_seconds=35.0, max_chapters=6)

    assert [[d.id for d in s] for s in slices] == [["seg-001", "seg-002"], ["seg-003", "seg-004"]]


def test_le_nombre_de_chapitres_est_plafonne():
    slices = slice_into_chapters(narration([10.0] * 40), chapter_seconds=35.0, max_chapters=6)

    assert len(slices) == 6


def test_une_video_courte_n_a_pas_de_chapitres():
    """Un chapitre unique répéterait le titre."""
    assert slice_into_chapters(narration([10.0] * 5), 35.0, 6) == []


def test_pas_de_chapitre_d_une_seule_phrase_par_manque_de_phrases():
    """Trois longues phrases sur 105 s : trois tranches d'une phrase chacune
    ne seraient que des sous-titres. Il en faut au moins deux par chapitre."""
    assert slice_into_chapters(narration([35.0, 35.0, 35.0]), 35.0, 6) == []


# --- noms -------------------------------------------------------------------

def test_le_modele_nomme_les_tranches_a_temperature_nulle():
    """Deux passages sur la même vidéo doivent donner les mêmes chapitres."""
    decisions = narration([10.0] * 8)
    slices = slice_into_chapters(decisions, 40.0, 6)
    llm = FakeLlm(json.dumps({"titles": ["Key figures", "Filters"]}))

    titles = generate_chapter_titles(llm, slices)

    assert titles == ["Key figures", "Filters"]
    (call,) = llm.calls
    assert call["temperature"] == 0.0
    assert "Part 2:" in call["messages"][1]["content"]


def test_la_consigne_annonce_le_nombre_de_titres_et_proscrit_les_generiques():
    prompt = chapters_prompt(4)

    assert "4 parts" in prompt and "title of part 4" in prompt
    assert "Introduction" in prompt


def test_les_titres_sont_associes_aux_premieres_phrases_des_tranches():
    decisions = narration([10.0] * 8)
    slices = slice_into_chapters(decisions, 40.0, 6)

    plan = assemble_chapters(["Key figures", "Filters"], slices, max_chars=32)

    assert [(c.title, c.first_segment) for c in plan.chapters] == [
        ("Key figures", "seg-001"), ("Filters", slices[1][0].id),
    ]


def test_un_titre_en_trop_est_ignore():
    slices = slice_into_chapters(narration([10.0] * 8), 40.0, 6)

    plan = assemble_chapters(["A", "B", "C"], slices, max_chars=32)

    assert [c.title for c in plan.chapters] == ["A", "B"]


@pytest.mark.parametrize("titles", [["Only one"], ["Key figures", "   "]])
def test_un_titre_manquant_ou_vide_fait_tout_refuser(titles):
    """Un chapitre sans nom afficherait un index seul."""
    slices = slice_into_chapters(narration([10.0] * 8), 40.0, 6)

    assert assemble_chapters(titles, slices, max_chars=32) == ChapterPlan()


def test_un_titre_trop_long_est_coupe_entre_deux_mots():
    slices = slice_into_chapters(narration([10.0] * 8), 40.0, 6)

    plan = assemble_chapters(
        ['"Filtering the sales orders by region and category."', "Search"], slices, max_chars=24
    )

    assert plan.chapters[0].title == "Filtering the sales"


def test_une_reponse_illisible_ne_fait_pas_tomber_la_traduction(config):
    decisions = narration([10.0] * 14)

    assert build_chapters(FakeLlm("pas du json"), decisions, config) == ChapterPlan()
    assert build_chapters(FakeLlm('{"titles": "Filters"}'), decisions, config) == ChapterPlan()
    assert build_chapters(FakeLlm('{"chapters": []}'), decisions, config) == ChapterPlan()


def test_une_bonne_reponse_donne_un_plan_complet(config):
    decisions = narration([10.0] * 14)
    llm = FakeLlm(json.dumps({"titles": ["Page", "Table", "Filters", "Search"]}))

    plan = build_chapters(llm, decisions, config)

    assert [c.title for c in plan.chapters] == ["Page", "Table", "Filters", "Search"]
    assert plan.chapters[0].first_segment == "seg-001"


# --- carton de fin ----------------------------------------------------------

def test_le_carton_de_fin_reprend_le_titre_de_l_introduction(config):
    patched = config.model_copy(deep=True)
    patched.intro.title = "Sales Report Overview"
    patched.outro.title = None

    assert load_outro_text(patched) == IntroText(
        title="Sales Report Overview", subtitle=patched.outro.subtitle
    )


def test_le_carton_de_fin_peut_avoir_son_propre_titre(config):
    patched = config.model_copy(deep=True)
    patched.outro.title = "Questions ?"
    patched.outro.subtitle = None

    assert load_outro_text(patched) == IntroText(title="Questions ?", subtitle="")


def test_sans_titre_nulle_part_pas_de_carton_de_fin(config, monkeypatch, tmp_path):
    patched = config.model_copy(deep=True)
    patched.intro.title = None
    patched.outro.title = None
    monkeypatch.setattr(type(patched.paths), "resolve", lambda self, field: tmp_path)

    assert load_outro_text(patched) is None


def test_le_carton_de_fin_fond_selon_sa_propre_duree(config):
    fragment = build_card_filter(IntroText(title="Fin"), config, duration=4.0, fade=0.5)

    assert "(4.00-t)/0.50" in fragment
    assert "lt(t,3.50)" in fragment


# --- logo -------------------------------------------------------------------

def test_sans_logo_rien_a_resoudre(config):
    assert resolve_logo(config) is None


def test_un_logo_introuvable_est_signale_avant_l_encodage(config, tmp_path):
    patched = config.model_copy(deep=True)
    patched.layout.logo_path = str(tmp_path / "absent.png")

    with pytest.raises(FileNotFoundError, match="logo_path"):
        resolve_logo(patched)


def test_un_logo_present_est_resolu(config, tmp_path):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"png")
    patched = config.model_copy(deep=True)
    patched.layout.logo_path = str(logo)

    assert resolve_logo(patched) == logo
