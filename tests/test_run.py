"""Tests du point d'entrée `run.py` : ordre des étapes, reprise, échec."""

from __future__ import annotations

import importlib
import inspect

import pytest
import typer

import run


# --- ordre ---------------------------------------------------------------------

def test_le_pipeline_complet_comprend_toutes_les_etapes():
    """Avant, `python run.py` omettait recadrage, OCR, souris et rapprochement :
    un passage complet demandait douze commandes tapées à la main."""
    assert set(run.STEP_ORDER) >= {"crop", "screen", "cursor", "match"}
    assert run.plan_steps(None, None) == run.STEP_ORDER


@pytest.mark.parametrize(
    "before, after",
    [
        ("inspect", "crop"),       # le recadrage part du rapport d'inspection
        ("crop", "transcribe"),    # tout l'aval travaille sur la vidéo recadrée
        ("crop", "screen"),
        ("screen", "cursor"),      # le suivi réutilise les images de l'OCR
        ("translate", "match"),    # le rapprochement lit le conducteur
        ("screen", "match"),
        ("cursor", "match"),
        ("match", "render"),       # les cadres doivent être posés avant le rendu
        ("narrate", "retime"),
        ("retime", "subtitles"),
        ("subtitles", "render"),
        ("render", "validate"),
    ],
)
def test_chaque_etape_vient_apres_ce_dont_elle_depend(before, after):
    assert run.STEP_ORDER.index(before) < run.STEP_ORDER.index(after)


# --- choix des étapes --------------------------------------------------------------

def test_reprendre_a_une_etape_execute_la_suite_jusqu_a_la_fin():
    assert run.plan_steps(None, "narrate") == ["narrate", "retime", "subtitles", "render", "validate"]


def test_une_etape_seule():
    assert run.plan_steps("render", None) == ["render"]


def test_step_et_from_s_excluent():
    with pytest.raises(typer.BadParameter, match="s'excluent"):
        run.plan_steps("render", "narrate")


def test_une_etape_inconnue_est_refusee_en_listant_les_etapes():
    with pytest.raises(typer.BadParameter, match="translate"):
        run.plan_steps(None, "tranlsate")


# --- exécution ------------------------------------------------------------------------

def test_les_etapes_s_executent_dans_l_ordre_avec_leurs_durees():
    calls = []

    durations = run.run_steps(["a", "b"], None, None, runner=lambda s, c, i: calls.append(s))

    assert calls == ["a", "b"]
    assert [step for step, _ in durations] == ["a", "b"]


def test_un_echec_arrete_le_passage_et_indique_la_reprise(caplog):
    calls = []

    def runner(step, config, input_override):
        calls.append(step)
        if step == "screen":
            raise RuntimeError("OCR indisponible")

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        run.run_steps(["translate", "screen", "cursor"], None, None, runner=runner)

    assert calls == ["translate", "screen"]
    assert "python run.py --from screen" in caplog.text


def test_le_bilan_totalise_les_durees():
    text = run.summary([("translate", 120.0), ("render", 42.5)])

    assert "translate" in text and "162.5 s" in text


# --- appels des étapes ------------------------------------------------------------------

MODULES = {
    "crop": "crop_chrome", "transcribe": "transcribe", "translate": "translate",
    "screen": "detect_screen_text", "cursor": "detect_cursor", "match": "match_overlays",
    "narrate": "build_narration", "retime": "build_timeline", "subtitles": "subtitles",
    "render": "render_video", "validate": "validate_output",
}


@pytest.mark.parametrize("step, module_name", MODULES.items())
def test_chaque_etape_recoit_tous_ses_parametres(step, module_name, monkeypatch):
    """Régression : `run.py` appelle `main` hors de typer, où une option omise
    reçoit son objet `typer.Option` — qui vaut vrai. `chapters_only` omis
    réduisait la traduction à la seule régénération des chapitres, en silence."""
    module = importlib.import_module(module_name)
    expected = set(inspect.signature(module.main).parameters)
    received = []
    monkeypatch.setattr(module, "main", lambda **kwargs: received.append(set(kwargs)))

    run._run_step(step, None, None)

    assert received == [expected]


def test_la_traduction_traduit_vraiment(monkeypatch):
    import translate

    calls = []
    monkeypatch.setattr(translate, "main", lambda **kwargs: calls.append(kwargs))

    run._run_step("translate", None, None)

    assert calls == [{"config_path": None, "chapters_only": False}]


def test_dans_le_pipeline_le_rapprochement_pose_les_cadres(monkeypatch):
    import match_overlays

    calls = []
    monkeypatch.setattr(match_overlays, "main", lambda **kwargs: calls.append(kwargs))

    run._run_step("match", None, None)

    assert calls[0]["apply"] is True
