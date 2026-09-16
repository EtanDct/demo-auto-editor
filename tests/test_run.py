"""Tests du point d'entrée `run.py`."""

from __future__ import annotations

import importlib
import inspect

import run


def test_les_etapes_simples_n_ont_pas_d_autre_option_que_la_config():
    """Régression : `run.py` appelle `main` hors de typer. Une option qu'il ne
    passe pas reçoit alors son objet `typer.Option`, qui vaut vrai. Ajouter
    `--chapters-only` à la traduction l'avait réduite, en silence, à la seule
    régénération des chapitres. Une étape qui prend une autre option doit être
    appelée explicitement, avec tous ses paramètres."""
    for step, module_name in run.SIMPLE_STEPS.items():
        parameters = inspect.signature(importlib.import_module(module_name).main).parameters
        assert list(parameters) == ["config_path"], (step, list(parameters))


def test_la_traduction_traduit_vraiment(monkeypatch):
    import translate

    calls = []
    monkeypatch.setattr(translate, "main", lambda **kwargs: calls.append(kwargs))

    run._run_step("translate", None, None)

    assert calls == [{"config_path": None, "chapters_only": False}]
