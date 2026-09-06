"""Tests du carton d'introduction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from intro_card import build_card_filter, load_intro_text
from schemas import IntroText

FFMPEG = shutil.which("ffmpeg")


def libelles_ecrits(fragment: str) -> list[str]:
    """Contenus des fichiers de libellé référencés par un fragment."""
    return [
        Path(part.split("'")[0].replace("\\:", ":")).read_text(encoding="utf-8")
        for part in fragment.split("textfile='")[1:]
    ]


@pytest.fixture
def intro_config(config):
    patched = config.model_copy(deep=True)
    patched.overlays.font_path = "C:/Windows/Fonts/arial.ttf"
    return patched


def test_le_titre_et_le_sous_titre_sont_affiches(intro_config):
    fragment = build_card_filter(
        IntroText(title="Ma démo", subtitle="Un tour du produit"), intro_config
    )

    assert fragment.count("drawtext=") == 2
    assert libelles_ecrits(fragment) == ["Ma démo", "Un tour du produit"]


def test_un_carton_sans_sous_titre_n_affiche_qu_une_ligne(intro_config):
    fragment = build_card_filter(IntroText(title="Ma démo"), intro_config)

    assert fragment.count("drawtext=") == 1


def test_le_texte_passe_par_un_fichier(intro_config):
    """Un titre de LLM contient couramment une apostrophe, inéchappable dans un
    argument de filtre FFmpeg."""
    fragment = build_card_filter(IntroText(title="Note : l'écran"), intro_config)

    assert libelles_ecrits(fragment) == ["Note : l'écran"]


def test_le_carton_fond_a_l_entree_et_a_la_sortie(intro_config):
    fragment = build_card_filter(IntroText(title="Ma démo"), intro_config)

    assert "alpha='if(lt(t," in fragment


def test_le_titre_de_config_prime_sur_celui_du_llm(intro_config, tmp_path):
    patched = intro_config.model_copy(deep=True)
    patched.intro.title = "Titre impose"
    patched.intro.subtitle = "Sous-titre impose"

    text = load_intro_text(patched)

    assert text == IntroText(title="Titre impose", subtitle="Sous-titre impose")


def test_sans_titre_nulle_part_le_carton_est_ignore(intro_config, monkeypatch, tmp_path):
    patched = intro_config.model_copy(deep=True)
    patched.intro.title = None
    monkeypatch.setattr(
        type(patched.paths), "resolve", lambda self, field: tmp_path
    )

    assert load_intro_text(patched) is None


def test_un_titre_vide_est_refuse_par_le_schema():
    with pytest.raises(ValueError):
        IntroText(title="")


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg absent du PATH")
def test_ffmpeg_accepte_le_filtre_du_carton(intro_config):
    fragment = build_card_filter(
        IntroText(title="Ma démo : le produit", subtitle="Un tour d'horizon"), intro_config
    )

    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
         "-i", "color=c=0x101418:s=640x360:d=5", "-vf", fragment, "-f", "null", "-"],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr[:300]
