"""Tests du tri sémantique des désignations (étape C).

C'est la règle qui décide si le narrateur nomme un élément ou décrit seulement
une position. Elle est la première ligne de défense contre l'incrustation au
mauvais endroit, et elle se teste sans LLM.
"""

from __future__ import annotations

import pytest

from schemas import TranslationResult, UiReference
from translate import resolve_ui_reference
from ui_reference import identifying_tokens, is_specific_label, match_key, normalize_text


def result(**kwargs) -> TranslationResult:
    return TranslationResult(text_en="Some text", **kwargs)


@pytest.mark.parametrize(
    "label",
    ["Post", "Pull requests", "Top repositories", "Enregistrer", "Écritures", "Manage Journal Entries"],
)
def test_un_libelle_affiche_est_une_designation_valable(label):
    assert is_specific_label(label)


@pytest.mark.parametrize(
    "label",
    [
        "en haut à gauche",       # position pure
        "the top of the page",
        "le bouton",              # catégorie sans nom
        "the button",
        "ce menu",
        "le second menu déroulant",
        "the next tab",
        "cette zone de l'écran",
        "",
        None,
    ],
)
def test_une_position_ou_une_categorie_n_est_pas_une_designation(label):
    assert not is_specific_label(label)


def test_les_cibles_produites_par_le_llm_au_tour_precedent_sont_toutes_rejetees():
    """Sur l'extrait de référence, le modèle avait proposé ces six cibles. Aucune
    ne nomme un élément : les poser telles quelles aurait encadré n'importe
    quoi."""
    for label in ["menu", "second dropdown", "top menu", "button", "button", "button"]:
        assert not is_specific_label(label)


def test_un_mot_de_position_n_invalide_pas_un_libelle_qui_en_contient_un():
    """"Top repositories" est un vrai libellé de l'interface GitHub : le rejeter
    parce qu'il commence par "top" perdrait une cible légitime. C'est le reste
    du libellé qui décide."""
    assert is_specific_label("Top repositories")
    assert identifying_tokens("Top repositories") == ["repositories"]
    assert identifying_tokens("le haut de la page") == []


def test_un_libelle_purement_numerique_reste_valable():
    """Les compteurs d'interface ("18", "3 open") sont des libellés affichés."""
    assert is_specific_label("18")


def test_le_kind_non_nomme_passe_tel_quel():
    for kind in ("spatial", "none"):
        reference = resolve_ui_reference(result(reference_kind=kind), "seg-001")
        assert reference == UiReference(kind=kind)


def test_une_cible_nommee_et_specifique_est_conservee():
    reference = resolve_ui_reference(
        result(reference_kind="named_control", ui_target="  Pull requests "), "seg-001"
    )

    assert reference == UiReference(kind="named_control", label="Pull requests")


def test_une_cible_nommee_mais_vague_est_retrogradee_en_spatial(caplog):
    """Un 3B annonce volontiers "named_control" pour "the button" : le filtre
    déterministe rattrape le modèle plutôt que de compter sur sa discipline."""
    with caplog.at_level("INFO", logger="translate"):
        reference = resolve_ui_reference(
            result(reference_kind="named_control", ui_target="the button"), "seg-007"
        )

    assert reference == UiReference(kind="spatial")
    # Le rejet doit être traçable : c'est la trace qui permettra de mesurer si
    # le modèle rate des cibles ou si le filtre est trop sévère.
    assert "seg-007" in caplog.text and "the button" in caplog.text


def test_une_cible_nommee_sans_libelle_est_retrogradee():
    reference = resolve_ui_reference(
        result(reference_kind="named_control", ui_target=None), "seg-001"
    )

    assert reference == UiReference(kind="spatial")


def test_le_defaut_est_de_ne_rien_designer():
    """Un LLM qui omet les champs ne doit pas produire de cible par accident."""
    parsed = TranslationResult.model_validate_json('{"text_en": "Hello everyone"}')

    assert parsed.reference_kind == "none"
    assert resolve_ui_reference(parsed, "seg-001") == UiReference(kind="none")


def test_le_schema_refuse_une_designation_incoherente():
    with pytest.raises(ValueError):
        UiReference(kind="named_control", label="  ")
    with pytest.raises(ValueError):
        UiReference(kind="spatial", label="Post")


def test_la_normalisation_est_commune_aux_deux_cotes_de_la_comparaison():
    """Le libellé annoncé par le narrateur et celui lu par l'OCR doivent se
    rejoindre : l'OCR rend le français sans accents et colle parfois les mots."""
    assert match_key("Écritures à contrôler") == match_key("Ecrituresacontroler")
    assert normalize_text("Créer une société") == "creer une societe"


def test_le_gabarit_de_reponse_ne_fixe_aucune_valeur_par_defaut():
    """Régression de prompt : le gabarit JSON montrait littéralement
    `"reference_kind": "none"`. Un modèle 3B recopie le squelette — 32 segments
    sur 34 sont ressortis en "none", y compris "Cliquez sur le bouton
    Enregistrer". Le gabarit doit présenter un emplacement, pas une valeur."""
    from translate import SYSTEM_PROMPT

    for kind in ("named_control", "spatial", "none"):
        assert f'"reference_kind": "{kind}"' not in SYSTEM_PROMPT
    assert '"reference_kind":' in SYSTEM_PROMPT


def test_le_prompt_documente_les_trois_classements():
    from translate import SYSTEM_PROMPT

    for kind in ("named_control", "spatial", "none"):
        assert kind in SYSTEM_PROMPT
