"""Tests de la relecture automatique des traductions (étape C).

Les exemples viennent des défauts relevés sur la démo Sales Report ; les seuils
ont été choisis pour les attraper sans toucher aux adaptations fidèles de la
même vidéo, reprises ici.
"""

from __future__ import annotations

import json

import pytest

from schemas import Glossary, GlossaryTerm
from translate import MAX_TOKENS, translate_segment
from translation_checks import (
    is_over_condensed,
    missing_names,
    missing_numbers,
    missing_terms,
    numbers,
    relevant_terms,
    repeated_passage,
    review,
)

GLOSSARY = Glossary(terms=[
    GlossaryTerm(fr="commande", en="order"),
    GlossaryTerm(fr="catégorie", en="category"),
    GlossaryTerm(fr="report des ventes", en="sales report"),
    GlossaryTerm(fr="vignette", en="tile"),
])


# --- glossaire ----------------------------------------------------------------

def test_le_defaut_d_origine_est_detecte():
    missing = missing_terms(
        "moyen par commande. Par exemple 32 commandes", "by command. For example 32 commands", GLOSSARY
    )

    assert [t.en for t in missing] == ["order"]


def test_pluriels_et_formes_derivees_sont_acceptes():
    assert missing_terms("32 commandes", "32 orders", GLOSSARY) == []
    assert missing_terms("le produit commandé", "the ordered product", GLOSSARY) == []


def test_la_comparaison_ignore_accents_et_casse():
    assert missing_terms("Quelle CATEGORIE ?", "Which Category?", GLOSSARY) == []


def test_un_terme_de_plusieurs_mots_est_reconnu():
    assert [t.en for t in relevant_terms("la démo du report des ventes", GLOSSARY)] == ["sales report"]


def test_un_terme_n_est_pas_trouve_a_l_interieur_d_un_autre_mot():
    """« commander » n'est pas « commande » au pluriel."""
    assert relevant_terms("il faut commander", GLOSSARY) == []


def test_seuls_les_termes_presents_sont_proposes():
    assert [t.fr for t in relevant_terms("trois vignettes", GLOSSARY)] == ["vignette"]


# --- nombres ------------------------------------------------------------------

def test_les_separateurs_de_milliers_ne_comptent_pas():
    assert numbers("179 000 euros") == {"179000"} == numbers("€179,000")


def test_un_montant_perdu_est_detecte():
    assert missing_numbers("32 commandes pour 5618 euros", "Orders average €5,618.") == ["32"]


def test_des_nombres_tous_conserves_ne_signalent_rien():
    assert missing_numbers("179 000 euros de revenus, 32 commandes", "€179,000 in revenue, 32 orders") == []


# --- résumé excessif ----------------------------------------------------------

@pytest.mark.parametrize(
    "fr, en",
    [
        (   # seg-009 d'origine
            "ceux qui sont en cours et voilà, on a toutes nos commandes pour la région APAC de "
            "la catégorie service en statut process. Donc on a le bouton clear à",
            "Click Clear.",
        ),
        (   # seg-013 d'origine
            "commandes du client. Voilà pour cette petite démo qui fut un petit peu rapide "
            "effectivement, lorsqu'on vient mettre un filtre, c'est bien en fait ça vient",
            "Here are the client commands.",
        ),
    ],
)
def test_les_resumes_excessifs_d_origine_sont_detectes(fr, en):
    assert is_over_condensed(fr, en)


@pytest.mark.parametrize(
    "fr, en",
    [
        (   # seg-001 : les hésitations retirées, rien d'autre
            "Bonjour à toutes et à tous et bienvenue dans cette démo du présentation du report "
            "des ventes. Alors comment se présente la page et",
            "Welcome to this presentation demo of the sales report page. Here's the page.",
        ),
        (   # seg-006 : une énumération conservée
            "catégorie il est, la région ainsi que le client, la date de la commande, le montant "
            "et les statuts de la commande. Alors comment utiliser ces infos ? Par exemple",
            "Use the category, region, customer, order date, order amount, and order status. "
            "How to use these details?",
        ),
    ],
)
def test_les_adaptations_fideles_ne_sont_pas_signalees(fr, en):
    assert not is_over_condensed(fr, en)


def test_un_texte_court_n_est_pas_juge_sur_sa_longueur():
    assert not is_over_condensed("Merci à tous.", "Thanks.")


def test_la_relecture_formule_des_consignes_de_correction():
    issues = review("32 commandes en tout", "32 commands overall", GLOSSARY)

    assert issues == ["« commande » doit être traduit par « order »."]


# --- boucle de correction -----------------------------------------------------

class ScriptedLlm:
    """Rend les réponses prévues, dans l'ordre, et garde les échanges."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.answers.pop(0)}}]}


def answer(text_en: str) -> str:
    return json.dumps({"reference_kind": "none", "ui_target": None, "text_en": text_en})


def test_une_traduction_fautive_est_redemandee_en_nommant_le_probleme():
    llm = ScriptedLlm(answer("32 commands."), answer("32 orders."))

    result = translate_segment(llm, "32 commandes.", GLOSSARY, 0.2, segment_id="seg-003")

    assert result.text_en == "32 orders."
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "« commande » doit être traduit par « order »" in correction


def test_une_correction_qui_n_arrange_rien_n_est_pas_retenue():
    llm = ScriptedLlm(answer("32 commands."), answer("Some commands."))

    result = translate_segment(llm, "32 commandes.", GLOSSARY, 0.2)

    assert result.text_en == "32 commands."  # la correction avait perdu le nombre en plus


def test_une_bonne_traduction_ne_coute_qu_un_appel():
    llm = ScriptedLlm(answer("32 orders."))

    translate_segment(llm, "32 commandes.", GLOSSARY, 0.2)

    assert len(llm.calls) == 1


def test_aucun_contexte_des_phrases_voisines_n_est_envoye():
    """Régression : fourni en contexte, le texte voisin était traduit aussi, et
    « Here are three tiles… » revenait dans trois segments de suite."""
    llm = ScriptedLlm(answer("32 orders."))

    translate_segment(llm, "32 commandes.", GLOSSARY, 0.2, previous_en="Here are three tiles.")

    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Contexte" not in prompt and "three tiles" not in prompt


def test_une_phrase_reprise_du_segment_precedent_est_redemandee():
    previous = "Here are three tiles: one for revenue, one for the number of orders."
    llm = ScriptedLlm(
        answer("Here are three tiles: one for revenue, one for the number of orders, and 32 orders."),
        answer("For example, 32 orders."),
    )

    result = translate_segment(llm, "Par exemple 32 commandes.", GLOSSARY, 0.2, previous_en=previous)

    assert result.text_en == "For example, 32 orders."
    assert "reprend la phrase précédente" in llm.calls[1]["messages"][-1]["content"]


def test_seuls_les_termes_utiles_du_glossaire_sont_envoyes():
    llm = ScriptedLlm(answer("Three tiles."))

    translate_segment(llm, "Trois vignettes.", GLOSSARY, 0.2)

    prompt = llm.calls[0]["messages"][1]["content"]
    assert "vignette -> tile" in prompt
    assert "commande -> order" not in prompt


def test_chaque_reponse_est_plafonnee_en_longueur():
    """Régression : sans plafond, le mode JSON de llama.cpp a tourné neuf
    minutes sur une correction, en émettant des blancs sans fin."""
    llm = ScriptedLlm(answer("32 commands."), answer("32 orders."))

    translate_segment(llm, "32 commandes.", GLOSSARY, 0.2)

    assert all(call["max_tokens"] == MAX_TOKENS for call in llm.calls)


# --- reprise d'une phrase déjà traduite --------------------------------------------

def test_une_reprise_de_six_mots_est_detectee():
    passage = repeated_passage(
        "Here are three tiles: one for revenue, one for orders.",
        "Here are three tiles, one for revenue, and the price.",
    )

    assert passage == "here are three tiles one for"


def test_des_mots_communs_isoles_ne_sont_pas_une_reprise():
    assert repeated_passage(
        "We filter the orders by region.", "Then we sort the orders by status."
    ) is None


def test_sans_phrase_precedente_rien_a_reprendre():
    assert repeated_passage("", "Here are three tiles: one for revenue, one for orders.") is None


# --- noms propres -----------------------------------------------------------------

def test_le_nom_tape_dans_la_recherche_ne_peut_pas_disparaitre():
    """Régression : « on va mettre Nordic Tech » était devenu « orders for the
    customer are well brought up », sans le nom."""
    fr = "on va mettre Nordic Tech et voilà, on remonte bien les commandes du client."

    assert missing_names(fr, "That's it, orders for the customer are well brought up.") == ["Nordic Tech"]
    assert missing_names(fr, "Enter Nordic Tech: the customer's orders show up.") == []


def test_un_sigle_compte_meme_en_tete_de_phrase():
    assert missing_names("APAC est sélectionné.", "The region is selected.") == ["APAC"]


def test_une_majuscule_de_debut_de_phrase_n_est_pas_un_nom():
    fr = "Donc on filtre. Voilà, on a tout. Ensuite on trie."

    assert missing_names(fr, "We filter, then sort.") == []
