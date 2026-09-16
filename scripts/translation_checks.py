"""Contrôles déterministes d'une traduction, avant de l'accepter (étape C).

Un modèle 3B respecte mal les consignes générales. Relu segment par segment sur
la démo Sales Report, il avait :
- rendu « commande » par *command* au lieu d'*order* ;
- réduit 30 mots à « Click Clear. », en perdant la phrase qui décrivait le
  résultat affiché ;
- laissé tomber des montants cités à l'oral.

Autre perte relevée : « on va mettre Nordic Tech » traduit sans le nom tapé
dans la recherche. Un nom propre s'affiche à l'écran ; la narration ne peut pas
le taire.

Un dernier défaut est apparu en fournissant au modèle les phrases voisines comme
contexte : il les traduisait aussi, et la même phrase anglaise revenait d'un
segment à l'autre — audible en voix off.

Ces défauts se détectent sans modèle. Quand l'un apparaît, la traduction
est redemandée une fois en nommant précisément le problème — une consigne ciblée
passe là où la consigne générale n'est pas suivie.
"""

from __future__ import annotations

import re

from schemas import Glossary, GlossaryTerm
from ui_reference import normalize_text

# En dessous de ce rapport de longueur, une adaptation a perdu de l'information,
# pas seulement des hésitations. Mesuré sur la démo : les segments fidèles
# tombaient entre 0,54 et 1,2 ; les résumés excessifs entre 0,07 et 0,27.
MIN_LENGTH_RATIO = 0.4
# Sous ce nombre de mots, le texte source est trop court pour juger un rapport.
MIN_WORDS_FOR_RATIO = 12


def _fr_pattern(fr: str) -> re.Pattern:
    """Terme français en mots entiers, pluriel en -s ou -x toléré, sans accents."""
    words = [re.escape(w) for w in normalize_text(fr).split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"[sx]?\b")


def _en_pattern(en: str) -> re.Pattern:
    """Terme anglais en début de mot : « order » accepte « orders », « ordered »."""
    words = [re.escape(w) for w in en.lower().split()]
    return re.compile(r"\b" + r"\s+".join(words))


def relevant_terms(text_fr: str, glossary: Glossary) -> list[GlossaryTerm]:
    """Termes du glossaire présents dans le texte français."""
    normalized = normalize_text(text_fr)
    return [term for term in glossary.terms if _fr_pattern(term.fr).search(normalized)]


def missing_terms(text_fr: str, text_en: str, glossary: Glossary) -> list[GlossaryTerm]:
    """Termes présents en français dont la traduction imposée manque en anglais."""
    english = text_en.lower()
    return [
        term for term in relevant_terms(text_fr, glossary)
        if not _en_pattern(term.en).search(english)
    ]


_NUMBER = re.compile(r"\d(?:[\d\s., ]*\d)?")


def numbers(text: str) -> set[str]:
    """Nombres écrits en chiffres, séparateurs retirés : « 179 000 » = « 179,000 »."""
    return {re.sub(r"[\s., ]", "", match) for match in _NUMBER.findall(text)}


def missing_numbers(text_fr: str, text_en: str) -> list[str]:
    return sorted(numbers(text_fr) - numbers(text_en), key=lambda n: (len(n), n))


def proper_names(text: str) -> list[str]:
    """Noms propres et sigles cités : « Nordic Tech », « APAC », « SAP ».

    Un mot capitalisé en tête de phrase ne compte pas (« Donc », « Voilà ») ;
    un sigle tout en capitales compte partout. Les mots capitalisés qui se
    suivent forment un seul nom.
    """
    names: list[str] = []
    current: list[str] = []
    sentence_start = True
    for raw in text.split():
        word = raw.strip(".,;:!?…«»\"'()")
        is_acronym = len(word) >= 2 and word.isupper() and word.isalpha()
        is_capitalized = word[:1].isupper() and not sentence_start
        if word and (is_acronym or is_capitalized):
            current.append(word)
        else:
            if current:
                names.append(" ".join(current))
            current = []
        sentence_start = raw.endswith((".", "!", "?", "…"))
    if current:
        names.append(" ".join(current))
    return names


def missing_names(text_fr: str, text_en: str) -> list[str]:
    english = text_en.lower()
    unique = dict.fromkeys(proper_names(text_fr))
    return [name for name in unique if name.lower() not in english]


def is_over_condensed(text_fr: str, text_en: str) -> bool:
    fr_words, en_words = len(text_fr.split()), len(text_en.split())
    return fr_words >= MIN_WORDS_FOR_RATIO and en_words / fr_words < MIN_LENGTH_RATIO


# Suite de mots qu'on ne retrouve pas par hasard dans deux phrases voisines.
REPEAT_WORDS = 6


def repeated_passage(previous_en: str, text_en: str) -> str | None:
    """Passage de `text_en` déjà présent dans la traduction précédente, ou None."""
    def tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    before, current = tokens(previous_en), tokens(text_en)
    seen = {tuple(before[i : i + REPEAT_WORDS]) for i in range(len(before) - REPEAT_WORDS + 1)}
    for i in range(len(current) - REPEAT_WORDS + 1):
        if tuple(current[i : i + REPEAT_WORDS]) in seen:
            return " ".join(current[i : i + REPEAT_WORDS])
    return None


def review(text_fr: str, text_en: str, glossary: Glossary, previous_en: str = "") -> list[str]:
    """Problèmes détectés, formulés comme des consignes de correction."""
    issues = [
        f"« {term.fr} » doit être traduit par « {term.en} »."
        for term in missing_terms(text_fr, text_en, glossary)
    ]
    names = missing_names(text_fr, text_en)
    if names:
        issues.append(f"Des noms cités ont disparu : {', '.join(names)}. Garde-les tels quels.")
    lost = missing_numbers(text_fr, text_en)
    if lost:
        issues.append(f"Des nombres cités ont disparu : {', '.join(lost)}. Garde-les.")
    if is_over_condensed(text_fr, text_en):
        issues.append(
            "Tu as trop résumé : garde toutes les informations (éléments d'une liste, "
            "résultat affiché, conclusion). Supprime seulement les hésitations."
        )
    repeated = repeated_passage(previous_en, text_en)
    if repeated:
        issues.append(
            f"« {repeated} » reprend la phrase précédente, déjà traduite : "
            "traduis seulement le texte demandé."
        )
    return issues
