"""Interprétation de ce que le narrateur désigne à l'écran.

Deux services, partagés par l'étape de traduction (qui produit une cible) et par
l'étape d'index du texte à l'écran (qui produit les candidats) :

1. **Normalisation des libellés.** L'OCR et la transcription n'écrivent pas de
   la même façon : casse, accents et espaces varient. Le modèle OCR par défaut
   restitue d'ailleurs les libellés français sans accents ("Ecritures a
   controler"), et colle les mots de façon instable selon le crénage
   ("Toprepositories"). Les deux côtés de la comparaison sont donc ramenés à la
   même forme.

2. **Rejet des désignations non spécifiques.** C'est la règle qui évite
   d'encadrer n'importe quoi. Quand le narrateur dit « en haut à gauche » ou
   « on clique sur le bouton », il ne nomme aucun élément : chercher ces mots à
   l'écran trouverait n'importe quel libellé contenant « haut », ou n'importe
   quel bouton. Sur l'extrait de référence, « tout en haut à gauche » aurait
   pointé « Top repositories », qui n'a rien à voir. Une désignation n'est
   retenue que si, une fois retirés les mots de position, les mots de catégorie
   et les ordinaux, il reste quelque chose : c'est ce reste qui identifie
   vraiment un élément.
"""

from __future__ import annotations

import re
import unicodedata

# Mots de position : décrivent où regarder, jamais quoi regarder.
SPATIAL_WORDS = {
    "haut", "bas", "gauche", "droite", "milieu", "centre", "coin", "cote", "bord",
    "dessus", "dessous", "ici", "la", "zone", "partie", "endroit", "ecran", "page",
    "top", "bottom", "left", "right", "middle", "center", "centre", "corner", "side",
    "here", "there", "area", "part", "screen", "upper", "lower", "above", "below",
}

# Noms de catégorie : désignent un type d'élément, pas un élément.
GENERIC_UI_NOUNS = {
    "bouton", "menu", "icone", "lien", "champ", "onglet", "fenetre", "barre",
    "liste", "case", "encadre", "element", "section", "colonne", "ligne", "titre",
    "deroulant", "popup", "fleche", "curseur", "image", "logo", "texte",
    "button", "icon", "link", "field", "tab", "window", "bar", "list", "box",
    "element", "item", "column", "row", "title", "dropdown", "arrow", "cursor",
    "text", "label", "panel", "sidebar", "header", "footer", "toolbar", "widget",
}

# Ordinaux et déterminants : « le second menu » ne nomme toujours rien.
ORDINAL_WORDS = {
    "premier", "premiere", "deuxieme", "second", "seconde", "troisieme", "dernier",
    "derniere", "prochain", "prochaine", "precedent", "precedente", "autre", "meme",
    "first", "next", "previous", "last", "third", "fourth", "other", "same", "another",
}

# Articles, prépositions et liaisons : jamais porteurs d'identité.
FILLER_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "au", "aux", "en",
    "sur", "sous", "dans", "a", "et", "ou", "ce", "cet", "cette", "ces", "son", "sa",
    "the", "a", "an", "of", "on", "in", "at", "to", "and", "or", "this", "that",
    "these", "those", "its", "your", "our",
}

NON_IDENTIFYING_WORDS = SPATIAL_WORDS | GENERIC_UI_NOUNS | ORDINAL_WORDS | FILLER_WORDS


def normalize_text(text: str) -> str:
    """Forme lisible canonique : sans casse, sans accents, espaces normalisés."""
    stripped = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(c for c in stripped if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", without_accents)


def match_key(text: str) -> str:
    """Clé d'identité d'un libellé, espaces compris.

    L'OCR colle les mots de façon instable d'une frame à l'autre selon le
    crénage : sur l'extrait de référence, le même libellé ressort tantôt
    "Top repositories", tantôt "Toprepositories". Les traiter comme deux
    éléments distincts fragmente l'index et fabrique de fausses ambiguïtés,
    donc la clé ignore les espaces. Là où elle sert au regroupement, la fusion
    reste bornée par le recouvrement des boîtes.
    """
    return normalize_text(text).replace(" ", "")


def identifying_tokens(label: str) -> list[str]:
    """Mots du libellé qui l'identifient vraiment, une fois le décor retiré."""
    words = re.findall(r"[a-z0-9]+", normalize_text(label))
    return [w for w in words if w not in NON_IDENTIFYING_WORDS]


def is_specific_label(label: str | None) -> bool:
    """Le libellé désigne-t-il un élément précis, ou seulement une position/catégorie ?

    "Post", "Pull requests", "Top repositories" -> oui.
    "en haut à gauche", "le bouton", "le second menu" -> non.
    """
    return bool(label and identifying_tokens(label))
