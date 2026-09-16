"""Tests du découpage en phrases au mot près (scripts/segmentation.py).

Le cas qui l'a motivé, sur la démo Sales Report : « …une pour le prix » et
« moyen par commande » étaient deux segments, et le second, traduit seul,
devenait « by command ».
"""

from __future__ import annotations

from schemas import TranscriptSegment, TranscriptWord
from segmentation import best_cut, split_into_sentences


def words_from(text: str, start: float = 0.0, step: float = 0.4, pauses: dict | None = None):
    """Un mot toutes les `step` secondes ; `pauses` ajoute un silence après un mot."""
    words, t = [], start
    for index, token in enumerate(text.split()):
        words.append(TranscriptWord(text=token, start=round(t, 3), end=round(t + step * 0.8, 3)))
        t += step + (pauses or {}).get(index, 0.0)
    return words


def whisper(*chunks: list[TranscriptWord]) -> list[TranscriptSegment]:
    """Segments Whisper découpés arbitrairement, comme sur un silence."""
    return [
        TranscriptSegment(
            id=f"w-{i}", start=c[0].start, end=c[-1].end,
            text_fr=" ".join(w.text for w in c), words=c,
        )
        for i, c in enumerate(chunks)
    ]


def texts(segments):
    return [s.text_fr for s in segments]


def test_une_phrase_coupee_par_whisper_est_recousue_au_mot_pres():
    words = words_from("On a trois vignettes, une pour le prix moyen par commande. Ensuite le tableau.")
    segments = whisper(words[:8], words[8:])  # coupure entre « prix » et « moyen »

    result = split_into_sentences(segments, max_seconds=14.0, min_seconds=1.0)

    assert texts(result) == [
        "On a trois vignettes, une pour le prix moyen par commande.",
        "Ensuite le tableau.",
    ]


def test_les_horaires_suivent_les_mots_et_non_les_segments_whisper():
    words = words_from("Voici la page d'accueil du rapport. Puis on filtre les commandes par région.")
    result = split_into_sentences(whisper(words[:4], words[4:]), max_seconds=14.0, min_seconds=1.0)

    assert result[0].start == words[0].start
    assert result[0].end == words[5].end   # « rapport. »
    assert result[1].start == words[6].start


def test_une_phrase_trop_longue_est_coupee_sur_sa_virgule():
    text = (
        "Le tableau montre le numéro de commande le produit et la catégorie, "
        "puis la région le client la date le montant et le statut."
    )
    result = split_into_sentences(whisper(words_from(text)), max_seconds=6.0)

    assert len(result) == 2
    assert result[0].text_fr.endswith("catégorie,")
    assert result[1].text_fr.startswith("puis")


def test_sans_virgule_la_coupure_tombe_sur_le_plus_long_silence():
    text = "on filtre par région on prend la région APAC et on regarde ce qui remonte"
    # Long silence après « APAC » (index 8).
    result = split_into_sentences(whisper(words_from(text, pauses={8: 1.2})), max_seconds=5.0)

    assert result[0].text_fr.endswith("APAC")


def test_aucune_coupure_ne_laisse_un_fragment_trop_court():
    words = words_from("Donc, on va regarder maintenant tout le détail du tableau des commandes client")

    cut = best_cut(words, min_side_seconds=1.25)

    # « Donc, » a une virgule, mais le couper seul donnerait un plan d'une
    # demi-seconde : la coupure retenue laisse de la matière des deux côtés.
    assert cut is not None and cut >= 3


def test_une_phrase_tres_courte_rejoint_la_suivante():
    words = words_from("Voilà. On passe maintenant au tableau des commandes du client.")

    result = split_into_sentences(whisper(words), max_seconds=14.0, min_seconds=2.5)

    assert texts(result) == ["Voilà. On passe maintenant au tableau des commandes du client."]


def test_une_derniere_phrase_tres_courte_rejoint_la_precedente():
    words = words_from("Voici la fin de la démonstration du rapport des ventes. Merci.")

    result = split_into_sentences(whisper(words), max_seconds=14.0, min_seconds=2.5)

    assert len(result) == 1


def test_les_segments_sont_renumerotes_dans_l_ordre():
    words = words_from("Première phrase assez longue pour rester seule. Deuxième phrase assez longue aussi.")

    result = split_into_sentences(whisper(words), max_seconds=14.0, min_seconds=1.0)

    assert [s.id for s in result] == ["seg-001", "seg-002"]


def test_sans_horodatage_des_mots_on_retombe_sur_le_recollage():
    """Une transcription produite avant ce changement reste utilisable."""
    segments = [
        TranscriptSegment(id="a", start=0.0, end=5.0, text_fr="on se retrouve sur la page"),
        TranscriptSegment(id="b", start=5.0, end=9.0, text_fr="d'accueil et voilà."),
    ]

    result = split_into_sentences(segments, max_seconds=30.0)

    assert texts(result) == ["on se retrouve sur la page d'accueil et voilà."]


def test_aucun_mot_n_est_perdu_ni_duplique():
    text = (
        "Bonjour à tous. On a trois vignettes, une pour les revenus, une pour le nombre de "
        "commandes et une pour le prix moyen par commande. Voilà. Ensuite on descend un peu "
        "et on voit le tableau des commandes avec le numéro le produit la catégorie la région "
        "le client la date le montant et le statut. Merci."
    )
    words = words_from(text)
    segments = whisper(words[:7], words[7:19], words[19:40], words[40:])

    result = split_into_sentences(segments, max_seconds=8.0)

    assert " ".join(texts(result)).split() == text.split()
    assert all(a.end <= b.start for a, b in zip(result, result[1:]))


def test_les_morceaux_d_une_phrase_longue_sont_reassembles_jusqu_a_la_limite():
    """Régression : la coupure en cascade laissait « faire au hasard services, »
    seul, 2,7 s, traduit en charabia. Les morceaux se regroupent tant qu'ils
    tiennent dans la limite."""
    text = (
        "On prend une catégorie, je ne sais pas trop laquelle, faire au hasard services, "
        "qui remonte quatre commandes et on peut trier sur le statut, par exemple ceux "
        "qui sont en cours et on a toutes nos commandes pour la région APAC."
    )
    result = split_into_sentences(whisper(words_from(text)), max_seconds=14.0)

    assert all(s.end - s.start <= 14.0 for s in result)
    assert not any(s.text_fr == "faire au hasard services," for s in result)
    assert len(result) == 2
