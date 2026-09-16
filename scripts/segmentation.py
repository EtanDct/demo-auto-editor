"""Découpage de la transcription en phrases, avant traduction (étape C).

Whisper découpe sur les silences, pas sur la syntaxe. Traduits isolément, ces
morceaux perdent leur sens : sur la démo Sales Report, « …une pour le prix » et
« moyen par commande » étaient deux segments, et le second devenait « by
command ». Le recollage à la ponctuation ne suffisait pas : plafonné à
`max_segment_seconds` par segment Whisper entier, il coupait encore en pleine
phrase dès qu'une phrase débordait (« le bouton clear à | droite »).

Avec l'horodatage de chaque mot, on découpe au mot près :

1. la transcription est remise à plat en une suite de mots ;
2. on coupe après chaque mot qui termine une phrase (`.`, `!`, `?`, `…`) ;
3. une phrase trop longue est recoupée à ses meilleures frontières internes —
   virgules, deux-points, silences — puis ses morceaux sont réassemblés tant
   qu'ils tiennent dans la limite. Sans ce réassemblage, la coupure en cascade
   laissait des bribes comme « faire au hasard services, » (2,7 s), qu'aucune
   traduction ne rend intelligibles ;
4. une phrase très courte (« Voilà. ») rejoint la suivante tant que l'ensemble
   tient dans la limite, pour ne pas produire des plans d'une demi-seconde.

Une transcription sans horodatage des mots (produite avant ce changement)
retombe sur le recollage par segments Whisper entiers.
"""

from __future__ import annotations

from schemas import TranscriptSegment, TranscriptWord

SENTENCE_END = (".", "!", "?", "…", ":")
# Le deux-points termine un segment Whisper, mais pas une phrase au sens où on
# l'entend ici : « trois vignettes : une pour… » annonce la suite.
STRONG_END = (".", "!", "?", "…")
SOFT_BREAK = (",", ";", ":")


def merge_into_sentences(
    segments: list[TranscriptSegment], max_seconds: float
) -> list[TranscriptSegment]:
    """Recolle les segments que Whisper a coupés en pleine phrase.

    Secours pour les transcriptions sans horodatage des mots : un segment est
    prolongé tant qu'il ne se termine pas sur une ponctuation forte, dans la
    limite de `max_seconds` (au-delà, le sous-titre serait illisible et le
    recalage sans marge).
    """
    merged: list[TranscriptSegment] = []
    for segment in sorted(segments, key=lambda s: s.start):
        previous = merged[-1] if merged else None
        joinable = (
            previous is not None
            and not previous.text_fr.rstrip().endswith(SENTENCE_END)
            and segment.end - previous.start <= max_seconds
        )
        if joinable:
            merged[-1] = TranscriptSegment(
                id=previous.id,
                start=previous.start,
                end=segment.end,
                text_fr=f"{previous.text_fr.rstrip()} {segment.text_fr.lstrip()}",
            )
        else:
            merged.append(segment)

    return _renumber(merged)


def _renumber(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    return [
        s.model_copy(update={"id": f"seg-{i + 1:03d}"}) for i, s in enumerate(segments)
    ]


def _to_segment(words: list[TranscriptWord]) -> TranscriptSegment:
    return TranscriptSegment(
        id="seg-000",
        start=words[0].start,
        # Un mot isolé peut être horodaté sans durée : le schéma exige une fin
        # postérieure au début.
        end=max(words[-1].end, words[0].start + 0.05),
        text_fr=" ".join(w.text.strip() for w in words if w.text.strip()),
        words=words,
    )


def _duration(words: list[TranscriptWord]) -> float:
    return words[-1].end - words[0].start


def best_cut(words: list[TranscriptWord], min_side_seconds: float) -> int | None:
    """Index après lequel couper une phrase trop longue, ou None.

    Une coupure vaut d'autant plus que le silence qui suit le mot est long, et
    qu'elle tombe sur une virgule ou un point-virgule. Elle doit laisser au
    moins `min_side_seconds` de chaque côté : couper « Donc, » seul en tête
    produirait un plan inutilisable.
    """
    best, best_score = None, 0.0
    for index in range(len(words) - 1):
        left, right = words[: index + 1], words[index + 1 :]
        if _duration(left) < min_side_seconds or _duration(right) < min_side_seconds:
            continue
        pause = max(0.0, words[index + 1].start - words[index].end)
        score = pause + (0.6 if words[index].text.rstrip().endswith(SOFT_BREAK) else 0.0)
        # À score égal, la coupure la plus centrée : deux moitiés équilibrées
        # laissent de la marge au recalage des deux côtés.
        balance = abs(_duration(left) - _duration(right)) / max(_duration(words), 1e-6)
        score -= 0.05 * balance
        if best is None or score > best_score:
            best, best_score = index, score
    return best


def _fit(words: list[TranscriptWord], max_seconds: float, min_side_seconds: float):
    if _duration(words) <= max_seconds or len(words) < 2:
        return [words]
    cut = best_cut(words, min_side_seconds)
    if cut is None:
        # Aucune coupure ne laisse deux côtés utilisables : mieux vaut un
        # segment un peu long qu'un fragment de mot isolé.
        return [words]
    return _fit(words[: cut + 1], max_seconds, min_side_seconds) + _fit(
        words[cut + 1 :], max_seconds, min_side_seconds
    )


def _pack(parts: list[list[TranscriptWord]], max_seconds: float) -> list[list[TranscriptWord]]:
    """Réassemble les morceaux consécutifs d'une phrase tant qu'ils tiennent."""
    packed: list[list[TranscriptWord]] = []
    for part in parts:
        if packed and _duration(packed[-1] + part) <= max_seconds:
            packed[-1] = packed[-1] + part
        else:
            packed.append(part)
    return packed


def split_into_sentences(
    segments: list[TranscriptSegment],
    max_seconds: float,
    min_seconds: float = 2.5,
) -> list[TranscriptSegment]:
    """Segments de traduction alignés sur les phrases, au mot près."""
    ordered = sorted(segments, key=lambda s: s.start)
    if not ordered or any(not s.words for s in ordered):
        return merge_into_sentences(ordered, max_seconds)

    words = [w for s in ordered for w in s.words if w.text.strip()]
    sentences: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []
    for word in words:
        current.append(word)
        if word.text.rstrip().endswith(STRONG_END):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    pieces = [
        part
        for sentence in sentences
        for part in _pack(_fit(sentence, max_seconds, min_side_seconds=min_seconds / 2), max_seconds)
    ]

    # Une phrase très courte rejoint la suivante si l'ensemble tient.
    merged: list[list[TranscriptWord]] = []
    for piece in pieces:
        if merged and _duration(merged[-1]) < min_seconds and _duration(merged[-1] + piece) <= max_seconds:
            merged[-1] = merged[-1] + piece
        else:
            merged.append(piece)
    if len(merged) > 1 and _duration(merged[-1]) < min_seconds:
        if _duration(merged[-2] + merged[-1]) <= max_seconds:
            merged[-2:] = [merged[-2] + merged[-1]]

    return _renumber([_to_segment(piece) for piece in merged])
