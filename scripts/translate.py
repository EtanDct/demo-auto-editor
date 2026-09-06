"""Étape C : traduction et conducteur maître (plan-technique.md, section 3).

Traduit chaque segment français en anglais via le LLM local, en s'appuyant
sur le glossaire SAP pour une terminologie cohérente, et écrit
data/edit_decision_list.yaml.

Le LLM déclare aussi, pour chaque segment, ce que le narrateur désigne à
l'écran (`ui_reference`) : un élément nommé par son libellé, une simple
position, ou rien. C'est le tri sémantique qui rend le montage automatique
possible — chercher les mots du narrateur directement dans le texte de l'écran
pointerait « Top repositories » dès qu'il dit « en haut ». Ce que le modèle
annonce est re-filtré par `ui_reference.is_specific_label`, un 3B annonçant
volontiers un élément nommé pour « le bouton ».

Le champ `visual_action` reste vide (null) : les coordonnées de zoom /
highlight sont définies manuellement dans le fichier généré, en attendant
l'étape d'appariement qui les dérivera de `ui_reference` et de l'index du
texte à l'écran.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml

from llm_client import load_llm
from pipeline_config import PROJECT_ROOT, PipelineConfig, load_config
from schemas import (
    EditDecision,
    Glossary,
    IntroText,
    NarrationSpec,
    TranscriptSegment,
    TranslationResult,
    UiReference,
)
from ui_reference import is_specific_label

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

MAX_RETRIES = 3

SYSTEM_PROMPT = """Tu adaptes en anglais la narration d'une vidéo de démonstration SAP Fiori.

Ce n'est pas une traduction mot à mot : c'est une réécriture pour une voix off.

1. Le texte anglais doit être BREF et NATUREL, tel qu'un présentateur le dirait.
   - supprime hésitations, répétitions, bafouillages, « donc », « alors »,
     « en fait », « voilà », « etc. », et les reprises de la même idée ;
   - une phrase claire vaut mieux que deux phrases hésitantes ;
   - vise plus court que l'original : la voix de synthèse est plus lente.
   - garde le sens et les termes SAP, en priorité ceux du glossaire fourni.

2. Classe ce que le narrateur désigne à l'écran :
   - "named_control" : il cite le LIBELLÉ d'un élément (bouton, onglet, champ,
     entrée de menu). Recopie ce libellé seul dans "ui_target", sans mot de
     position ni mot de catégorie.
   - "spatial" : il indique une position ou une catégorie sans citer de
     libellé. "ui_target" vaut null.
   - "none" : il ne montre rien. "ui_target" vaut null.

Exemples d'adaptation :
  "Alors donc euh, on se retrouve sur la page d'accueil, la page d'accueil de Github"
     -> "Here's the GitHub home page."
  "et qu'est-ce qu'on peut y retrouver ? Donc sur la gauche, on y retrouve un menu"
     -> "On the left, there's a menu."
  "cliquez sur le bouton Enregistrer pour valider"
     -> "Click Save to confirm."  (named_control, ui_target "Enregistrer")

Exemples de classement :
  "cliquez sur le bouton Enregistrer"        -> named_control, ui_target "Enregistrer"
  "ouvrez l'onglet Écritures à contrôler"    -> named_control, ui_target "Écritures à contrôler"
  "tout en haut à gauche il y a un bouton"   -> spatial, ui_target null
  "bonjour à tous et bienvenue"              -> none, ui_target null

Règle d'arbitrage : si un libellé est cité, c'est "named_control", même si la
phrase donne aussi une position.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{"reference_kind": <"named_control" ou "spatial" ou "none">,
 "ui_target": <le libellé cité, ou null>,
 "text_en": <l'adaptation anglaise, brève et naturelle>,
 "sap_terms": [<termes SAP anglais identifiés>]}"""


def load_transcript(path: Path) -> list[TranscriptSegment]:
    if not path.exists():
        raise FileNotFoundError(
            f"Transcription introuvable : {path}. Lance d'abord "
            "'python run.py --step transcribe'."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment.model_validate(item) for item in raw]


SENTENCE_END = (".", "!", "?", "…", ":")


def merge_into_sentences(
    segments: list[TranscriptSegment], max_seconds: float
) -> list[TranscriptSegment]:
    """Recolle les segments que Whisper a coupés en pleine phrase.

    Whisper découpe sur les silences, pas sur la syntaxe : sur l'extrait de
    référence, 25 segments sur 34 s'arrêtent en milieu de phrase. Traduits
    isolément, ils donnent des bouts qui ne veulent rien dire — « page » et
    « d'accueil et qu'est-ce qu'on peut y retrouver ? » deviennent deux
    fragments sans rapport. Un segment est donc prolongé tant qu'il ne se
    termine pas sur une ponctuation forte, dans la limite de `max_seconds`
    (au-delà, le sous-titre serait illisible et le recalage sans marge).
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

    return [
        TranscriptSegment(id=f"seg-{i + 1:03d}", start=s.start, end=s.end, text_fr=s.text_fr)
        for i, s in enumerate(merged)
    ]


def load_glossary(path: Path) -> Glossary:
    if not path.exists():
        return Glossary()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Glossary.model_validate(raw)


def _build_user_prompt(text_fr: str, glossary: Glossary) -> str:
    glossary_lines = "\n".join(f"- {t.fr} -> {t.en}" for t in glossary.terms) or "(vide)"
    return f"Glossaire SAP :\n{glossary_lines}\n\nTexte à traduire :\n{text_fr}"


def translate_segment(llm, text_fr: str, glossary: Glossary, temperature: float) -> TranslationResult:
    user_prompt = _build_user_prompt(text_fr, glossary)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"]["content"]
        try:
            return TranslationResult.model_validate_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("Sortie LLM invalide (essai %d/%d) : %s", attempt, MAX_RETRIES, exc)

    raise RuntimeError(
        f"Le LLM n'a pas produit de JSON valide après {MAX_RETRIES} essais : {last_error}"
    )


def resolve_ui_reference(result: TranslationResult, segment_id: str) -> UiReference:
    """Traduit la sortie du LLM en désignation exploitable, ou la rejette.

    Un modèle 3B annonce régulièrement "named_control" avec un libellé qui n'en
    est pas un ("the button", "en haut à gauche", "le second menu"). Ces
    libellés ne désignent aucun élément : les chercher à l'écran trouverait
    n'importe quoi. Un filtre déterministe rattrape donc le modèle, plutôt que
    de compter sur sa seule discipline.
    """
    if result.reference_kind != "named_control":
        return UiReference(kind=result.reference_kind)

    if not is_specific_label(result.ui_target):
        logger.info(
            "%s : cible '%s' rejetée (position ou catégorie, ne nomme aucun élément).",
            segment_id, result.ui_target,
        )
        return UiReference(kind="spatial")

    return UiReference(kind="named_control", label=result.ui_target.strip())


def translate(
    segments: list[TranscriptSegment], glossary: Glossary, config: PipelineConfig
) -> tuple[list[EditDecision], IntroText | None]:
    llm = load_llm(config)
    decisions = []
    for segment in segments:
        logger.info("Traduction de %s", segment.id)
        result = translate_segment(llm, segment.text_fr, glossary, config.llm.temperature)
        decisions.append(
            EditDecision(
                id=segment.id,
                source_start=segment.start,
                source_end=segment.end,
                text_fr=segment.text_fr,
                text_en=result.text_en,
                sap_terms=result.sap_terms,
                ui_reference=resolve_ui_reference(result, segment.id),
                visual_action=None,
                narration=NarrationSpec(voice=config.tts.voice, pause_before_ms=150, pause_after_ms=250),
            )
        )

    intro = None
    try:
        intro = generate_intro_text(llm, segments, config.llm.temperature)
    except (ValueError, KeyError) as exc:
        # Un titre manquant ne doit pas faire échouer la traduction : le rendu
        # se passera simplement de carton.
        logger.warning("Titre d'introduction non généré : %s", exc)

    return decisions, intro


TITLE_PROMPT = """Tu résumes une vidéo de démonstration logicielle en un titre de carton d'introduction.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{"title": <titre en anglais, 2 à 6 mots, sans point final>,
 "subtitle": <sous-titre en anglais, une courte phrase de 3 à 8 mots, ou "">}

Le titre nomme le sujet ; le sous-titre précise ce que la vidéo montre."""


def generate_intro_text(llm, segments: list[TranscriptSegment], temperature: float) -> IntroText:
    """Titre du carton d'introduction, déduit de la transcription.

    Seul le début de la transcription est envoyé : une démonstration annonce son
    sujet dans ses premières phrases, et le contexte du modèle est limité.
    """
    excerpt = " ".join(s.text_fr for s in segments[:8])[:1500]
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": TITLE_PROMPT},
            {"role": "user", "content": f"Début de la narration :\n{excerpt}"},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return IntroText.model_validate_json(response["choices"][0]["message"]["content"])


def write_intro_text(text: IntroText, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Titre d'introduction : %r / %r (%s)", text.title, text.subtitle, out_path)


def write_edl(decisions: list[EditDecision], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [d.model_dump() for d in decisions]
    out_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Conducteur de montage écrit dans %s (%d segments)", out_path, len(decisions))
    named = sum(1 for d in decisions if d.ui_reference and d.ui_reference.kind == "named_control")
    logger.info(
        "%d/%d segments désignent un élément nommé (ui_reference). 'visual_action' reste "
        "vide partout : à compléter manuellement (coordonnées normalisées 0-1) avant "
        "l'étape 'render', en attendant l'appariement automatique.",
        named, len(decisions),
    )


@app.command()
def main(config_path: Path = typer.Option(None, help="Chemin vers config.yaml.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    segments = load_transcript(config.paths.resolve("data_dir") / "transcript_fr.json")
    merged = merge_into_sentences(segments, config.llm.max_segment_seconds)
    if len(merged) < len(segments):
        logger.info(
            "%d segments recollés en %d phrases (Whisper coupe sur les silences, "
            "pas sur la syntaxe).",
            len(segments), len(merged),
        )
    segments = merged
    glossary = load_glossary(PROJECT_ROOT / config.glossary_file)
    decisions, intro = translate(segments, glossary, config)
    write_edl(decisions, config.paths.resolve("data_dir") / "edit_decision_list.yaml")
    if intro is not None:
        write_intro_text(intro, config.paths.resolve("data_dir") / "intro.json")


if __name__ == "__main__":
    app()
