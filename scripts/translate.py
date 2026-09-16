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
# `merge_into_sentences` reste importable d'ici : c'est le secours des
# transcriptions sans horodatage des mots.
from segmentation import merge_into_sentences, split_into_sentences  # noqa: F401
from translation_checks import load_glossary, relevant_terms, review
from pipeline_config import PROJECT_ROOT, PipelineConfig, load_config
from schemas import (
    Chapter,
    ChapterPlan,
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
# Plafond de longueur de chaque réponse. Le mode JSON de llama.cpp laisse le
# modèle émettre des blancs sans fin : sans plafond, une correction de la démo
# Sales Report a tourné neuf minutes sans rendre la main. Une réponse coupée est
# un JSON invalide, que la boucle de relance rattrape. Une traduction de segment
# tient largement en dessous.
MAX_TOKENS = 600

SYSTEM_PROMPT = """Tu adaptes en anglais la narration d'une vidéo de démonstration SAP Fiori.

Ce n'est pas une traduction mot à mot : c'est une réécriture pour une voix off.

1. Le texte anglais doit être BREF et NATUREL, tel qu'un présentateur le dirait.
   - supprime hésitations, répétitions, bafouillages, « donc », « alors »,
     « en fait », « voilà », « etc. », et les reprises de la même idée ;
   - une phrase claire vaut mieux que deux phrases hésitantes ;
   - ne supprime QUE le superflu : chaque information est conservée, y compris
     les éléments d'une énumération, les chiffres cités, le résultat affiché
     à l'écran et les conclusions ;
   - le glossaire fourni est impératif : un terme qui y figure se traduit
     toujours ainsi.

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


def _build_user_prompt(text_fr: str, glossary: Glossary) -> str:
    """Consigne d'un segment : glossaire utile, puis texte à traduire.

    Seuls les termes du glossaire présents dans le texte sont listés : un
    glossaire entier noierait la consigne d'un modèle 3B.

    Pas de contexte des phrases voisines. Essayé sur la démo Sales Report, il
    faisait plus de mal que de bien : le modèle traduisait aussi le contexte, et
    « Here are three tiles… » revenait dans trois segments de suite. Le
    découpage au mot près (scripts/segmentation.py) règle la cause qui
    justifiait ce contexte : les phrases coupées en deux.
    """
    terms = relevant_terms(text_fr, glossary)
    glossary_lines = "\n".join(f"- {t.fr} -> {t.en}" for t in terms) or "(aucun terme concerné)"
    return f"Glossaire (impératif) :\n{glossary_lines}\n\nTexte à traduire :\n{text_fr}"


def translate_segment(
    llm,
    text_fr: str,
    glossary: Glossary,
    temperature: float,
    previous_en: str = "",
    segment_id: str = "",
) -> TranslationResult:
    """Traduction d'un segment, relue et corrigée une fois si besoin.

    La relecture (`translation_checks.review`) est déterministe : terme du
    glossaire manquant, nombre perdu, résumé excessif. Un problème détecté est
    renvoyé au modèle, nommé précisément ; la version corrigée n'est retenue que
    si elle en compte moins. Ce qui reste est journalisé, pas bloquant : une
    traduction imparfaite vaut mieux qu'une étape qui échoue.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text_fr, glossary)},
    ]
    result = _ask(llm, messages, temperature)
    issues = review(text_fr, result.text_en, glossary, previous_en)
    if not issues:
        return result

    logger.info("%s : traduction à corriger (%s)", segment_id, " ; ".join(issues))
    correction = messages + [
        {"role": "assistant", "content": result.model_dump_json()},
        {
            "role": "user",
            "content": "Corrige ta réponse, même format JSON :\n"
            + "\n".join(f"- {issue}" for issue in issues),
        },
    ]
    try:
        corrected = _ask(llm, correction, temperature)
    except RuntimeError as exc:
        logger.warning("%s : correction impossible (%s), première version gardée", segment_id, exc)
        return result
    remaining = review(text_fr, corrected.text_en, glossary, previous_en)
    if len(remaining) < len(issues):
        result, issues = corrected, remaining
    if issues:
        logger.warning("%s : problème(s) restant(s) : %s", segment_id, " ; ".join(issues))
    return result


def _ask(llm, messages: list[dict], temperature: float) -> TranslationResult:
    """Un échange JSON avec le modèle, relancé si la sortie est invalide."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=MAX_TOKENS,
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
) -> tuple[list[EditDecision], IntroText | None, ChapterPlan]:
    llm = load_llm(config)
    decisions = []
    for segment in segments:
        logger.info("Traduction de %s", segment.id)
        previous_en = decisions[-1].text_en if decisions else ""
        result = translate_segment(
            llm, segment.text_fr, glossary, config.llm.temperature, previous_en, segment.id
        )
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
        intro = generate_intro_text(llm, segments, config.llm.temperature, glossary)
    except (ValueError, KeyError) as exc:
        # Un titre manquant ne doit pas faire échouer la traduction : le rendu
        # se passera simplement de carton.
        logger.warning("Titre d'introduction non généré : %s", exc)

    chapters = ChapterPlan()
    if config.layout.enabled and config.layout.chapters_enabled:
        chapters = build_chapters(llm, decisions, config)

    return decisions, intro, chapters


def slice_into_chapters(
    decisions: list[EditDecision], chapter_seconds: float, max_chapters: int
) -> list[list[EditDecision]]:
    """Découpe la narration en tranches consécutives de durées voisines.

    Les frontières sont calculées, pas demandées au modèle. Sondé sur la démo
    Sales Report : prié de choisir lui-même où couper, un 3B faisait un chapitre
    par phrase (15 sur 15) ; prié d'en faire exactement quatre, il rendait des
    frontières dans le désordre, ou un dernier chapitre de 11 phrases sur 15.
    Nommer des tranches déjà découpées, il le fait bien.

    Chaque phrase va à la tranche où tombe son milieu : une phrase longue à
    cheval sur deux tranches ne fausse pas le partage.
    """
    total = sum(d.source_end - d.source_start for d in decisions)
    count = min(max_chapters, round(total / chapter_seconds)) if chapter_seconds > 0 else 0
    if count < 2 or len(decisions) < 2 * count:
        return []

    slices: list[list[EditDecision]] = [[] for _ in range(count)]
    elapsed = 0.0
    for decision in decisions:
        length = decision.source_end - decision.source_start
        slices[min(count - 1, int((elapsed + length / 2) / total * count))].append(decision)
        elapsed += length
    return [s for s in slices if s]


def chapters_prompt(count: int) -> str:
    return f"""You name the parts of a software demo video. You get {count} parts, in order.
Give each part a title of 1 to 3 English words naming what is SHOWN in it.
Never use generic titles such as "Introduction", "Overview", "Conclusion", "Demo", "Part 2".

Answer ONLY with valid JSON, without any text before or after:
{{"titles": [<title of part 1>, ..., <title of part {count}>]}}"""


def generate_chapter_titles(llm, slices: list[list[EditDecision]]) -> list[str]:
    """Un titre par tranche, à température nulle : les chapitres doivent être
    reproductibles d'un passage à l'autre. La narration adaptée plutôt que la
    transcription : les titres s'affichent en anglais, et un texte déjà épuré de
    ses hésitations se résume mieux."""
    listing = "\n\n".join(
        f"Part {index}:\n" + "\n".join(d.text_en[:160] for d in part)
        for index, part in enumerate(slices, 1)
    )
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": chapters_prompt(len(slices))},
            {"role": "user", "content": listing},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS,
    )
    titles = json.loads(response["choices"][0]["message"]["content"])["titles"]
    if not isinstance(titles, list):
        raise ValueError(f"'titles' n'est pas une liste : {titles!r}")
    return [str(title) for title in titles]


def assemble_chapters(
    titles: list[str], slices: list[list[EditDecision]], max_chars: int
) -> ChapterPlan:
    """Associe les titres aux tranches, ou refuse.

    Un titre en trop se laisse tomber ; un titre manquant ou vide ne se devine
    pas, et un chapitre sans nom afficherait un index seul : on refuse tout, la
    bande affichera le titre de la vidéo.
    """
    cleaned = [_shorten(title, max_chars) for title in titles[: len(slices)]]
    if len(cleaned) < len(slices) or not all(cleaned):
        return ChapterPlan()
    return ChapterPlan(
        chapters=[
            Chapter(title=title, first_segment=part[0].id)
            for title, part in zip(cleaned, slices)
        ]
    )


def _shorten(title: str, max_chars: int) -> str:
    """Titre nettoyé et borné, coupé entre deux mots."""
    title = " ".join(title.split()).strip(" .:;-\"'")
    if len(title) <= max_chars:
        return title
    return title[:max_chars].rsplit(" ", 1)[0].strip(" .:;-")


def write_chapters(plan: ChapterPlan, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    if plan.chapters:
        logger.info(
            "Chapitres : %s (%s)", " | ".join(c.title for c in plan.chapters), out_path
        )
    else:
        logger.info("Aucun chapitre retenu : la bande du haut affichera le titre (%s)", out_path)


def build_chapters(llm, decisions: list[EditDecision], config: PipelineConfig) -> ChapterPlan:
    """Chapitres validés, ou aucun : un échec ne doit pas faire tomber l'étape."""
    slices = slice_into_chapters(
        decisions, config.layout.chapter_seconds, config.layout.max_chapters
    )
    if not slices:
        logger.info("Vidéo trop courte pour la découper en chapitres.")
        return ChapterPlan()
    try:
        titles = generate_chapter_titles(llm, slices)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("Titres de chapitres non générés : %s", exc)
        return ChapterPlan()
    plan = assemble_chapters(titles, slices, config.layout.max_chapter_chars)
    if not plan.chapters:
        logger.warning(
            "Titres de chapitres refusés (%d pour %d tranches) : %s",
            len(titles), len(slices), titles,
        )
    return plan


TITLE_PROMPT = """Tu résumes une vidéo de démonstration logicielle en un titre de carton d'introduction.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{"title": <titre en anglais, 2 à 6 mots, sans point final>,
 "subtitle": <sous-titre en anglais, une courte phrase de 3 à 8 mots, ou "">}

Le titre nomme le sujet ; le sous-titre précise ce que la vidéo montre."""


def generate_intro_text(
    llm, segments: list[TranscriptSegment], temperature: float, glossary: Glossary | None = None
) -> IntroText:
    """Titre du carton d'introduction, déduit de la transcription.

    Seul le début de la transcription est envoyé : une démonstration annonce son
    sujet dans ses premières phrases, et le contexte du modèle est limité. Le
    glossaire s'applique aussi au titre : sans lui, le carton annonçait « Three
    Vignettes & Command Details ».
    """
    excerpt = " ".join(s.text_fr for s in segments[:8])[:1500]
    terms = relevant_terms(excerpt, glossary) if glossary else []
    glossary_block = (
        "Glossaire (impératif) :\n" + "\n".join(f"- {t.fr} -> {t.en}" for t in terms) + "\n\n"
        if terms else ""
    )
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": TITLE_PROMPT},
            {"role": "user", "content": f"{glossary_block}Début de la narration :\n{excerpt}"},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS,
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
def main(
    config_path: Path = typer.Option(None, help="Chemin vers config.yaml."),
    chapters_only: bool = typer.Option(
        False,
        "--chapters-only",
        help="Ne régénérer que les chapitres, à partir du conducteur existant.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(config_path)
    data_dir = config.paths.resolve("data_dir")
    if chapters_only:
        from build_narration import load_edl

        decisions = load_edl(data_dir / "edit_decision_list.yaml")
        write_chapters(build_chapters(load_llm(config), decisions, config), data_dir / "chapters.json")
        return

    segments = load_transcript(config.paths.resolve("data_dir") / "transcript_fr.json")
    sentences = split_into_sentences(segments, config.llm.max_segment_seconds)
    logger.info(
        "%d segments Whisper redécoupés en %d phrases (%s).",
        len(segments), len(sentences),
        "au mot près" if all(s.words for s in segments) else
        "par segments entiers : transcription sans horodatage des mots, relance --step transcribe",
    )
    segments = sentences
    glossary = load_glossary(PROJECT_ROOT / config.glossary_file)
    decisions, intro, chapters = translate(segments, glossary, config)
    write_edl(decisions, data_dir / "edit_decision_list.yaml")
    if intro is not None:
        write_intro_text(intro, data_dir / "intro.json")
    write_chapters(chapters, data_dir / "chapters.json")


if __name__ == "__main__":
    app()
