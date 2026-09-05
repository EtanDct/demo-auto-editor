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
    NarrationSpec,
    TranscriptSegment,
    TranslationResult,
    UiReference,
)
from ui_reference import is_specific_label

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

MAX_RETRIES = 3

SYSTEM_PROMPT = """Tu es un traducteur technique spécialisé SAP Fiori (FR -> EN).

Pour chaque texte, tu produis DEUX choses.

1. La traduction anglaise professionnelle, adaptée à une vidéo de démonstration
   logicielle, en utilisant en priorité le glossaire fourni.

2. Le classement de ce que le narrateur désigne à l'écran :
   - "named_control" : le narrateur cite le LIBELLÉ d'un élément (bouton,
     onglet, champ, entrée de menu). Recopie ce libellé seul dans "ui_target",
     sans mot de position ni mot de catégorie.
   - "spatial" : le narrateur indique une position ou une catégorie sans citer
     de libellé. "ui_target" vaut null.
   - "none" : le narrateur ne montre rien à l'écran. "ui_target" vaut null.

Exemples de classement :
  "cliquez sur le bouton Enregistrer"        -> named_control, ui_target "Enregistrer"
  "ouvrez l'onglet Écritures à contrôler"    -> named_control, ui_target "Écritures à contrôler"
  "le champ Société est en haut à droite"    -> named_control, ui_target "Société"
  "on y retrouve les repositories, les projets" -> named_control, ui_target "Repositories"
  "tout en haut à gauche il y a un bouton"   -> spatial, ui_target null
  "sur la gauche on retrouve un menu"        -> spatial, ui_target null
  "bonjour à tous et bienvenue"              -> none, ui_target null

Règle d'arbitrage : si un libellé est cité, c'est "named_control", même si la
phrase donne aussi une position. Si aucun libellé n'est cité mais qu'un élément
est montré, c'est "spatial".

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après :
{"reference_kind": <"named_control" ou "spatial" ou "none">,
 "ui_target": <le libellé cité, ou null>,
 "text_en": <la traduction anglaise>,
 "sap_terms": [<termes SAP anglais identifiés>]}"""


def load_transcript(path: Path) -> list[TranscriptSegment]:
    if not path.exists():
        raise FileNotFoundError(
            f"Transcription introuvable : {path}. Lance d'abord "
            "'python run.py --step transcribe'."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment.model_validate(item) for item in raw]


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
) -> list[EditDecision]:
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
    return decisions


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
    glossary = load_glossary(PROJECT_ROOT / config.glossary_file)
    decisions = translate(segments, glossary, config)
    write_edl(decisions, config.paths.resolve("data_dir") / "edit_decision_list.yaml")


if __name__ == "__main__":
    app()
