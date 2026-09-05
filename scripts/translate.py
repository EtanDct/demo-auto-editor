"""Étape C : traduction et conducteur maître (plan-technique.md, section 3).

Traduit chaque segment français en anglais via le LLM local, en s'appuyant
sur le glossaire SAP pour une terminologie cohérente, et écrit
data/edit_decision_list.yaml.

Le champ `visual_action` est laissé vide (null) : les coordonnées de zoom /
highlight sont définies manuellement dans le fichier généré, comme recommandé
dans le plan technique pour une interface SAP stable (section 6).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml

from llm_client import load_llm
from pipeline_config import PROJECT_ROOT, PipelineConfig, load_config
from schemas import EditDecision, Glossary, NarrationSpec, TranscriptSegment, TranslationResult

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

MAX_RETRIES = 3

SYSTEM_PROMPT = """Tu es un traducteur technique spécialisé SAP Fiori (FR -> EN).
Traduis le texte fourni en anglais professionnel, adapté à une vidéo de démonstration logicielle.
Utilise en priorité les traductions du glossaire fourni pour les termes SAP.
Réponds UNIQUEMENT avec un objet JSON valide de la forme :
{"text_en": "...", "sap_terms": ["..."]}
"sap_terms" liste les termes SAP (en anglais) identifiés dans le texte traduit.
Ne mets aucun texte avant ou après le JSON."""


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
    logger.info(
        "'visual_action' est vide pour chaque segment : à compléter manuellement "
        "(coordonnées normalisées 0-1) avant l'étape 'render'."
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
