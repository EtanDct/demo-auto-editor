"""Modèles Pydantic partagés par toutes les étapes du pipeline.

Ces schémas sont la source de vérité pour le format des fichiers
intermédiaires (transcript_fr.json, edit_decision_list.yaml,
narration_manifest.json) décrits dans docs/plan-technique.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    """Un segment de la transcription française horodatée (étape B)."""

    id: str
    start: float = Field(ge=0)
    end: float
    text_fr: str

    @model_validator(mode="after")
    def check_order(self) -> "TranscriptSegment":
        if self.end <= self.start:
            raise ValueError(f"{self.id}: end ({self.end}) must be after start ({self.start})")
        return self


VisualActionType = Literal["zoom", "highlight", "callout", "popup", "cursor_emphasis"]


class VisualAction(BaseModel):
    """Annotation visuelle associée à un segment (étape C / F).

    Les coordonnées x, y, width, height sont normalisées entre 0 et 1,
    relatives à l'image, pour rester valides quelle que soit la résolution.
    """

    type: VisualActionType
    target: str
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class NarrationSpec(BaseModel):
    voice: str
    pause_before_ms: int = Field(default=0, ge=0)
    pause_after_ms: int = Field(default=0, ge=0)


class EditDecision(BaseModel):
    """Une entrée du conducteur de montage `edit_decision_list.yaml` (étape C)."""

    id: str
    source_start: float = Field(ge=0)
    source_end: float
    text_fr: str
    text_en: str
    sap_terms: list[str] = Field(default_factory=list)
    visual_action: VisualAction | None = None
    narration: NarrationSpec

    @model_validator(mode="after")
    def check_order(self) -> "EditDecision":
        if self.source_end <= self.source_start:
            raise ValueError(
                f"{self.id}: source_end ({self.source_end}) must be after source_start ({self.source_start})"
            )
        return self


class NarrationManifestEntry(BaseModel):
    """Une entrée de `narration_manifest.json` (étape D)."""

    segment_id: str
    audio_file: str
    duration: float = Field(gt=0)
    provider: str
    voice: str


class GlossaryTerm(BaseModel):
    fr: str
    en: str


class Glossary(BaseModel):
    terms: list[GlossaryTerm] = Field(default_factory=list)
