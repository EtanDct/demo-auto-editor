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
    # Fenêtre d'affichage, en secondes depuis le début du segment. None = tout
    # le segment. Encadrer un élément pendant qu'on parle d'autre chose se voit
    # tout de suite ; c'est ce que ces deux champs permettent d'éviter.
    start_offset: float | None = Field(default=None, ge=0)
    end_offset: float | None = Field(default=None, gt=0)


    @model_validator(mode="after")
    def check_window(self) -> "VisualAction":
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError(
                f"{self.target}: end_offset ({self.end_offset}) doit suivre "
                f"start_offset ({self.start_offset})"
            )
        return self


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
    ui_reference: UiReference | None = None
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


UiReferenceKind = Literal["named_control", "spatial", "none"]


class UiReference(BaseModel):
    """Ce que le narrateur désigne à l'écran pendant un segment (étape C).

    `kind` distingue les trois cas qui appellent des suites différentes :

    - `named_control` : un élément nommé, dont `label` porte le libellé tel
      qu'il est censé être affiché — le seul cas exploitable pour poser une
      incrustation ;
    - `spatial` : une position ("en haut à gauche") ou une catégorie ("le
      bouton"), qui ne désigne rien de précis ;
    - `none` : le narrateur ne montre rien.

    Distinguer `spatial` de `none` n'est pas cosmétique : c'est ce qui permet de
    mesurer si le modèle rate des cibles ou s'il n'y en avait pas.
    """

    kind: UiReferenceKind
    label: str | None = None

    @model_validator(mode="after")
    def check_label(self) -> "UiReference":
        if self.kind == "named_control" and not (self.label or "").strip():
            raise ValueError("named_control exige un label non vide")
        if self.kind != "named_control" and self.label:
            raise ValueError(f"un label n'a pas de sens avec kind={self.kind}")
        return self


class TranslationResult(BaseModel):
    """Sortie JSON attendue du LLM local pour un segment (étape C).

    Volontairement plate : c'est un modèle 3B quantifié qui la produit, et une
    structure imbriquée multiplie les sorties invalides.
    """

    text_en: str
    sap_terms: list[str] = Field(default_factory=list)
    ui_target: str | None = None
    reference_kind: UiReferenceKind = "none"


class GlossaryTerm(BaseModel):
    fr: str
    en: str


class Glossary(BaseModel):
    terms: list[GlossaryTerm] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    """Une entrée retimée de `timeline.json` (étape E)."""

    id: str
    source_start: float
    source_end: float
    new_start: float
    new_end: float
    narration_duration: float
    audio_speed_factor: float = 1.0
    extended: bool = False
    needs_review: bool = False


class TimelineReport(BaseModel):
    entries: list[TimelineEntry]
    warnings: list[str] = Field(default_factory=list)


class BoundingBox(BaseModel):
    """Rectangle normalisé entre 0 et 1, relatif à l'image.

    Même convention que `VisualAction` : indépendant de la résolution, donc
    directement transposable en expressions FFmpeg à l'étape de rendu.
    """

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def iou(self, other: "BoundingBox") -> float:
        """Intersection sur union : sert à décider si deux détections OCR de
        frames différentes désignent le même élément d'interface."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        return intersection / (self.area + other.area - intersection)


class ScreenElement(BaseModel):
    """Un élément de texte stable à l'écran, agrégé sur plusieurs frames.

    Produit par l'étape `screen` : c'est l'inventaire de ce qui est affiché et
    quand, sur lequel s'appuiera l'appariement entre ce que dit le narrateur et
    ce que montre l'écran.
    """

    id: str
    text: str
    box: BoundingBox
    first_seen: float = Field(ge=0)
    last_seen: float
    confidence: float = Field(ge=0, le=1)
    occurrences: int = Field(gt=0)

    @model_validator(mode="after")
    def check_order(self) -> "ScreenElement":
        if self.last_seen < self.first_seen:
            raise ValueError(
                f"{self.id}: last_seen ({self.last_seen}) précède first_seen ({self.first_seen})"
            )
        return self

    def visible_at(self, start: float, end: float) -> float:
        """Fraction de l'intervalle [start, end] pendant laquelle l'élément est affiché."""
        if end <= start:
            return 0.0
        overlap = min(self.last_seen, end) - max(self.first_seen, start)
        return max(0.0, overlap) / (end - start)


class ScreenTextIndex(BaseModel):
    """Sortie de l'étape `screen` : `data/screen_elements.json`."""

    sample_fps: float
    frames_sampled: int
    frames_analysed: int  # frames réellement passées à l'OCR (les autres sont inchangées)
    elements: list[ScreenElement] = Field(default_factory=list)


class OverlayCandidate(BaseModel):
    """Verdict d'appariement pour un segment (étape `match`).

    Chaque segment désignant un élément nommé produit une entrée, acceptée ou
    non, avec le motif du refus. C'est ce qui rend le tri relisible : sans le
    motif, un rappel faible est indiscernable d'un bug.
    """

    segment_id: str
    label: str
    accepted: bool
    reason: str
    element_id: str | None = None
    element_text: str | None = None
    box: BoundingBox | None = None
    score: float = 0.0
    visible_fraction: float = 0.0
    rivals: list[str] = Field(default_factory=list)
    # Signaux ayant conduit au verdict : "ocr" seul, ou "ocr" + "cursor" quand
    # la position du pointeur a départagé des libellés équivalents.
    evidence: list[str] = Field(default_factory=list)
    cursor_distance: float | None = None
    # Fenêtre d'affichage proposée, en secondes depuis le début du segment.
    # None = tout le segment.
    start_offset: float | None = None
    end_offset: float | None = None


class OverlayMatchReport(BaseModel):
    """Sortie de l'étape `match` : `data/overlay_candidates.json`."""

    candidates: list[OverlayCandidate] = Field(default_factory=list)
    segments_total: int = 0
    segments_named: int = 0

    @property
    def accepted(self) -> list[OverlayCandidate]:
        return [c for c in self.candidates if c.accepted]


class CursorSample(BaseModel):
    """Position du pointeur à un instant, normalisée entre 0 et 1."""

    timestamp: float = Field(ge=0)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class CursorTrack(BaseModel):
    """Sortie de l'étape `cursor` : `data/cursor_track.json`."""

    sample_fps: float
    samples: list[CursorSample] = Field(default_factory=list)
