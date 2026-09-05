"""Chargement typé de config.yaml (voir docs/plan-technique.md, section 10)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    input_video: str
    data_dir: str
    audio_dir: str
    frames_dir: str
    overlays_dir: str
    models_dir: str
    output_dir: str
    logs_dir: str

    def resolve(self, field: str) -> Path:
        return PROJECT_ROOT / getattr(self, field)


class HardwareConfig(BaseModel):
    device: str = "auto"


class InspectionConfig(BaseModel):
    thumbnail_interval_seconds: float = 2.0


class ScreenTextConfig(BaseModel):
    sample_fps: float = 2.0
    min_confidence: float = 0.6
    min_text_length: int = 2
    change_pixel_delta: int = 25
    change_ratio: float = 0.0005
    merge_iou: float = 0.6
    max_gap_seconds: float = 1.0


class WhisperConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_repo: str
    model_size: str
    compute_type: str = "auto"
    language: str = "fr"


class LlmConfig(BaseModel):
    repo_id: str
    filename: str
    n_ctx: int = 8192
    n_gpu_layers: str | int = "auto"
    temperature: float = 0.2


class TtsConfig(BaseModel):
    engine: str = "piper"
    piper_repo_id: str
    voice: str
    sample_rate: int = 22050


class RetimingConfig(BaseModel):
    max_speed_factor: float = 1.08
    min_speed_factor: float = 0.95
    min_pause_ms: int = 80


class SubtitlesConfig(BaseModel):
    max_chars_per_line: int = 42
    max_lines: int = 2
    font_path: str | None = None


class OverlaysConfig(BaseModel):
    font_path: str | None = None
    highlight_color: str = "yellow"
    callout_color: str = "white"
    line_width: int = 4


class ExportConfig(BaseModel):
    container: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_sample_rate: int = 48000
    crf: int = 18
    preview_max_height: int = 480


class PipelineConfig(BaseModel):
    paths: PathsConfig
    hardware: HardwareConfig
    inspection: InspectionConfig
    screen_text: ScreenTextConfig
    whisper: WhisperConfig
    llm: LlmConfig
    tts: TtsConfig
    retiming: RetimingConfig
    subtitles: SubtitlesConfig
    overlays: OverlaysConfig
    export: ExportConfig
    glossary_file: str


def load_config(path: Path | None = None) -> PipelineConfig:
    config_path = path or (PROJECT_ROOT / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)
