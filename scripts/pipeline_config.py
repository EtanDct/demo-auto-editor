"""Chargement typé de config.yaml (voir docs/plan-technique.md, section 10)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    work_dir: str
    output_dir: str
    logs_dir: str

    def resolve(self, field: str) -> Path:
        return PROJECT_ROOT / getattr(self, field)


class HardwareConfig(BaseModel):
    device: str = "auto"


class InspectionConfig(BaseModel):
    thumbnail_interval_seconds: float = 2.0


class CropConfig(BaseModel):
    top: str | int = "auto"
    max_fraction: float = 0.25
    # Ancrage par la couleur de la barre applicative (méthode principale).
    anchor_colors: list[str] = ["#354a5f"]
    anchor_tolerance: int = 24
    anchor_coverage: float = 0.6
    anchor_min_rows: int = 6
    anchor_min_frames: float = 0.5
    run_rows: int = 20
    sample_fps: float = 1.0
    sample_frames: int = 120
    intermediate_crf: int = 14


class ScreenTextConfig(BaseModel):
    sample_fps: float = 2.0
    min_confidence: float = 0.6
    min_text_length: int = 2
    change_pixel_delta: int = 25
    change_ratio: float = 0.0005
    max_ocr_frames: int = 200
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
    max_segment_seconds: float = 14.0


class TtsConfig(BaseModel):
    engine: Literal["piper", "kokoro"] = "kokoro"
    voice: str
    speed: float = 1.0
    lang: str = "en-us"
    kokoro_release: str = "model-files-v1.0"
    kokoro_model: str = "kokoro-v1.0.onnx"
    kokoro_voices: str = "voices-v1.0.bin"
    piper_repo_id: str = "rhasspy/piper-voices"


class RetimingConfig(BaseModel):
    max_speed_factor: float = 1.08
    min_speed_factor: float = 0.95
    min_pause_ms: int = 80
    max_slack_seconds: float = 0.8
    min_shot_seconds: float = 1.2


class SubtitlesConfig(BaseModel):
    max_chars_per_line: int = 42
    max_lines: int = 2
    font_path: str | None = None


class CursorConfig(BaseModel):
    sample_fps: float = 8.0
    pixel_delta: int = 25
    min_area_fraction: float = 0.00002
    max_area_fraction: float = 0.002
    match_tolerance_fraction: float = 0.01
    hold_seconds: float = 2.0


class CursorOverlayConfig(BaseModel):
    enabled: bool = True
    max_hold_seconds: float = 6.0
    hover_step_seconds: float = 0.125
    follow_enabled: bool = False
    marker_size: float = 0.055
    marker_color: str = "cyan"
    marker_opacity: float = 0.7
    marker_thickness: int = 2
    hover_enabled: bool = True
    hover_max_distance: float = 0.012
    hover_max_box_area: float = 0.05
    hover_max_box_width: float = 0.22
    hover_max_chars: int = 30
    min_hover_seconds: float = 0.35
    hover_join_seconds: float = 0.5
    hover_tail_seconds: float = 0.4
    hover_padding: float = 0.004
    hover_color: str = "yellow"
    hover_opacity: float = 0.9
    hover_thickness: int = 4


class OverlayMatchingConfig(BaseModel):
    min_score: float = 0.75
    ambiguity_margin: float = 0.1
    cursor_max_distance: float = 0.02
    min_visible_fraction: float = 0.5
    min_box_area: float = 0.0002
    max_box_area: float = 0.25
    time_margin_seconds: float = 1.0
    action_type: str = "highlight"


class OverlaysConfig(BaseModel):
    fade_seconds: float = 0.25
    fade_steps: int = 4
    font_path: str | None = None
    highlight_color: str = "yellow"
    callout_color: str = "white"
    line_width: int = 4


class IntroConfig(BaseModel):
    enabled: bool = True
    duration_seconds: float = 5.0
    fade_seconds: float = 0.6
    title: str | None = None
    subtitle: str | None = None
    background: str = "0x101418"
    text_color: str = "white"
    title_size: int = 84
    subtitle_size: int = 40


class LayoutConfig(BaseModel):
    enabled: bool = True
    width: int = 1920
    height: int = 1080
    background: str = "0x354a5f"
    top_height: int = 64
    min_bottom_height: int = 110
    band_padding: int = 32
    font_name: str = "Arial"
    subtitle_size: int = 40
    subtitle_color: str = "white"
    top_text_size: int = 28
    text_color: str = "white"
    muted_color: str = "0xC5D0DA"
    accent_color: str = "0xF5B800"
    track_color: str = "0x24333F"
    fade_seconds: float = 0.3
    chapters_enabled: bool = True
    chapter_seconds: float = 35.0
    max_chapters: int = 6
    max_chapter_chars: int = 32
    echo_enabled: bool = True
    echo_text_color: str = "0x1A1A1A"
    progress_enabled: bool = True
    progress_thickness: int = 4
    logo_path: str | None = None
    logo_height: int = 36


class OutroConfig(BaseModel):
    enabled: bool = True
    duration_seconds: float = 4.0
    fade_seconds: float = 0.6
    title: str | None = None
    subtitle: str | None = "Thanks for watching"


class ExportConfig(BaseModel):
    container: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_sample_rate: int = 48000
    crf: int = 18
    preview_max_height: int = 480
    loudness_lufs: float = -16
    true_peak_db: float = -1.5


class PipelineConfig(BaseModel):
    paths: PathsConfig
    hardware: HardwareConfig
    inspection: InspectionConfig
    crop: CropConfig
    screen_text: ScreenTextConfig
    whisper: WhisperConfig
    llm: LlmConfig
    tts: TtsConfig
    retiming: RetimingConfig
    subtitles: SubtitlesConfig
    cursor: CursorConfig
    cursor_overlay: CursorOverlayConfig
    overlay_matching: OverlayMatchingConfig
    overlays: OverlaysConfig
    intro: IntroConfig
    outro: OutroConfig = OutroConfig()
    layout: LayoutConfig = LayoutConfig()
    export: ExportConfig
    glossary_file: str


def load_config(path: Path | None = None) -> PipelineConfig:
    config_path = path or (PROJECT_ROOT / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)
