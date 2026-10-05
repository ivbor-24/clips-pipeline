import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputConfig(BaseModel):
    default_source: str = ""
    cache_dir: str = "artifacts/cache"


class PreprocessingConfig(BaseModel):
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    video_max_height: int = 1080
    video_codec: str = "h264"
    video_crf: int = 23
    # Remux instead of re-encoding when the source is already 8-bit H.264 no
    # taller than video_max_height (a 90-min 1080p re-encode takes ~45 min).
    copy_compatible_video: bool = True
    denoise: bool = False
    denoise_method: Literal["afftdn", "rnnoise"] = "afftdn"


class TranscriptionConfig(BaseModel):
    model: str = "large-v3-turbo"
    language: str = "auto"
    batch_size: int = 8
    compute_type: str = "float16"
    chunk_duration_min: int = 30
    engine: str = "faster_whisper"  # faster_whisper | whisper_cpp
    whisper_cpp_binary: str = "whisper-cli"  # path or name resolved via PATH
    whisper_cpp_model_path: str = ""  # path to ggml model file
    whisper_cpp_threads: int = Field(default=8, ge=1)
    whisper_cpp_beam_size: int = Field(default=4, ge=1)
    # Text-context tokens carried between 30 s decoding windows (whisper-cli -mc;
    # -1 = model maximum). With carry-over on, long chunks degrade into 1-3 word
    # fragments without punctuation, so it is off by default.
    whisper_cpp_max_context: int = Field(default=0, ge=-1)

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        # An empty language (the Settings page sends "" when the field is
        # cleared, the JSON panel can too) would reach faster-whisper and fail
        # transcription with a message unrelated to the user. Treat it as
        # "auto" (detect), which is also the default.
        value = value.strip()
        return value or "auto"


class LLMConfig(BaseModel):
    enabled: bool = False
    provider: Literal["openai", "anthropic", "qwen_local", "llama_cpp"] = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    prompt_file: str = "config/prompts/segment_analysis_v1.txt"
    llm_weight: float = 0.4
    # Local llama.cpp model settings
    model_repo: Optional[str] = None
    model_file: Optional[str] = None
    model_path: Optional[str] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 512
    top_p: float = 0.95
    chat_format: Optional[str] = None
    download_if_missing: bool = True
    verbose: bool = False
    # Fallback to heuristics if sustained throughput is below this (tokens/sec).
    min_tokens_per_sec: float = 10.0
    # Cap on segments sent to the LLM (top-scored heuristic candidates).
    # null = analyze everything.
    max_segments: Optional[int] = None
    # Prompt for scoring.strategy "llm_windows".
    window_prompt_file: str = "config/prompts/segment_analysis_v3.txt"
    # Append the "/no_think" soft switch (Qwen3): reasoning ate ~40% of tokens.
    disable_thinking: bool = False
    # Fixed sampling seed for reproducible scoring (llama_cpp); null = random.
    seed: Optional[int] = None


class ScoringConfig(BaseModel):
    min_duration: int = 45
    max_duration: int = 90
    min_score_threshold: float = 0.45  # clip score = max of its segments
    # Candidates per video: the user keeps 5-7 of them in review.
    max_clips_per_video: int = 12
    # "keybert": KeyBERT with its embedding model; "statistical": word
    # frequencies, no model. "yake" is read as "statistical": it always was
    # (the yake package was never imported and is no longer a dependency).
    term_extraction_method: Literal["keybert", "statistical"] = "keybert"
    # "segment_merge": sentence scores (heuristics + LLM) merged into tiles;
    # "llm_windows": the LLM picks hook-to-conclusion clips inside windows
    # (src/window_scoring.py); falls back to segment_merge without the LLM.
    # llm_windows is the default: the best measured.
    strategy: Literal["segment_merge", "llm_windows"] = "llm_windows"
    window_sec: float = Field(default=150.0, gt=0)
    window_step_sec: float = Field(default=75.0, gt=0)
    windows_per_call: int = Field(default=3, ge=1)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @field_validator("term_extraction_method", mode="before")
    @classmethod
    def _yake_is_statistical(cls, value: Any) -> Any:
        return "statistical" if value == "yake" else value


class CroppingConfig(BaseModel):
    output_width: int = 1080
    output_height: int = 1920
    face_confidence_threshold: float = 0.6
    moving_average_window: int = 5
    sample_fps: int = 1
    # Tracking segments (one crop window each), split on raw detections:
    # face box size ratio above this means a shot/zoom change ...
    shot_size_ratio_threshold: float = Field(default=1.35, gt=1.0)
    # ... and a center jump above this share of the box width means a cut/pan.
    shot_center_jump_threshold: float = Field(default=0.5, gt=0.0)
    # Segments shorter than this (s) merge into the longer neighbour.
    min_track_segment_sec: float = Field(default=2.0, ge=0.0)
    # ffmpeg scdet threshold for exact shot cuts (crop switches on the cut);
    # 0 disables detection (boundaries then fall on 1/sample_fps samples).
    scene_cut_threshold: float = Field(default=10.0, ge=0.0, le=100.0)
    # Frame decoding for the face detector: auto = VAAPI (Intel/AMD GPU) if a
    # test decode works, else the CPU; software = always the CPU.
    frame_decoder: Literal["auto", "software", "vaapi"] = "auto"
    # Within one shot (no cut) the window moves only if the face shifts by more
    # than this share of the crop width; smaller moves would look like a jerk.
    min_reframe_share: float = Field(default=0.25, ge=0.0, le=1.0)


class SubtitleStyle(BaseModel):
    """Look and layout of burned-in subtitles (see src/subtitles.py).

    Colors are "#RRGGBB". Margins are shares of the output frame, so a style
    works for any resolution; the defaults keep text clear of the TikTok /
    Reels / Shorts overlays (caption and buttons at the bottom and right).
    """

    model_config = ConfigDict(extra="forbid")

    font: str = "Nimbus Mono PS"  # typewriter (Courier-like) with Cyrillic
    font_size: int = Field(default=64, gt=0)  # in output pixels (PlayRes = frame)
    bold: bool = True
    italic: bool = False
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: float = Field(default=0.0, ge=0)  # text outline width (no box)
    shadow: float = Field(default=0.0, ge=0)
    box: bool = True  # one background box behind the whole cue
    box_color: str = "#4D4D4D"
    box_opacity: float = Field(default=0.75, ge=0.0, le=1.0)
    box_padding: float = Field(default=14.0, ge=0)
    anchor: Literal["bottom", "center", "top"] = "bottom"
    # Distance from the anchored edge to the text block, share of the height.
    margin_v: float = Field(default=0.25, ge=0.0, lt=0.5)
    # Left/right margin, share of the width.
    margin_h: float = Field(default=0.13, ge=0.0, lt=0.5)
    max_lines: int = Field(default=3, ge=1)  # lines per cue
    max_chars_per_line: int = Field(default=26, ge=8)  # tuned to font and size


class RenderingConfig(BaseModel):
    audio_loudnorm_i: int = -14
    audio_loudnorm_tp: float = -1.5
    audio_loudnorm_lra: int = 11
    padding_color: str = "black"
    subtitle_format: Literal["srt", "ass"] = "srt"
    # Built-in preset (src/subtitles.py); "default"/"modern"/"minimal" are the
    # legacy looks driven by the subtitle_font/fontsize/*_color keys below.
    subtitle_style: Literal["typewriter", "default", "modern", "minimal"] = "typewriter"
    # Per-field changes on top of the preset, e.g. {"box_opacity": 0.5}; job
    # config_overrides can set them, which is the hook for a style picker.
    subtitle_style_overrides: Dict[str, Any] = Field(default_factory=dict)
    # Extra font directory for libass (fonts not registered in fontconfig).
    subtitle_fonts_dir: Optional[str] = None
    subtitle_font: str = "Arial"
    subtitle_fontsize: int = 48
    subtitle_primary_color: str = "&H00FFFFFF"
    subtitle_outline_color: str = "&H00000000"
    subtitle_back_color: str = "&H80000000"
    video_codec: str = "h264"
    video_preset: str = "medium"
    # auto: the GPU's encoder (NVENC, Quick Sync, VAAPI) if a test encode works,
    # else libx264; software: always libx264; qsv / vaapi / nvenc: only that one.
    video_encoder: Literal["auto", "software", "qsv", "vaapi", "nvenc"] = "auto"
    render_timeout_sec: int = 300

    @field_validator("subtitle_style_overrides")
    @classmethod
    def _check_style_overrides(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        SubtitleStyle(**value)  # unknown keys / bad values fail at load time
        return value


class ChaptersConfig(BaseModel):
    enabled: bool = False
    min_minutes: int = 5
    max_minutes: int = 7
    target_count: Optional[int] = (
        None  # exact chapter count; None = model's discretion (uses min/max_minutes as a guide)
    )
    prompt_file: str = "config/prompts/chapter_detection_v1.txt"
    output_format: Literal["json", "youtube", "both"] = "both"


class BrollConfig(BaseModel):
    enabled: bool = False
    max_suggestions: Optional[int] = None  # cap on suggestions; None = model's discretion
    prompt_file: str = "config/prompts/broll_suggestions_v1.txt"


class CleanupConfig(BaseModel):
    # Days after a job finished when its source video (upload, the job's copy
    # and audio) is removed; clips and the job card stay. 0 = never.
    retention_days: int = Field(default=30, ge=0)
    # Only a warning in the log when uploads/ grows beyond this.
    max_upload_size_gb: int = 50


class OutputConfig(BaseModel):
    clips_dir: str = "output/clips"
    manifest_file: str = "output/manifest.json"
    review_file: str = "artifacts/review.json"
    output_dir_template: str = "{video_name}_{timestamp}"
    timestamp_format: str = "%Y-%m-%d_%H-%M-%S"
    isolated: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = "artifacts/logs/pipeline.log"
    json_format: bool = True


class GPUSettings(BaseModel):
    backend: Literal["auto", "cuda", "openvino", "rocm", "cpu"] = "auto"
    whisper_device_override: Optional[str] = None
    whisper_compute_override: Optional[str] = None


class Config(BaseModel):
    work_dir: Path = Path(".")
    input: InputConfig = Field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    cropping: CroppingConfig = Field(default_factory=CroppingConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    chapters: ChaptersConfig = Field(default_factory=ChaptersConfig)
    broll: BrollConfig = Field(default_factory=BrollConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    gpu: GPUSettings = Field(default_factory=GPUSettings)
    profiles_dir: str = "config/profiles"
    # Settings for the GPU the service is installed for, picked by the
    # PIPELINE_HARDWARE environment variable (see with_hardware_profile).
    hardware_dir: str = "config/hardware"

    @classmethod
    def load(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def with_hardware_profile(self, name: Optional[str] = None) -> "Config":
        """Apply ``<hardware_dir>/<name>.yaml``, the settings of the installed GPU.

        ``name`` defaults to the ``PIPELINE_HARDWARE`` environment variable,
        which the Docker image sets to its backend (cpu / cuda / openvino).
        Applied before a user profile, so a profile (low_vram) still wins.
        A backend without a file needs no changes: the config is returned as is.
        """
        import structlog

        name = (name if name is not None else os.environ.get("PIPELINE_HARDWARE", "")).strip()
        if not name:
            return self
        hardware_dir = Path(self.hardware_dir)
        path = hardware_dir / f"{name}.yaml"
        if path.resolve().parent != hardware_dir.resolve():
            raise ValueError(f"Invalid hardware profile name: {name}")
        if not path.exists():
            return self
        with open(path, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        structlog.get_logger("config").info("hardware_profile_applied", hardware=name)
        merged = self.model_dump()
        _deep_merge(merged, data)
        return Config(**merged)

    def load_profile(self, profile_name: str) -> "Config":
        """Load and merge a profile into this config, returning updated Config."""
        import structlog

        logger = structlog.get_logger("config")

        profile_name = RENAMED_PROFILES.get(profile_name, profile_name)
        profiles_dir = Path(self.profiles_dir)
        profile_path = profiles_dir / f"{profile_name}.yaml"

        if profile_path.resolve().parent != profiles_dir.resolve():
            raise ValueError(f"Invalid profile name: {profile_name}")

        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, "r") as f:
            profile_data: Dict[str, Any] = yaml.safe_load(f) or {}

        logger.info("profile_loaded", profile=profile_name, path=str(profile_path))

        merged = self.model_dump()
        _deep_merge(merged, profile_data)
        return Config(**merged)


# Old profile names still found in the settings of existing jobs.
RENAMED_PROFILES = {"local_arc": "intel_arc"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge override dict into base dict in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
