#!/usr/bin/env python3
"""
Pre-download the AI models config/config.yaml is configured to use, without
loading them into memory/VRAM: Whisper (faster-whisper or whisper.cpp), the
LLM, KeyBERT's embedding model and the YuNet face detector.

Without this, the first `just process` silently blocks for however long a
multi-GB download of Whisper/GGUF weights takes, with only structlog lines
(no progress bar) to show for it. Run this once right after install so that
wait happens up front, visibly, instead of mid-pipeline.

Models listed in config/models.lock.yaml are downloaded from their pinned
revision, and single-file models (GGUF, ggml) are checked against the pinned
sha256. A mismatch exits with code 1.

Usage:
    python3 scripts/prefetch_models.py [--config config/config.yaml] [--profile NAME]
    just prefetch-models [config] [profile]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402

from src.config import Config, LLMConfig, TranscriptionConfig  # noqa: E402
from src.model_registry import (  # noqa: E402
    KEYBERT_MODEL,
    download_pinned_file,
    find_pinned_file,
    find_pinned_snapshot,
    link_into_place,
    verify_file,
)

logger = structlog.get_logger("prefetch_models")


def prefetch_whisper(model_name: str) -> None:
    """Download the faster-whisper model without loading it onto a device."""
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        logger.warning(
            "faster_whisper_not_installed",
            hint="pip install faster-whisper (or: pip install -e .)",
        )
        return

    pin = find_pinned_snapshot(model_name)
    revision = pin.revision if pin else None
    logger.info("prefetching_whisper_model", model=model_name, revision=revision or "latest")
    download_model(model_name, revision=revision)
    logger.info("whisper_model_ready", model=model_name)


WHISPER_CPP_REPO = "ggerganov/whisper.cpp"


def prefetch_whisper_cpp_model(config: TranscriptionConfig) -> Path:
    """Download (or verify) the ggml model for transcription.engine=whisper_cpp.

    The model goes to transcription.whisper_cpp_model_path, or to
    artifacts/cache/models/ggml-<model>.bin when that is empty.
    """
    filename = f"ggml-{config.model}.bin"
    target = Path(config.whisper_cpp_model_path or f"artifacts/cache/models/{filename}")
    pin = find_pinned_file(WHISPER_CPP_REPO, filename)

    if target.exists():
        if pin is None:
            logger.warning("whisper_cpp_model_not_pinned", path=str(target))
        else:
            verify_file(target, pin)
        logger.info("whisper_cpp_model_ready", path=str(target))
        return target

    if pin is None:
        raise RuntimeError(
            f"{filename} is not in config/models.lock.yaml; download it manually " f"to {target}"
        )
    cached = download_pinned_file(pin)
    link_into_place(target, cached)
    logger.info("whisper_cpp_model_ready", path=str(target))
    return target


def prefetch_llm(llm_config: LLMConfig) -> None:
    """Download the configured LLM analysis model, if any is configured."""
    if not llm_config.enabled:
        logger.info("llm_disabled_skipping_prefetch")
        return

    if llm_config.provider == "llama_cpp":
        from src.llm_adapter import LlamaCppAdapter, LLMAdapterError

        logger.info(
            "prefetching_llama_cpp_model",
            model_repo=llm_config.model_repo,
            model_file=llm_config.model_file,
            model_path=llm_config.model_path,
        )
        adapter = LlamaCppAdapter(llm_config)
        try:
            path = adapter._resolve_model_path()
        except LLMAdapterError as e:
            logger.error("llama_cpp_prefetch_failed", error=str(e))
            raise
        pin = find_pinned_file(llm_config.model_repo or "", llm_config.model_file or "")
        if pin is not None:
            verify_file(path, pin)
        logger.info("llama_cpp_model_ready", path=str(path))

    elif llm_config.provider == "qwen_local":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.warning(
                "huggingface_hub_not_installed",
                hint="pip install huggingface_hub",
            )
            return

        model_name = llm_config.model or "Qwen/Qwen2.5-7B-Instruct"
        logger.info("prefetching_qwen_model", model=model_name)
        snapshot_download(repo_id=model_name)
        logger.info("qwen_model_ready", model=model_name)

    else:
        # openai / anthropic make live API calls; nothing to pre-download.
        logger.info(
            "llm_provider_needs_no_prefetch",
            provider=llm_config.provider,
        )


def prefetch_keybert(method: str) -> None:
    """Download the KeyBERT embedding model, used on the first scoring otherwise."""
    if method != "keybert":
        logger.info("keybert_not_used_skipping_prefetch", method=method)
        return
    from huggingface_hub import snapshot_download

    logger.info("prefetching_keybert_model", model=KEYBERT_MODEL)
    snapshot_download(repo_id=KEYBERT_MODEL)
    logger.info("keybert_model_ready", model=KEYBERT_MODEL)


def prefetch_face_detector() -> None:
    """Download the YuNet face detector (~230 KB) from its pinned revision."""
    from src.face_cropping import FACE_DETECTOR_PATH, face_detector_model

    logger.info("prefetching_face_detector", path=str(FACE_DETECTOR_PATH))
    face_detector_model()
    logger.info("face_detector_ready", path=str(FACE_DETECTOR_PATH))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--profile", default=None, help="Optional profile to merge in first")
    parser.add_argument("--skip-whisper", action="store_true", help="Don't prefetch Whisper")
    parser.add_argument("--skip-llm", action="store_true", help="Don't prefetch the LLM model")
    args = parser.parse_args()

    config = Config.load(args.config).with_hardware_profile()
    if args.profile:
        config = config.load_profile(args.profile)

    try:
        if not args.skip_whisper:
            if config.transcription.engine == "whisper_cpp":
                prefetch_whisper_cpp_model(config.transcription)
            else:
                prefetch_whisper(config.transcription.model)

        if not args.skip_llm:
            prefetch_llm(config.scoring.llm)

        prefetch_keybert(config.scoring.term_extraction_method)
        prefetch_face_detector()
    except Exception as e:
        logger.error("prefetch_failed", error=str(e))
        return 1

    logger.info("prefetch_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
