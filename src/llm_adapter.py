"""
TASK-P1-05: LLM Content Analysis Adapter

Provides optional LLM-based content analysis for transcript segments.
Supports:
- OpenAI API and OpenAI-compatible services
- Anthropic API (Claude)
- Local Qwen2.5 via transformers
- Local GGUF models via llama.cpp (e.g. Gemma 4)

Inputs:
- List of transcript segments
- System prompt from config/prompts/

Outputs:
- List of dicts with LLM-enhanced scores and metadata per segment
"""

import ctypes
import json
import os
import re
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import structlog

from src.config import LLMConfig
from src.model_registry import (
    ModelIntegrityError,
    check_size,
    find_pinned_file,
    verify_file,
)

logger = structlog.get_logger("llm_adapter")


class LLMAdapterError(Exception):
    """Custom exception for LLM adapter errors."""

    pass


class LLMUnavailableError(LLMAdapterError):
    """The backend cannot serve this run at all (model load failed, too slow).

    Unlike a failure of a single request, retrying other batches is pointless.
    """

    pass


class LLMAdapter(ABC):
    """Abstract base class for LLM content analysis adapters."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._prompt_template: Optional[str] = None

    def _load_prompt(self) -> str:
        """Load the analysis prompt template from file."""
        if self._prompt_template is not None:
            return self._prompt_template

        prompt_path = Path(self.config.prompt_file)
        if not prompt_path.exists():
            raise LLMAdapterError(f"Prompt file not found: {prompt_path}")

        self._prompt_template = prompt_path.read_text(encoding="utf-8")
        logger.info("prompt_loaded", path=str(prompt_path))
        return self._prompt_template

    def _build_prompt(self, segments: List[Dict[str, Any]]) -> str:
        """Build the full prompt with segment data."""
        template = self._load_prompt()
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        return template.replace("{{SEGMENTS}}", segments_json)

    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured segment analysis.

        Accepts a list, or a dict wrapping a list under a known container key
        ("segments", "chapters", "results"), or any single-list value.
        Handles reasoning blocks (<think>...</think>), code fences and
        prose around the JSON payload.
        """
        text = response_text.strip()

        # Strip code fences
        lines = text.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

        # Strip <think>...</think> reasoning blocks (e.g. Qwen3)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # A response cut off by max_tokens inside reasoning has no closing tag;
        # drop the partial block so JSON-like drafts in it are never parsed.
        unclosed = text.find("<think>")
        if unclosed != -1:
            logger.warning("llm_response_truncated_in_think")
            text = text[:unclosed].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Extract the JSON object from surrounding prose
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    result = json.loads(text[start : end + 1])
                except json.JSONDecodeError as e:
                    logger.error("llm_response_parse_failed", error=str(e))
                    raise LLMAdapterError(f"Failed to parse LLM response as JSON: {e}") from e
            else:
                logger.error("llm_response_no_json_found")
                raise LLMAdapterError("Failed to parse LLM response as JSON: no JSON object found")

        if isinstance(result, dict):
            for key in ("segments", "chapters", "results"):
                if key in result and isinstance(result[key], list):
                    return result[key]
            for value in result.values():
                if isinstance(value, list):
                    return value
            raise LLMAdapterError("Unexpected response format")
        if isinstance(result, list):
            return result
        raise LLMAdapterError("Unexpected response format")

    @abstractmethod
    def analyze_segments(
        self, transcript: List[Dict[str, Any]], prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze transcript segments using LLM.

        Args:
            transcript: List of segment dicts with 'text', 'start', 'end', 'id' keys
            prompt: Optional override prompt (uses config prompt_file if None)

        Returns:
            List of dicts with keys: id, llm_score, content_type, key_concepts,
            virality_score, self_containment_llm, reasoning
        """
        ...

    def unload(self) -> None:
        """Unload model and free resources (VRAM for local models)."""
        pass


# Room for the reasoning that current API models do before answering: it
# counts toward the output limit, and scoring.llm.max_tokens is sized for the
# local model's context window.
API_REASONING_MAX_TOKENS = 16000


def _api_error(provider: str, error: Exception) -> LLMAdapterError:
    """Wrap an API client error.

    A wrong key, a model the account cannot use and no connection fail every
    request alike: LLMUnavailableError, so the other batches are not tried.
    """
    message = f"{provider} API call failed: {error}"
    if (
        getattr(error, "status_code", None) in (401, 403, 404)
        or type(error).__name__ == "APIConnectionError"
    ):
        return LLMUnavailableError(message)
    return LLMAdapterError(message)


class OpenAIAdapter(LLMAdapter):
    """LLM adapter for the OpenAI API and OpenAI-compatible services (api_base)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        # Request parameters the model turned down (see _adjust_for_error); a
        # job sends many requests and must not pay for the refusal every time.
        self._params: Optional[Dict[str, Any]] = None

    def _get_client(self):
        """Lazily initialize OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            raise LLMAdapterError("openai package not installed. Run: pip install openai")

        api_key = self.config.api_key
        if not api_key:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise LLMUnavailableError("OpenAI API key not provided (config or OPENAI_API_KEY env)")

        kwargs = {"api_key": api_key}
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base

        self._client = OpenAI(**kwargs)
        logger.info("openai_client_initialized", model=self.config.model)
        return self._client

    def _request_params(self) -> Dict[str, Any]:
        if self._params is None:
            self._params = {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            }
        return self._params

    def _adjust_for_error(self, error: Exception) -> bool:
        """Drop or rename a parameter the model rejected with HTTP 400.

        Models differ: reasoning models take only the default temperature and
        ``max_completion_tokens`` instead of ``max_tokens``; some compatible
        services have no JSON mode. Returns whether the request is worth
        repeating.
        """
        if getattr(error, "status_code", None) != 400:
            return False
        message = str(error)
        params = self._request_params()
        if "max_tokens" in params and "max_completion_tokens" in message:
            del params["max_tokens"]
            params["max_completion_tokens"] = max(self.config.max_tokens, API_REASONING_MAX_TOKENS)
            changed = "max_tokens"
        elif "temperature" in params and "temperature" in message:
            del params["temperature"]
            changed = "temperature"
        elif "response_format" in params and "response_format" in message:
            del params["response_format"]
            changed = "response_format"
        else:
            return False
        logger.warning("openai_parameter_rejected", parameter=changed, error=message)
        return True

    def analyze_segments(
        self, transcript: List[Dict[str, Any]], prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze segments using OpenAI API."""
        logger.info(
            "openai_analysis_start",
            model=self.config.model,
            segment_count=len(transcript),
        )

        if prompt is None:
            prompt = self._build_prompt(transcript)

        client = self._get_client()
        messages = [
            {"role": "system", "content": "You are a content analysis expert."},
            {"role": "user", "content": prompt},
        ]

        # At most one retry per parameter that can be adjusted.
        for _ in range(4):
            try:
                response = client.chat.completions.create(
                    model=self.config.model, messages=messages, **self._request_params()
                )
                break
            except Exception as e:
                if self._adjust_for_error(e):
                    continue
                logger.error("openai_api_error", error=str(e))
                raise _api_error("OpenAI", e) from e
        else:
            raise LLMAdapterError("OpenAI API call failed: the model rejected every request")

        response_text = response.choices[0].message.content
        logger.info(
            "openai_analysis_complete",
            tokens_used=response.usage.total_tokens if response.usage else None,
        )
        if not response_text:
            # A reasoning model can spend the whole output limit on reasoning.
            raise LLMAdapterError(
                "OpenAI API returned no text "
                f"(finish_reason: {response.choices[0].finish_reason})"
            )
        return self._parse_response(response_text)


class AnthropicAdapter(LLMAdapter):
    """LLM adapter for Anthropic API (Claude)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazily initialize Anthropic client."""
        if self._client is not None:
            return self._client

        try:
            import anthropic
        except ImportError:
            raise LLMAdapterError("anthropic package not installed. Run: pip install anthropic")

        api_key = self.config.api_key
        if not api_key:
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise LLMUnavailableError(
                "Anthropic API key not provided (config or ANTHROPIC_API_KEY env)"
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info("anthropic_client_initialized", model=self.config.model)
        return self._client

    def analyze_segments(
        self, transcript: List[Dict[str, Any]], prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze segments using Anthropic API.

        No temperature: current Claude models reject sampling parameters. They
        think before answering by default, which counts toward max_tokens.
        """
        logger.info(
            "anthropic_analysis_start",
            model=self.config.model,
            segment_count=len(transcript),
        )

        if prompt is None:
            prompt = self._build_prompt(transcript)

        client = self._get_client()

        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=max(self.config.max_tokens, API_REASONING_MAX_TOKENS),
                system="You are a content analysis expert.",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error("anthropic_api_error", error=str(e))
            raise _api_error("Anthropic", e) from e

        # Thinking blocks come before the answer; only text blocks hold it.
        response_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        logger.info(
            "anthropic_analysis_complete",
            tokens_used=response.usage.output_tokens if response.usage else None,
            stop_reason=response.stop_reason,
        )
        if not response_text:
            raise LLMAdapterError(
                f"Anthropic API returned no text (stop_reason: {response.stop_reason})"
            )
        return self._parse_response(response_text)


class QwenLocalAdapter(LLMAdapter):
    """LLM adapter for local Qwen2.5 model via transformers."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazily load Qwen model and tokenizer."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise LLMAdapterError(
                "transformers/torch not installed. Run: pip install transformers torch"
            )

        model_name = self.config.model or "Qwen/Qwen2.5-7B-Instruct"
        logger.info("qwen_model_loading", model=model_name)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("qwen_model_loaded", model=model_name)
        except Exception as e:
            logger.error("qwen_model_load_failed", error=str(e))
            raise LLMAdapterError(f"Failed to load Qwen model: {e}") from e

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("qwen_model_unloaded")

    def analyze_segments(
        self, transcript: List[Dict[str, Any]], prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze segments using local Qwen2.5 model."""
        logger.info(
            "qwen_analysis_start",
            model=self.config.model,
            segment_count=len(transcript),
        )

        self._load_model()

        if prompt is None:
            prompt = self._build_prompt(transcript)

        try:
            import torch

            messages = [
                {"role": "system", "content": "You are a content analysis expert."},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                )

            response_ids = outputs[0][inputs["input_ids"].shape[1] :]
            response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

            logger.info("qwen_analysis_complete")
            return self._parse_response(response_text)
        except Exception as e:
            logger.error("qwen_inference_error", error=str(e))
            raise LLMAdapterError(f"Qwen inference failed: {e}") from e


def _hf_hub_download(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download a file from Hugging Face (extracted for testability)."""
    try:
        from huggingface_hub import hf_hub_download as _download
    except ImportError:
        raise LLMAdapterError("huggingface_hub not installed. Run: pip install huggingface_hub")
    return _download(repo_id=repo_id, filename=filename, token=token, revision=revision)


def _create_llama(**kwargs: Any) -> Any:
    """Create a llama_cpp.Llama instance (extracted for testability)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise LLMAdapterError(
            "llama-cpp-python not installed. "
            "Install the Vulkan wheel for Intel Arc: "
            "pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/vulkan"
        )
    return Llama(**kwargs)


# llama.cpp says where the model went only in its log, e.g.
#   llama_prepare_model_devices: using device Vulkan0 (Intel(R) Arc(tm) B580 Graphics (BMG G21))
#       (0000:05:00.0) - 10210 MiB free
#   load_tensors: offloaded 41/41 layers to GPU
# and llama-cpp-python drops that log unless verbose is on.
_OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_DEVICE_RE = re.compile(r"using device (\S+) \((.+?)\)(?: \([0-9a-fA-F:.]+\))? - (\d+) MiB free")


@contextmanager
def _capture_llama_log(lines: List[str]) -> Iterator[None]:
    """Collect llama.cpp's log into ``lines`` while the block runs.

    llama-cpp-python's own handler (which prints when verbose) still gets
    every line; it is put back afterwards.
    """
    try:
        import llama_cpp
        from llama_cpp import _logger
    except ImportError:
        yield
        return

    @llama_cpp.llama_log_callback
    def _collect(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:
        lines.append(text.decode("utf-8", errors="replace"))
        _logger.llama_log_callback(level, text, user_data)

    llama_cpp.llama_log_set(_collect, ctypes.c_void_p(0))
    try:
        yield
    finally:
        llama_cpp.llama_log_set(_logger.llama_log_callback, ctypes.c_void_p(0))


def parse_gpu_offload(log: str) -> Optional[Dict[str, Any]]:
    """Layers on the GPU and the device, from llama.cpp's model loading log."""
    offload = _OFFLOAD_RE.search(log)
    if offload is None:
        return None
    device = _DEVICE_RE.search(log)
    return {
        "offloaded": int(offload.group(1)),
        "layers": int(offload.group(2)),
        "device": f"{device.group(1)} ({device.group(2)})" if device else None,
        "free_mib": int(device.group(3)) if device else None,
    }


class LlamaCppAdapter(LLMAdapter):
    """LLM adapter for local GGUF models via llama-cpp-python."""

    def __init__(
        self,
        config: LLMConfig,
        work_dir: Optional[Union[str, Path]] = None,
        stage: Optional[str] = None,
    ):
        super().__init__(config)
        self._llm: Optional[Any] = None
        self._calls = 0
        # Where a partly-on-CPU model is reported to the job page (notices).
        self._work_dir = work_dir
        self._stage = stage
        self.gpu_offload: Optional[Dict[str, Any]] = None

    # The first call of a fresh process is also a warm-up: Vulkan compiles its
    # shaders (minutes without a shader cache, e.g. the first job in a new
    # container) and the long prompt is processed. It only rules out a backend
    # that is hopeless even allowing for that.
    WARMUP_SLOWDOWN = 3.0

    def _check_throughput(self, tokens_per_sec: float) -> None:
        """Give up on a backend that is too slow to be the GPU (e.g. a silent CPU run).

        Throughput is a property of the backend, so the second call decides
        (the first one within WARMUP_SLOWDOWN); later slow calls only warn and
        keep their valid results.
        """
        minimum = self.config.min_tokens_per_sec
        if minimum <= 0 or tokens_per_sec >= minimum:
            return
        limit = minimum / self.WARMUP_SLOWDOWN if self._calls == 1 else minimum
        if self._calls <= 2 and tokens_per_sec < limit:
            raise LLMUnavailableError(
                f"llama.cpp throughput too low: {tokens_per_sec:.2f} tok/s "
                f"(below {limit:.1f} tok/s)"
            )
        logger.warning(
            "llama_cpp_throughput_low",
            tokens_per_sec=round(tokens_per_sec, 2),
            min_tokens_per_sec=minimum,
            call=self._calls,
        )

    def _link_model_path(self, local_path: Path, cached_path: Path) -> None:
        """Best-effort symlink so config.model_path points at the downloaded model.

        Keeps repeated runs on the fast "local file exists" branch above,
        and makes the resolved model discoverable at the configured path
        instead of buried in the Hugging Face cache. Never raises: a failure
        here (cross-device link, permissions, race) must not break inference.
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                local_path.symlink_to(cached_path.resolve())
                logger.info(
                    "llama_cpp_model_linked",
                    link=str(local_path),
                    target=str(cached_path),
                )
        except OSError as e:
            logger.warning("llama_cpp_model_link_failed", path=str(local_path), error=str(e))

    def _pinned_model(self):
        """Pin for the configured model from config/models.lock.yaml, if any."""
        if not self.config.model_repo or not self.config.model_file:
            return None
        return find_pinned_file(self.config.model_repo, self.config.model_file)

    def _resolve_model_path(self) -> Path:
        """Resolve local GGUF path, downloading from Hugging Face if needed."""
        local_path: Optional[Path] = None
        if self.config.model_path:
            local_path = Path(os.path.expandvars(self.config.model_path)).expanduser()
            if local_path.exists():
                logger.info("llama_cpp_using_local_model", path=str(local_path))
                pin = self._pinned_model()
                if pin is not None:
                    check_size(local_path, pin)
                return local_path
            logger.warning(
                "llama_cpp_local_model_missing_will_download",
                path=str(local_path),
            )

        repo_id = self.config.model_repo
        filename = self.config.model_file

        if not repo_id or not filename:
            raise LLMAdapterError("llama_cpp requires model_path or both model_repo and model_file")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        if not self.config.download_if_missing:
            raise LLMAdapterError(
                f"Model not available locally and download_if_missing=False "
                f"({repo_id}/{filename})"
            )

        pin = self._pinned_model()
        logger.info(
            "llama_cpp_downloading_model",
            repo_id=repo_id,
            filename=filename,
            revision=pin.revision if pin else "latest (not in models.lock.yaml)",
        )

        try:
            cached_path = _hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                revision=pin.revision if pin else None,
            )
        except Exception as e:
            logger.error("llama_cpp_download_failed", error=str(e))
            raise LLMAdapterError(
                f"Failed to download model {repo_id}/{filename}. "
                "Ensure HF_TOKEN is set and you accepted the model terms."
            ) from e

        cached = Path(cached_path)
        if pin is not None:
            try:
                verify_file(cached, pin)
            except ModelIntegrityError as e:
                logger.error("llama_cpp_model_checksum_failed", error=str(e))
                raise LLMAdapterError(str(e)) from e
        if local_path is not None:
            self._link_model_path(local_path, cached)

        return cached

    def _load_model(self) -> None:
        """Lazily load the llama.cpp model."""
        if self._llm is not None:
            return

        try:
            model_path = self._resolve_model_path()
        except LLMAdapterError as e:
            raise LLMUnavailableError(str(e)) from e

        logger.info(
            "llama_cpp_loading_model",
            path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
        )

        try:
            kwargs: Dict[str, Any] = {
                "model_path": str(model_path),
                "n_ctx": self.config.n_ctx,
                "n_gpu_layers": self.config.n_gpu_layers,
                "n_batch": self.config.n_batch,
                "verbose": self.config.verbose,
            }
            if self.config.chat_format:
                kwargs["chat_format"] = self.config.chat_format

            log: List[str] = []
            with _capture_llama_log(log):
                self._llm = _create_llama(**kwargs)
            logger.info("llama_cpp_model_loaded", path=str(model_path))
        except Exception as e:
            logger.error("llama_cpp_model_load_failed", error=str(e))
            raise LLMUnavailableError(f"Failed to load llama.cpp model: {e}") from e
        self._report_gpu_offload("".join(log))

    def _report_gpu_offload(self, log: str) -> None:
        """Write where the model runs to job.log; a model partly on the CPU is a notice."""
        self.gpu_offload = parse_gpu_offload(log)
        if self.gpu_offload is None:
            logger.info("llama_cpp_gpu_offload", offloaded=None, reason="not in llama.cpp log")
            return
        offloaded, layers = self.gpu_offload["offloaded"], self.gpu_offload["layers"]
        logger.info("llama_cpp_gpu_offload", **self.gpu_offload)
        if self.config.n_gpu_layers == 0 or offloaded >= layers:
            return
        logger.warning(
            "llama_cpp_partly_on_cpu", offloaded=offloaded, layers=layers, hint="not enough VRAM"
        )
        if self._work_dir is not None and self._stage:
            from src.notices import add_notice

            add_notice(
                self._work_dir,
                self._stage,
                "llm_partly_on_cpu",
                f"Only {offloaded} of {layers} LLM layers fit in GPU memory; the rest ran on "
                "the CPU, so choosing clips was slow. Close other programs that use the GPU, "
                "or use a smaller model or an API provider.",
                offloaded=offloaded,
                layers=layers,
            )

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None

        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("llama_cpp_model_unloaded")

    def analyze_segments(
        self, transcript: List[Dict[str, Any]], prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Analyze segments using a local llama.cpp GGUF model."""
        logger.info(
            "llama_cpp_analysis_start",
            model=self.config.model or self.config.model_repo,
            segment_count=len(transcript),
        )

        self._load_model()

        if prompt is None:
            prompt = self._build_prompt(transcript)

        if self.config.disable_thinking:
            prompt = f"{prompt}\n/no_think"

        completion_kwargs: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if self.config.seed is not None:
            completion_kwargs["seed"] = self.config.seed

        try:
            start = time.monotonic()
            response = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a content analysis expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **completion_kwargs,
            )
            elapsed = time.monotonic() - start
            response_text = response["choices"][0]["message"]["content"]

            usage = response.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or 0)
            tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

            logger.info(
                "llama_cpp_analysis_complete",
                completion_tokens=completion_tokens,
                elapsed_sec=round(elapsed, 2),
                tokens_per_sec=round(tokens_per_sec, 2),
            )

            self._calls += 1
            self._check_throughput(tokens_per_sec)

            return self._parse_response(response_text)
        except LLMAdapterError:
            raise
        except Exception as e:
            logger.error("llama_cpp_inference_error", error=str(e))
            raise LLMAdapterError(f"llama.cpp inference failed: {e}") from e


def create_llm_adapter(
    config: LLMConfig,
    work_dir: Optional[Union[str, Path]] = None,
    stage: Optional[str] = None,
) -> LLMAdapter:
    """
    Factory function to create the appropriate LLM adapter.

    Args:
        config: LLM configuration
        work_dir: Job directory: a local model only partly on the GPU becomes a
            notice of ``stage`` there
        stage: Pipeline stage that uses the LLM (scoring, chapters, broll)

    Returns:
        Configured LLMAdapter instance

    Raises:
        LLMAdapterError: If provider is unsupported
    """
    provider = config.provider.lower()

    if provider == "openai":
        logger.info("creating_adapter", provider="openai", model=config.model)
        return OpenAIAdapter(config)
    elif provider == "anthropic":
        logger.info("creating_adapter", provider="anthropic", model=config.model)
        return AnthropicAdapter(config)
    elif provider == "qwen_local":
        logger.info("creating_adapter", provider="qwen_local", model=config.model)
        return QwenLocalAdapter(config)
    elif provider == "llama_cpp":
        logger.info("creating_adapter", provider="llama_cpp", model=config.model)
        return LlamaCppAdapter(config, work_dir=work_dir, stage=stage)
    else:
        raise LLMAdapterError(f"Unsupported LLM provider: {provider}")
