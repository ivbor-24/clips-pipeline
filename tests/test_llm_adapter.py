"""
Tests for TASK-P1-05: LLM Content Analysis Adapter and Scoring Integration
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLLMAdapterModule:
    """Test llm_adapter module structure."""

    def test_module_exists(self):
        from src import llm_adapter

        assert llm_adapter is not None

    def test_llm_adapter_error_exists(self):
        from src.llm_adapter import LLMAdapterError

        assert issubclass(LLMAdapterError, Exception)

    def test_abstract_base_class_exists(self):
        from src.llm_adapter import LLMAdapter

        assert LLMAdapter is not None

    def test_openai_adapter_exists(self):
        from src.llm_adapter import OpenAIAdapter

        assert OpenAIAdapter is not None

    def test_anthropic_adapter_exists(self):
        from src.llm_adapter import AnthropicAdapter

        assert AnthropicAdapter is not None

    def test_qwen_local_adapter_exists(self):
        from src.llm_adapter import QwenLocalAdapter

        assert QwenLocalAdapter is not None

    def test_llama_cpp_adapter_exists(self):
        from src.llm_adapter import LlamaCppAdapter

        assert LlamaCppAdapter is not None

    def test_factory_function_exists(self):
        from src.llm_adapter import create_llm_adapter

        assert callable(create_llm_adapter)


class TestLLMConfig:
    """Test LLM configuration."""

    def test_llm_config_defaults(self):
        from src.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.enabled is False
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4"
        assert cfg.api_key is None
        assert cfg.api_base is None
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1000
        assert cfg.llm_weight == 0.4

    def test_llm_config_custom_values(self):
        from src.config import LLMConfig

        cfg = LLMConfig(
            enabled=True,
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            api_key="test-key",
            temperature=0.5,
            max_tokens=500,
            llm_weight=0.6,
        )
        assert cfg.enabled is True
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-3-sonnet-20240229"
        assert cfg.api_key == "test-key"
        assert cfg.temperature == 0.5
        assert cfg.llm_weight == 0.6

    def test_scoring_config_has_llm(self):
        from src.config import LLMConfig, ScoringConfig

        cfg = ScoringConfig()
        assert hasattr(cfg, "llm")
        assert isinstance(cfg.llm, LLMConfig)
        assert cfg.llm.enabled is False

    def test_config_yaml_loads_llm(self):
        from src.config import Config

        cfg = Config.load("config/config.yaml")
        assert hasattr(cfg.scoring, "llm")
        assert cfg.scoring.llm.enabled is True
        assert cfg.scoring.llm.provider == "llama_cpp"
        assert cfg.scoring.llm.model_repo == "bartowski/Qwen_Qwen3-14B-GGUF"
        assert cfg.scoring.llm.model_file == "Qwen_Qwen3-14B-Q4_K_M.gguf"
        assert cfg.scoring.llm.min_tokens_per_sec == 10.0

    def test_llm_config_qwen_local(self):
        from src.config import LLMConfig

        cfg = LLMConfig(
            enabled=True,
            provider="qwen_local",
            model="Qwen/Qwen2.5-7B-Instruct",
        )
        assert cfg.provider == "qwen_local"
        assert cfg.model == "Qwen/Qwen2.5-7B-Instruct"


class TestCreateLLMAdapter:
    """Test factory function."""

    def test_creates_openai_adapter(self):
        from src.config import LLMConfig
        from src.llm_adapter import OpenAIAdapter, create_llm_adapter

        cfg = LLMConfig(provider="openai", model="gpt-4")
        adapter = create_llm_adapter(cfg)
        assert isinstance(adapter, OpenAIAdapter)

    def test_creates_anthropic_adapter(self):
        from src.config import LLMConfig
        from src.llm_adapter import AnthropicAdapter, create_llm_adapter

        cfg = LLMConfig(provider="anthropic", model="claude-3-sonnet-20240229")
        adapter = create_llm_adapter(cfg)
        assert isinstance(adapter, AnthropicAdapter)

    def test_creates_qwen_local_adapter(self):
        from src.config import LLMConfig
        from src.llm_adapter import QwenLocalAdapter, create_llm_adapter

        cfg = LLMConfig(provider="qwen_local", model="Qwen/Qwen2.5-7B-Instruct")
        adapter = create_llm_adapter(cfg)
        assert isinstance(adapter, QwenLocalAdapter)

    def test_unsupported_provider_raises(self):
        from src.config import LLMConfig
        from src.llm_adapter import LLMAdapterError, create_llm_adapter

        cfg = LLMConfig(provider="openai")
        cfg.provider = "unsupported"
        with pytest.raises(LLMAdapterError, match="Unsupported LLM provider"):
            create_llm_adapter(cfg)


class TestOpenAIAdapter:
    """Test OpenAI adapter with mocked API."""

    def _make_adapter(self, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import OpenAIAdapter

        cfg = LLMConfig(provider="openai", api_key="test-key", **kwargs)
        return OpenAIAdapter(cfg)

    def test_init(self):
        adapter = self._make_adapter()
        assert adapter.config.provider == "openai"
        assert adapter._client is None

    @patch("src.llm_adapter.OpenAIAdapter._get_client")
    def test_analyze_segments_success(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "segments": [
                    {
                        "id": "seg_1",
                        "llm_score": 0.85,
                        "content_type": "definition",
                        "key_concepts": ["quantum", "entanglement"],
                        "virality_score": 0.7,
                        "self_containment_llm": 0.9,
                        "reasoning": "Clear definition",
                    }
                ]
            }
        )
        mock_response.usage = MagicMock(total_tokens=100)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        adapter = self._make_adapter()
        transcript = [
            {"id": "seg_1", "start": 0, "end": 60, "text": "Quantum entanglement is a phenomenon"}
        ]

        results = adapter.analyze_segments(transcript)

        assert len(results) == 1
        assert results[0]["id"] == "seg_1"
        assert results[0]["llm_score"] == 0.85
        assert results[0]["content_type"] == "definition"

    @patch("src.llm_adapter.OpenAIAdapter._get_client")
    def test_analyze_segments_api_error(self, mock_get_client):
        from src.llm_adapter import LLMAdapterError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit")
        mock_get_client.return_value = mock_client

        adapter = self._make_adapter()
        transcript = [{"id": "seg_1", "start": 0, "end": 60, "text": "test"}]

        with pytest.raises(LLMAdapterError, match="OpenAI API call failed"):
            adapter.analyze_segments(transcript)

    def test_missing_api_key_raises(self):
        from src.llm_adapter import LLMAdapterError

        adapter = self._make_adapter()
        adapter.config.api_key = None
        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(LLMAdapterError, match="API key not provided"):
                    adapter._get_client()

    def test_load_prompt_missing_file(self):
        from src.llm_adapter import LLMAdapterError

        adapter = self._make_adapter(prompt_file="/nonexistent/prompt.txt")
        with pytest.raises(LLMAdapterError, match="Prompt file not found"):
            adapter._load_prompt()

    def test_load_prompt_from_file(self, tmp_path):
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("Analyze {{SEGMENTS}} please")
        adapter = self._make_adapter(prompt_file=str(prompt_file))
        prompt = adapter._load_prompt()
        assert "Analyze" in prompt

    def test_build_prompt(self, tmp_path):
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text("Analyze: {{SEGMENTS}}")
        adapter = self._make_adapter(prompt_file=str(prompt_file))
        segments = [{"id": "s1", "text": "hello"}]
        result = adapter._build_prompt(segments)
        assert '"id": "s1"' in result
        assert "{{SEGMENTS}}" not in result

    def test_parse_response_json(self):
        adapter = self._make_adapter()
        response = json.dumps({"segments": [{"id": "s1", "llm_score": 0.8}]})
        result = adapter._parse_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    def test_parse_response_code_block(self):
        adapter = self._make_adapter()
        response = '```json\n{"segments": [{"id": "s1", "llm_score": 0.9}]}\n```'
        result = adapter._parse_response(response)
        assert len(result) == 1

    def test_parse_response_list(self):
        adapter = self._make_adapter()
        response = json.dumps([{"id": "s1", "llm_score": 0.8}])
        result = adapter._parse_response(response)
        assert len(result) == 1

    def test_parse_response_invalid_json(self):
        from src.llm_adapter import LLMAdapterError

        adapter = self._make_adapter()
        with pytest.raises(LLMAdapterError, match="Failed to parse"):
            adapter._parse_response("not json at all")

    def test_parse_response_think_block(self):
        adapter = self._make_adapter()
        response = (
            "<think>\nreasoning about the segments\n</think>\n"
            '{"segments": [{"id": "s1", "llm_score": 0.8}]}'
        )
        result = adapter._parse_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    def test_parse_response_truncated_think_block_is_not_parsed(self):
        """Cut off by max_tokens inside <think>: a JSON draft there is not the answer."""
        from src.llm_adapter import LLMAdapterError

        adapter = self._make_adapter()
        response = (
            "<think>\nMaybe the output should be "
            '{"segments": [{"id": "s1", "llm_score": 0.1}]} but let me check seg'
        )
        with pytest.raises(LLMAdapterError, match="Failed to parse"):
            adapter._parse_response(response)

    def test_parse_response_prose_around_json(self):
        adapter = self._make_adapter()
        response = (
            "Here is the analysis:\n\n"
            '{"segments": [{"id": "s1", "llm_score": 0.7}]}\n\n'
            "Let me know if you need more."
        )
        result = adapter._parse_response(response)
        assert len(result) == 1
        assert result[0]["llm_score"] == 0.7

    def test_parse_response_think_fence_and_prose(self):
        adapter = self._make_adapter()
        response = (
            "<think>analyzing</think>\n"
            "Sure, here you go:\n"
            '```json\n{"segments": [{"id": "s1", "llm_score": 0.9}]}\n```\n'
            "Hope that helps!"
        )
        result = adapter._parse_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    def test_unload_is_noop(self):
        adapter = self._make_adapter()
        adapter.unload()


class TestAnthropicAdapter:
    """Test Anthropic adapter with mocked API."""

    def _make_adapter(self, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import AnthropicAdapter

        cfg = LLMConfig(provider="anthropic", api_key="test-key", **kwargs)
        return AnthropicAdapter(cfg)

    def test_init(self):
        adapter = self._make_adapter()
        assert adapter.config.provider == "anthropic"

    @patch("src.llm_adapter.AnthropicAdapter._get_client")
    def test_analyze_segments_success(self, mock_get_client):
        mock_content = MagicMock(type="text")
        mock_content.text = json.dumps(
            {
                "segments": [
                    {
                        "id": "seg_1",
                        "llm_score": 0.9,
                        "content_type": "story",
                        "key_concepts": ["physics"],
                        "virality_score": 0.8,
                        "self_containment_llm": 0.85,
                        "reasoning": "Engaging story",
                    }
                ]
            }
        )
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.usage = MagicMock(output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        adapter = self._make_adapter()
        transcript = [{"id": "seg_1", "start": 0, "end": 60, "text": "A story about physics"}]

        results = adapter.analyze_segments(transcript)
        assert len(results) == 1
        assert results[0]["llm_score"] == 0.9

    @patch("src.llm_adapter.AnthropicAdapter._get_client")
    def test_analyze_segments_api_error(self, mock_get_client):
        from src.llm_adapter import LLMAdapterError

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Rate limited")
        mock_get_client.return_value = mock_client

        adapter = self._make_adapter()
        with pytest.raises(LLMAdapterError, match="Anthropic API call failed"):
            adapter.analyze_segments([{"id": "s1", "start": 0, "end": 60, "text": "test"}])

    def test_missing_api_key_raises(self):
        from src.llm_adapter import LLMAdapterError

        adapter = self._make_adapter()
        adapter.config.api_key = None
        mock_anthropic_module = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(LLMAdapterError, match="API key not provided"):
                    adapter._get_client()


class TestQwenLocalAdapter:
    """Test Qwen local adapter with mocked model."""

    def _make_adapter(self, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import QwenLocalAdapter

        cfg = LLMConfig(provider="qwen_local", model="Qwen/Qwen2.5-7B-Instruct", **kwargs)
        return QwenLocalAdapter(cfg)

    def test_init(self):
        adapter = self._make_adapter()
        assert adapter.config.provider == "qwen_local"
        assert adapter._model is None
        assert adapter._tokenizer is None

    def test_unload_clears_model(self):
        adapter = self._make_adapter()
        adapter._model = MagicMock()
        adapter._tokenizer = MagicMock()
        adapter.unload()
        assert adapter._model is None
        assert adapter._tokenizer is None

    @patch("src.llm_adapter.QwenLocalAdapter._load_model")
    def test_analyze_segments_with_mocked_model(self, mock_load):
        adapter = self._make_adapter()

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 10)
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: mock_input_ids
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model.device = "cpu"

        mock_output_ids = MagicMock()
        mock_output_tensor = MagicMock()
        mock_output_tensor.__getitem__ = lambda self, key: mock_output_ids
        mock_model.generate.return_value = [mock_output_tensor]

        adapter._model = mock_model
        adapter._tokenizer = mock_tokenizer

        mock_tokenizer.decode.return_value = json.dumps(
            {"segments": [{"id": "seg_1", "llm_score": 0.75, "content_type": "explanation"}]}
        )

        transcript = [{"id": "seg_1", "start": 0, "end": 60, "text": "Physics is cool"}]

        with patch("torch.no_grad"):
            results = adapter.analyze_segments(transcript)
            assert len(results) == 1
            assert results[0]["llm_score"] == 0.75


class TestLlamaCppAdapter:
    """Test llama.cpp adapter with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _skip_checksum(self, request):
        """Fake model files here fail the real pin; TestLlamaCppModelPin covers it."""
        with patch("src.llm_adapter.verify_file"):
            yield

    def _make_adapter(self, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import LlamaCppAdapter

        cfg = LLMConfig(
            provider="llama_cpp",
            model_repo="google/gemma-4-E4B-it-qat-q4_0-gguf",
            model_file="gemma-4-E4B_q4_0-it.gguf",
            **kwargs,
        )
        return LlamaCppAdapter(cfg)

    def test_factory_creates_llama_cpp_adapter(self):
        from src.config import LLMConfig
        from src.llm_adapter import LlamaCppAdapter, create_llm_adapter

        cfg = LLMConfig(provider="llama_cpp")
        adapter = create_llm_adapter(cfg)
        assert isinstance(adapter, LlamaCppAdapter)

    def test_init(self):
        adapter = self._make_adapter()
        assert adapter.config.provider == "llama_cpp"
        assert adapter._llm is None

    def test_resolve_model_path_from_local(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("dummy")
        adapter = self._make_adapter(model_path=str(model_file))
        resolved = adapter._resolve_model_path()
        assert resolved == model_file

    @patch("src.llm_adapter._hf_hub_download")
    def test_resolve_model_path_missing_local_downloads(self, mock_download, tmp_path):
        """A missing local model_path now falls back to downloading."""

        cached = tmp_path / "downloaded.gguf"
        cached.write_text("dummy")
        mock_download.return_value = str(cached)

        adapter = self._make_adapter(model_path=str(tmp_path / "missing.gguf"))
        resolved = adapter._resolve_model_path()

        assert resolved == cached
        mock_download.assert_called_once()

    @patch("src.llm_adapter._hf_hub_download")
    def test_resolve_model_path_downloads(self, mock_download, tmp_path):

        cached = tmp_path / "downloaded.gguf"
        cached.write_text("dummy")
        mock_download.return_value = str(cached)

        adapter = self._make_adapter()
        resolved = adapter._resolve_model_path()
        assert resolved == cached
        mock_download.assert_called_once()

    @patch("src.llm_adapter._hf_hub_download")
    def test_resolve_model_path_download_disabled_raises(self, mock_download, tmp_path):
        from src.llm_adapter import LLMAdapterError

        adapter = self._make_adapter(download_if_missing=False)
        with pytest.raises(LLMAdapterError, match="download_if_missing=False"):
            adapter._resolve_model_path()
        mock_download.assert_not_called()

    @patch("src.llm_adapter._hf_hub_download")
    def test_resolve_model_path_downloads_links_configured_model_path(
        self, mock_download, tmp_path
    ):
        """Downloading when model_path is configured but missing should leave
        a symlink at model_path pointing at the cached download, so a repeat
        run takes the fast local-file branch instead of re-downloading."""
        cached = tmp_path / "hf_cache" / "downloaded.gguf"
        cached.parent.mkdir()
        cached.write_text("dummy")
        mock_download.return_value = str(cached)

        configured_path = tmp_path / "models" / "model.gguf"
        adapter = self._make_adapter(model_path=str(configured_path))
        resolved = adapter._resolve_model_path()

        assert resolved == cached
        assert configured_path.is_symlink()
        assert configured_path.resolve() == cached.resolve()

    @patch("src.llm_adapter._hf_hub_download")
    def test_resolve_model_path_link_failure_is_non_fatal(self, mock_download, tmp_path):
        """A symlink failure (e.g. cross-device) must not break resolution."""
        cached = tmp_path / "downloaded.gguf"
        cached.write_text("dummy")
        mock_download.return_value = str(cached)

        configured_path = tmp_path / "models" / "model.gguf"
        adapter = self._make_adapter(model_path=str(configured_path))

        with patch.object(Path, "symlink_to", side_effect=OSError("cross-device link")):
            resolved = adapter._resolve_model_path()

        assert resolved == cached
        assert not configured_path.exists()

    @patch("src.llm_adapter._create_llama")
    def test_load_model_uses_local_path(self, mock_create_llama, tmp_path):

        model_file = tmp_path / "model.gguf"
        model_file.write_text("dummy")
        mock_llama = MagicMock()
        mock_create_llama.return_value = mock_llama

        adapter = self._make_adapter(model_path=str(model_file))
        adapter._load_model()

        assert adapter._llm is mock_llama
        mock_create_llama.assert_called_once()
        call_kwargs = mock_create_llama.call_args.kwargs
        assert call_kwargs["model_path"] == str(model_file)
        assert call_kwargs["n_ctx"] == adapter.config.n_ctx
        assert call_kwargs["n_gpu_layers"] == adapter.config.n_gpu_layers

    @patch("src.llm_adapter._create_llama")
    def test_load_model_passes_chat_format(self, mock_create_llama, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("dummy")
        mock_create_llama.return_value = MagicMock()

        adapter = self._make_adapter(model_path=str(model_file), chat_format="gemma4")
        adapter._load_model()

        call_kwargs = mock_create_llama.call_args.kwargs
        assert call_kwargs["chat_format"] == "gemma4"

    @patch("src.llm_adapter._create_llama")
    def test_load_model_omits_null_chat_format(self, mock_create_llama, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_text("dummy")
        mock_create_llama.return_value = MagicMock()

        adapter = self._make_adapter(model_path=str(model_file), chat_format=None)
        adapter._load_model()

        call_kwargs = mock_create_llama.call_args.kwargs
        assert "chat_format" not in call_kwargs

    def test_analyze_segments_with_mocked_llm(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text('{"segments": [{{SEGMENTS}}]}')

        adapter = self._make_adapter(
            model_path=str(tmp_path / "model.gguf"),
            prompt_file=str(prompt_file),
        )
        adapter._llm = MagicMock()
        adapter._llm.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "segments": [
                                    {
                                        "id": "seg_1",
                                        "llm_score": 0.88,
                                        "content_type": "definition",
                                        "key_concepts": ["foo"],
                                        "virality_score": 0.7,
                                        "self_containment_llm": 0.9,
                                        "reasoning": "Good",
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
            "usage": {"completion_tokens": 50, "prompt_tokens": 100},
        }

        transcript = [{"id": "seg_1", "start": 0, "end": 60, "text": "Hello"}]
        results = adapter.analyze_segments(transcript)

        assert len(results) == 1
        assert results[0]["llm_score"] == 0.88
        adapter._llm.create_chat_completion.assert_called_once()

    @patch("src.llm_adapter.time.monotonic", side_effect=[0.0, 100.0])
    def test_analyze_segments_low_throughput_raises(self, mock_time, tmp_path):
        """Hopeless throughput (1 token in 100 s) must raise -> heuristics fallback."""
        from src.llm_adapter import LLMAdapterError

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text('{"segments": [{{SEGMENTS}}]}')

        adapter = self._make_adapter(
            model_path=str(tmp_path / "model.gguf"),
            prompt_file=str(prompt_file),
            min_tokens_per_sec=10.0,
        )
        adapter._llm = MagicMock()
        adapter._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"segments": []}'}}],
            "usage": {"completion_tokens": 1, "prompt_tokens": 10},
        }

        transcript = [{"id": "seg_1", "start": 0, "end": 60, "text": "Hello"}]
        with pytest.raises(LLMAdapterError, match="throughput too low"):
            adapter.analyze_segments(transcript)

    def test_unload_clears_model(self):
        adapter = self._make_adapter()
        adapter._llm = MagicMock()
        adapter.unload()
        assert adapter._llm is None


class TestApplyLLMScores:
    """Test combining heuristic and LLM scores."""

    def test_apply_llm_scores_basic(self):
        from src.scoring import ScoredSegment, apply_llm_scores

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="test", score=0.6),
            ScoredSegment(id="s2", start=60, end=120, text="test2", score=0.4),
        ]
        llm_results = [
            {"id": "s1", "llm_score": 0.9, "content_type": "definition", "virality_score": 0.5},
            {"id": "s2", "llm_score": 0.3, "content_type": "story", "virality_score": 0.8},
        ]

        result = apply_llm_scores(segments, llm_results, llm_weight=0.4)

        assert len(result) == 2
        expected_s1 = round(0.6 * 0.6 + 0.9 * 0.4, 3)
        assert result[0].score == expected_s1
        assert "llm:definition" in result[0].tags
        assert "viral_potential" in result[1].tags

    def test_apply_llm_scores_missing_segment(self):
        from src.scoring import ScoredSegment, apply_llm_scores

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="test", score=0.6),
        ]
        llm_results = [
            {"id": "nonexistent", "llm_score": 0.9},
        ]

        result = apply_llm_scores(segments, llm_results, llm_weight=0.4)
        assert result[0].score == 0.6

    def test_apply_llm_scores_zero_weight(self):
        from src.scoring import ScoredSegment, apply_llm_scores

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="test", score=0.6),
        ]
        llm_results = [{"id": "s1", "llm_score": 0.9}]

        result = apply_llm_scores(segments, llm_results, llm_weight=0.0)
        assert result[0].score == 0.6

    def test_apply_llm_scores_full_weight(self):
        from src.scoring import ScoredSegment, apply_llm_scores

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="test", score=0.6),
        ]
        llm_results = [{"id": "s1", "llm_score": 0.9}]

        result = apply_llm_scores(segments, llm_results, llm_weight=1.0)
        assert result[0].score == 0.9

    def test_apply_llm_scores_invalid_llm_score(self):
        from src.scoring import ScoredSegment, apply_llm_scores

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="test", score=0.6),
        ]
        llm_results = [{"id": "s1", "llm_score": "not_a_number"}]

        result = apply_llm_scores(segments, llm_results, llm_weight=0.4)
        assert result[0].score == 0.6

    def test_apply_llm_scores_empty(self):
        from src.scoring import ScoredSegment, apply_llm_scores

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="test", score=0.6),
        ]
        result = apply_llm_scores(segments, [], llm_weight=0.4)
        assert result[0].score == 0.6


class TestScoringLLMIntegration:
    """Test scoring pipeline with LLM integration."""

    def test_scoring_with_llm_disabled(self, tmp_path):
        from src.config import Config
        from src.scoring import score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        transcript_data = [
            {
                "id": "seg_001",
                "start": 0.0,
                "end": 60.0,
                "text": "Quantum entanglement is a physical phenomenon where particles are correlated",
            },
        ]
        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text(json.dumps(transcript_data))

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))
            cfg.scoring.llm.enabled = False
            assert cfg.scoring.llm.enabled is False

            result = score_transcript(cfg, dry_run=False)
            assert result is not None
        finally:
            os.chdir(original_cwd)

    def test_scoring_with_llm_enabled_mock(self, tmp_path):
        from src.config import Config, LLMConfig
        from src.scoring import score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        transcript_data = [
            {
                "id": "seg_001",
                "start": 0.0,
                "end": 60.0,
                "text": "Quantum entanglement is a physical phenomenon",
            },
        ]
        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text(json.dumps(transcript_data))

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))
            cfg.scoring.strategy = "segment_merge"  # written for per-segment scoring
            cfg.scoring.llm = LLMConfig(
                enabled=True,
                provider="openai",
                api_key="fake-key",
            )

            mock_llm_results = [
                {
                    "id": "seg_001",
                    "llm_score": 0.95,
                    "content_type": "definition",
                    "key_concepts": ["quantum", "entanglement"],
                    "virality_score": 0.8,
                    "self_containment_llm": 0.9,
                    "reasoning": "Clear definition",
                }
            ]

            with patch("src.llm_adapter.create_llm_adapter") as mock_factory:
                mock_adapter = MagicMock()
                mock_adapter.analyze_segments.return_value = mock_llm_results
                mock_factory.return_value = mock_adapter

                result = score_transcript(cfg, dry_run=False)

                assert result is not None
                mock_factory.assert_called_once()
                mock_adapter.analyze_segments.assert_called_once()
                mock_adapter.unload.assert_called_once()
        finally:
            os.chdir(original_cwd)

    def test_scoring_llm_failure_fallback(self, tmp_path):
        from src.config import Config, LLMConfig
        from src.scoring import score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        transcript_data = [
            {
                "id": "seg_001",
                "start": 0.0,
                "end": 60.0,
                "text": "Quantum entanglement is a physical phenomenon",
            },
        ]
        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text(json.dumps(transcript_data))

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))
            cfg.scoring.strategy = "segment_merge"  # written for per-segment scoring
            cfg.scoring.llm = LLMConfig(enabled=True, provider="openai", api_key="fake")

            with patch("src.llm_adapter.create_llm_adapter") as mock_factory:
                mock_factory.side_effect = Exception("LLM unavailable")

                result = score_transcript(cfg, dry_run=False)

                assert result is not None
        finally:
            os.chdir(original_cwd)

    def test_dry_run_skips_llm(self, tmp_path):
        from src.config import Config, LLMConfig
        from src.scoring import score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text('[{"id": "seg_001", "start": 0, "end": 60, "text": "Test"}]')

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))
            cfg.scoring.strategy = "segment_merge"  # written for per-segment scoring
            cfg.scoring.llm = LLMConfig(enabled=True)

            result = score_transcript(cfg, dry_run=True)
            assert result[0]["status"] == "dry_run_passed"
        finally:
            os.chdir(original_cwd)


class TestPromptFile:
    """Test prompt file exists and has correct format."""

    def test_prompt_file_exists(self):
        prompt_path = Path("config/prompts/segment_analysis_v1.txt")
        assert prompt_path.exists()

    def test_prompt_has_segments_placeholder(self):
        prompt_path = Path("config/prompts/segment_analysis_v1.txt")
        content = prompt_path.read_text()
        assert "{{SEGMENTS}}" in content

    def test_prompt_mentions_json_format(self):
        prompt_path = Path("config/prompts/segment_analysis_v1.txt")
        content = prompt_path.read_text()
        assert "JSON" in content or "json" in content

    def test_prompt_mentions_scoring_fields(self):
        prompt_path = Path("config/prompts/segment_analysis_v1.txt")
        content = prompt_path.read_text()
        assert "llm_score" in content
        assert "content_type" in content
        assert "virality_score" in content
        assert "self_containment_llm" in content


class TestScoringLLMBatchFailures:
    """B1: a failed LLM batch must not discard the other batches."""

    N_SEGMENTS = 40  # 3 batches: 15 + 15 + 10

    def _config(self, tmp_path):
        from src.config import Config, LLMConfig, ScoringConfig

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        transcript = [
            {
                "id": f"seg_{i:03d}",
                "start": float(i * 5),
                "end": float(i * 5 + 5),
                "text": f"Квантовая запутанность — это физическое явление номер {i}",
            }
            for i in range(self.N_SEGMENTS)
        ]
        (artifacts_dir / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
        )
        return Config(
            work_dir=tmp_path,
            scoring=ScoringConfig(
                strategy="segment_merge",  # batches of segments, not windows
                term_extraction_method="statistical",
                llm=LLMConfig(enabled=True, provider="openai", api_key="fake"),
            ),
        )

    @staticmethod
    def _results(batch):
        return [{"id": s["id"], "llm_score": 0.9} for s in batch]

    def _run(self, cfg, side_effect):
        from src.scoring import apply_llm_scores, score_transcript

        adapter = MagicMock()
        adapter.analyze_segments.side_effect = side_effect
        with (
            patch("src.llm_adapter.create_llm_adapter", return_value=adapter),
            patch("src.scoring.apply_llm_scores", wraps=apply_llm_scores) as apply_mock,
        ):
            result = score_transcript(cfg, dry_run=False)
        return result, adapter, apply_mock

    def test_failed_batch_is_skipped(self, tmp_path):
        from src.llm_adapter import LLMAdapterError

        calls = []

        def analyze(batch):
            calls.append(batch)
            if len(calls) == 2:
                raise LLMAdapterError("Failed to parse LLM response as JSON")
            return self._results(batch)

        result, adapter, apply_mock = self._run(self._config(tmp_path), analyze)

        assert result is not None
        assert adapter.analyze_segments.call_count == 3
        applied_ids = [r["id"] for r in apply_mock.call_args.args[1]]
        expected = [s["id"] for s in calls[0] + calls[2]]
        assert applied_ids == expected
        adapter.unload.assert_called_once()

        # B9: raw results are persisted next to the heuristic scores
        saved = json.loads((tmp_path / "artifacts" / "llm_analysis.json").read_text())
        assert saved["batches"] == {"total": 3, "failed": 1}
        assert [r["id"] for r in saved["results"]] == expected
        first = saved["results"][0]
        assert first["llm_score"] == 0.9
        assert (first["start"], first["end"]) == (0.0, 5.0)
        assert 0.0 <= first["heuristic_score"] < 0.9

    def test_majority_failure_falls_back_to_heuristics(self, tmp_path):
        from src.llm_adapter import LLMAdapterError

        result, adapter, apply_mock = self._run(
            self._config(tmp_path), LLMAdapterError("inference failed")
        )

        assert result is not None
        # 2 of 3 failed -> majority is lost, the 3rd batch is not attempted
        assert adapter.analyze_segments.call_count == 2
        apply_mock.assert_not_called()
        adapter.unload.assert_called_once()

    def test_unavailable_backend_aborts_immediately(self, tmp_path):
        from src.llm_adapter import LLMUnavailableError

        result, adapter, apply_mock = self._run(
            self._config(tmp_path), LLMUnavailableError("Failed to load llama.cpp model")
        )

        assert result is not None
        assert adapter.analyze_segments.call_count == 1
        apply_mock.assert_not_called()


class TestLlamaCppThroughputPolicy:
    """The first call is a warm-up (shader compilation); the second decides."""

    def _adapter(self, tmp_path):
        from src.config import LLMConfig
        from src.llm_adapter import LlamaCppAdapter

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("{{SEGMENTS}}")
        adapter = LlamaCppAdapter(
            LLMConfig(
                provider="llama_cpp",
                model_path=str(tmp_path / "model.gguf"),
                prompt_file=str(prompt_file),
                min_tokens_per_sec=10.0,
            )
        )
        adapter._llm = MagicMock()
        adapter._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"segments": [{"id": "seg_1"}]}'}}],
            "usage": {"completion_tokens": 100},
        }
        return adapter

    def test_hopeless_first_call_means_backend_unavailable(self, tmp_path):
        from src.llm_adapter import LLMUnavailableError

        adapter = self._adapter(tmp_path)
        # 100 tokens in 100 s = 1 tok/s: a CPU run, not a cold GPU
        with patch("src.llm_adapter.time.monotonic", side_effect=[0.0, 100.0]):
            with pytest.raises(LLMUnavailableError, match="throughput too low"):
                adapter.analyze_segments([{"id": "seg_1", "text": "x"}])

    def test_cold_first_call_is_a_warmup(self, tmp_path):
        """Found in Docker on Arc: 7.65 tok/s while Vulkan compiled its shaders."""
        adapter = self._adapter(tmp_path)
        segs = [{"id": "seg_1", "text": "x"}]
        # first call 100 tokens in 13 s (7.7 tok/s), second in 5 s (20 tok/s)
        with patch("src.llm_adapter.time.monotonic", side_effect=[0.0, 13.0, 0.0, 5.0]):
            assert adapter.analyze_segments(segs) == [{"id": "seg_1"}]
            assert adapter.analyze_segments(segs) == [{"id": "seg_1"}]

    def test_second_call_decides(self, tmp_path):
        from src.llm_adapter import LLMUnavailableError

        adapter = self._adapter(tmp_path)
        segs = [{"id": "seg_1", "text": "x"}]
        # both calls at 7.7 tok/s: not a warm-up effect, the backend is slow
        with patch("src.llm_adapter.time.monotonic", side_effect=[0.0, 13.0, 0.0, 13.0]):
            adapter.analyze_segments(segs)
            with pytest.raises(LLMUnavailableError, match="throughput too low"):
                adapter.analyze_segments(segs)

    def test_slow_later_call_keeps_result(self, tmp_path):
        adapter = self._adapter(tmp_path)
        segs = [{"id": "seg_1", "text": "x"}]
        # two fast calls, then 100 tokens in 100 s
        with patch("src.llm_adapter.time.monotonic", side_effect=[0.0, 1.0, 0.0, 1.0, 0.0, 100.0]):
            adapter.analyze_segments(segs)
            adapter.analyze_segments(segs)
            result = adapter.analyze_segments(segs)
        assert result == [{"id": "seg_1"}]

    def test_model_load_failure_is_unavailable(self, tmp_path):
        from src.llm_adapter import LLMUnavailableError

        adapter = self._adapter(tmp_path)
        adapter._llm = None
        (tmp_path / "model.gguf").write_text("fake")
        with patch("src.llm_adapter._create_llama", side_effect=RuntimeError("vk OOM")):
            with pytest.raises(LLMUnavailableError, match="Failed to load"):
                adapter.analyze_segments([{"id": "seg_1", "text": "x"}])


class TestLlamaCppModelPin:
    """Pinned revision and checksum from config/models.lock.yaml."""

    def _adapter(self, tmp_path, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import LlamaCppAdapter

        cfg = LLMConfig(
            provider="llama_cpp",
            model_repo="org/repo",
            model_file="model.gguf",
            **kwargs,
        )
        return LlamaCppAdapter(cfg)

    def _pin(self, content: bytes):
        import hashlib

        from src.model_registry import PinnedFile

        return PinnedFile(
            repo="org/repo",
            file="model.gguf",
            revision="abc123",
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )

    @patch("src.llm_adapter._hf_hub_download")
    def test_download_uses_pinned_revision_and_verifies(self, mock_download, tmp_path):
        cached = tmp_path / "model.gguf"
        cached.write_bytes(b"gguf weights")
        mock_download.return_value = str(cached)

        with patch("src.llm_adapter.find_pinned_file", return_value=self._pin(b"gguf weights")):
            resolved = self._adapter(tmp_path)._resolve_model_path()

        assert resolved == cached
        assert mock_download.call_args.kwargs["revision"] == "abc123"
        assert (tmp_path / "model.gguf.sha256").exists()

    @patch("src.llm_adapter._hf_hub_download")
    def test_checksum_mismatch_raises(self, mock_download, tmp_path):
        from src.llm_adapter import LLMAdapterError

        cached = tmp_path / "model.gguf"
        cached.write_bytes(b"tampered!!!!")  # same size, different bytes
        mock_download.return_value = str(cached)

        with patch("src.llm_adapter.find_pinned_file", return_value=self._pin(b"gguf weights")):
            with pytest.raises(LLMAdapterError, match="sha256"):
                self._adapter(tmp_path)._resolve_model_path()

    @patch("src.llm_adapter._hf_hub_download")
    def test_unpinned_model_downloads_latest(self, mock_download, tmp_path):
        cached = tmp_path / "model.gguf"
        cached.write_bytes(b"x")
        mock_download.return_value = str(cached)

        with patch("src.llm_adapter.find_pinned_file", return_value=None):
            self._adapter(tmp_path)._resolve_model_path()

        assert mock_download.call_args.kwargs["revision"] is None

    def test_local_file_size_mismatch_only_warns(self, tmp_path):
        local = tmp_path / "local.gguf"
        local.write_bytes(b"short")

        with patch("src.llm_adapter.find_pinned_file", return_value=self._pin(b"gguf weights")):
            resolved = self._adapter(tmp_path, model_path=str(local))._resolve_model_path()

        assert resolved == local


class TestGpuOffloadReport:
    """llama.cpp's log says how many layers went to the GPU."""

    LOG = (
        "llama_prepare_model_devices: using device Vulkan0 (Intel(R) Arc(tm) B580 Graphics "
        "(BMG G21)) (0000:05:00.0) - 10210 MiB free\n"
        "load_tensors: offloaded {n}/41 layers to GPU\n"
    )

    def test_parse(self):
        from src.llm_adapter import parse_gpu_offload

        assert parse_gpu_offload(self.LOG.format(n=41)) == {
            "offloaded": 41,
            "layers": 41,
            "device": "Vulkan0 (Intel(R) Arc(tm) B580 Graphics (BMG G21))",
            "free_mib": 10210,
        }
        assert parse_gpu_offload("no such line") is None

    def make_adapter(self, tmp_path):
        from src.config import LLMConfig
        from src.llm_adapter import LlamaCppAdapter

        return LlamaCppAdapter(
            LLMConfig(enabled=True, provider="llama_cpp"), work_dir=tmp_path, stage="scoring"
        )

    def test_partly_on_cpu_is_a_notice(self, tmp_path):
        from src.notices import read_notices

        adapter = self.make_adapter(tmp_path)
        adapter._report_gpu_offload(self.LOG.format(n=30))
        assert adapter.gpu_offload["offloaded"] == 30
        notices = read_notices(tmp_path)
        assert [n["code"] for n in notices] == ["llm_partly_on_cpu"]
        assert "30 of 41" in notices[0]["message"]

    def test_fully_on_gpu_is_no_notice(self, tmp_path):
        from src.notices import read_notices

        adapter = self.make_adapter(tmp_path)
        adapter._report_gpu_offload(self.LOG.format(n=41))
        assert read_notices(tmp_path) == []

    def test_log_is_captured_while_loading(self):
        """The capture hook hands every llama.cpp line to llama-cpp-python's own handler too."""
        import ctypes

        llama_cpp = pytest.importorskip("llama_cpp")
        from llama_cpp import _logger

        from src.llm_adapter import _capture_llama_log

        lines = []
        installed = []
        with (
            patch.object(
                llama_cpp, "llama_log_set", side_effect=lambda cb, _: installed.append(cb)
            ),
            patch.object(_logger, "llama_log_callback") as default,
        ):
            with _capture_llama_log(lines):
                installed[0](
                    1, b"load_tensors: offloaded 41/41 layers to GPU\n", ctypes.c_void_p(0)
                )
        assert lines == ["load_tensors: offloaded 41/41 layers to GPU\n"]
        default.assert_called_once()
        # llama-cpp-python's handler is put back.
        assert installed[-1] is default


class FakeAPIError(Exception):
    """An API client error with an HTTP status, as both SDKs raise."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def _openai_response(text, finish_reason="stop"):
    choice = MagicMock(finish_reason=finish_reason)
    choice.message.content = text
    return MagicMock(choices=[choice])


class TestOpenAIAdapterRequests:
    """Models differ in the parameters they take; the adapter adapts once per run."""

    def _adapter(self, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(LLMConfig(provider="openai", model="gpt-test", **kwargs))
        adapter._client = MagicMock()
        return adapter

    def test_first_request_uses_the_config(self):
        adapter = self._adapter(max_tokens=4000, temperature=0.2)
        create = adapter._client.chat.completions.create
        create.return_value = _openai_response('{"results": [{"ok": true}]}')

        assert adapter.analyze_segments([], prompt="p") == [{"ok": True}]

        kwargs = create.call_args.kwargs
        assert kwargs["max_tokens"] == 4000
        assert kwargs["temperature"] == 0.2
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_reasoning_model_gets_max_completion_tokens_and_keeps_it(self):
        adapter = self._adapter(max_tokens=4000)
        create = adapter._client.chat.completions.create
        create.side_effect = [
            FakeAPIError(400, "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'"),
            FakeAPIError(400, "Unsupported value: 'temperature' does not support 0.2"),
            _openai_response('{"results": []}'),
            _openai_response('{"results": []}'),
        ]

        adapter.analyze_segments([], prompt="p")
        adapter.analyze_segments([], prompt="p")

        assert create.call_count == 4
        last = create.call_args.kwargs
        assert "max_tokens" not in last
        assert "temperature" not in last
        assert last["max_completion_tokens"] == 16000

    def test_service_without_json_mode(self):
        adapter = self._adapter()
        create = adapter._client.chat.completions.create
        create.side_effect = [
            FakeAPIError(400, "response_format is not supported"),
            _openai_response('{"results": []}'),
        ]

        adapter.analyze_segments([], prompt="p")

        assert "response_format" not in create.call_args.kwargs

    def test_other_bad_request_is_not_repeated(self):
        from src.llm_adapter import LLMAdapterError, LLMUnavailableError

        adapter = self._adapter()
        create = adapter._client.chat.completions.create
        create.side_effect = FakeAPIError(400, "context length exceeded")

        with pytest.raises(LLMAdapterError) as error:
            adapter.analyze_segments([], prompt="p")

        assert create.call_count == 1
        assert not isinstance(error.value, LLMUnavailableError)

    @pytest.mark.parametrize("status", [401, 403, 404])
    def test_wrong_key_or_model_makes_the_llm_unavailable(self, status):
        from src.llm_adapter import LLMUnavailableError

        adapter = self._adapter()
        adapter._client.chat.completions.create.side_effect = FakeAPIError(status, "no")

        with pytest.raises(LLMUnavailableError):
            adapter.analyze_segments([], prompt="p")

    def test_server_error_fails_only_this_batch(self):
        from src.llm_adapter import LLMAdapterError, LLMUnavailableError

        adapter = self._adapter()
        adapter._client.chat.completions.create.side_effect = FakeAPIError(500, "oops")

        with pytest.raises(LLMAdapterError) as error:
            adapter.analyze_segments([], prompt="p")

        assert not isinstance(error.value, LLMUnavailableError)

    def test_empty_answer_names_the_finish_reason(self):
        from src.llm_adapter import LLMAdapterError

        adapter = self._adapter()
        adapter._client.chat.completions.create.return_value = _openai_response(None, "length")

        with pytest.raises(LLMAdapterError, match="length"):
            adapter.analyze_segments([], prompt="p")

    def test_missing_key_makes_the_llm_unavailable(self, monkeypatch):
        from src.config import LLMConfig
        from src.llm_adapter import LLMUnavailableError, OpenAIAdapter

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        adapter = OpenAIAdapter(LLMConfig(provider="openai", model="gpt-test"))

        with pytest.raises(LLMUnavailableError, match="OPENAI_API_KEY"):
            adapter.analyze_segments([], prompt="p")

    def test_api_base_is_passed_to_the_client(self, monkeypatch):
        from src.config import LLMConfig
        from src.llm_adapter import OpenAIAdapter

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        adapter = OpenAIAdapter(
            LLMConfig(provider="openai", model="m", api_base="https://api.example.com/v1")
        )

        assert str(adapter._get_client().base_url).startswith("https://api.example.com/v1")


class TestAnthropicAdapterRequests:
    def _adapter(self, **kwargs):
        from src.config import LLMConfig
        from src.llm_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(LLMConfig(provider="anthropic", model="claude-test", **kwargs))
        adapter._client = MagicMock()
        return adapter

    def test_no_sampling_parameters_and_room_for_thinking(self):
        adapter = self._adapter(max_tokens=4000)
        create = adapter._client.messages.create
        thinking = MagicMock(type="thinking", text="draft {not json}")
        answer = MagicMock(type="text", text='{"results": [{"ok": true}]}')
        create.return_value = MagicMock(content=[thinking, answer], stop_reason="end_turn")

        assert adapter.analyze_segments([], prompt="p") == [{"ok": True}]

        kwargs = create.call_args.kwargs
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 16000
        assert kwargs["model"] == "claude-test"

    def test_answer_without_text_names_the_stop_reason(self):
        from src.llm_adapter import LLMAdapterError

        adapter = self._adapter()
        adapter._client.messages.create.return_value = MagicMock(
            content=[MagicMock(type="thinking")], stop_reason="max_tokens"
        )

        with pytest.raises(LLMAdapterError, match="max_tokens"):
            adapter.analyze_segments([], prompt="p")

    def test_wrong_key_makes_the_llm_unavailable(self):
        from src.llm_adapter import LLMUnavailableError

        adapter = self._adapter()
        adapter._client.messages.create.side_effect = FakeAPIError(401, "invalid x-api-key")

        with pytest.raises(LLMUnavailableError, match="invalid x-api-key"):
            adapter.analyze_segments([], prompt="p")

    def test_real_sdk_accepts_the_request(self, monkeypatch):
        """The installed anthropic SDK takes exactly these arguments (no network)."""
        import inspect

        import anthropic

        params = inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters
        for name in ("model", "max_tokens", "system", "messages"):
            assert name in params
