"""Tests for scripts/llm_provider.py, setup.sh's helper for choosing the LLM."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "llm_provider.py"


@pytest.fixture
def llm_provider():
    spec = importlib.util.spec_from_file_location("llm_provider_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings(tmp_path):
    """The user settings file (conftest points PIPELINE_USER_SETTINGS here)."""
    return tmp_path / "settings.yaml"


def llm_config(**kwargs):
    from src.config import LLMConfig

    return LLMConfig(**{"enabled": True, **kwargs})


def run_main(module, *args):
    with patch.object(sys, "argv", ["llm_provider.py", *args]):
        return module.main()


class TestStatus:
    def test_local_model_ready_when_the_file_is_there(self, llm_provider, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_text("")

        status = llm_provider.status(llm_config(provider="llama_cpp", model_path=str(model)))

        assert status["LLM_READY"] == "yes"
        assert status["LLM_PROVIDER"] == "llama_cpp"

    def test_local_model_not_ready_without_the_file(self, llm_provider, tmp_path):
        cfg = llm_config(provider="llama_cpp", model_path=str(tmp_path / "missing.gguf"))

        assert llm_provider.status(cfg)["LLM_READY"] == "no"

    def test_api_provider_ready_only_with_its_key(self, llm_provider, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

        assert llm_provider.status(llm_config(provider="openai"))["LLM_READY"] == "yes"
        assert llm_provider.status(llm_config(provider="anthropic"))["LLM_READY"] == "no"

    def test_api_key_in_the_config_counts(self, llm_provider, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        status = llm_provider.status(llm_config(provider="anthropic", api_key="sk-ant"))

        assert status["LLM_READY"] == "yes"

    def test_disabled_llm_is_not_ready(self, llm_provider, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

        status = llm_provider.status(llm_config(enabled=False, provider="openai"))

        assert status["LLM_READY"] == "no"
        assert status["LLM_ENABLED"] == "no"

    def test_main_prints_lines_for_setup_sh(self, llm_provider, settings, capsys):
        settings.write_text(
            yaml.safe_dump({"scoring": {"llm": {"provider": "openai", "model": "gpt-test"}}})
        )

        assert run_main(llm_provider, "status") == 0

        out = capsys.readouterr().out.splitlines()
        assert "LLM_PROVIDER=openai" in out
        assert "LLM_MODEL=gpt-test" in out


class TestChoice:
    def test_use_api_saves_provider_and_model_and_keeps_other_settings(
        self, llm_provider, settings
    ):
        settings.write_text(
            yaml.safe_dump(
                {
                    "scoring": {
                        "max_clips_per_video": 8,
                        "llm": {"enabled": False, "api_key": "old", "api_base": "http://x"},
                    }
                }
            )
        )

        assert run_main(llm_provider, "use-api", "--provider", "anthropic", "--model", "m") == 0

        saved = yaml.safe_load(settings.read_text())
        assert saved == {
            "scoring": {
                "max_clips_per_video": 8,
                "llm": {"provider": "anthropic", "model": "m"},
            }
        }

    def test_use_api_with_an_address(self, llm_provider, settings):
        args = ["--provider", "openai", "--model", "m", "--api-base", "https://api.example.com/v1"]

        assert run_main(llm_provider, "use-api", *args) == 0

        llm = yaml.safe_load(settings.read_text())["scoring"]["llm"]
        assert llm == {"provider": "openai", "model": "m", "api_base": "https://api.example.com/v1"}

    def test_use_local_goes_back_to_the_config_file(self, llm_provider, settings):
        settings.write_text(
            yaml.safe_dump(
                {"scoring": {"max_clips_per_video": 8, "llm": {"provider": "openai", "model": "m"}}}
            )
        )

        assert run_main(llm_provider, "use-local") == 0

        assert yaml.safe_load(settings.read_text()) == {"scoring": {"max_clips_per_video": 8}}

    def test_broken_settings_file_is_not_overwritten(self, llm_provider, settings, capsys):
        settings.write_text("scoring: [not, a, mapping\n")

        assert run_main(llm_provider, "use-local") == 1

        assert settings.read_text() == "scoring: [not, a, mapping\n"
        assert "Cannot change the settings" in capsys.readouterr().err


class TestCheck:
    def test_one_request_through_the_job_adapter(self, llm_provider):
        adapter = MagicMock()
        adapter.analyze_segments.return_value = [{"ok": True}]
        with patch("src.llm_adapter.create_llm_adapter", return_value=adapter) as create:
            summary = llm_provider.check(llm_config(provider="openai", model="gpt-test"))

        create.assert_called_once()
        assert adapter.analyze_segments.call_args.kwargs["prompt"] == llm_provider.CHECK_PROMPT
        assert "openai model gpt-test answered" in summary

    def test_answer_without_the_json_fails(self, llm_provider):
        from src.llm_adapter import LLMAdapterError

        adapter = MagicMock()
        adapter.analyze_segments.return_value = []
        with patch("src.llm_adapter.create_llm_adapter", return_value=adapter):
            with pytest.raises(LLMAdapterError, match="not with the JSON"):
                llm_provider.check(llm_config(provider="anthropic"))

    def test_local_model_is_not_checked(self, llm_provider):
        from src.llm_adapter import LLMAdapterError

        with pytest.raises(LLMAdapterError, match="API providers only"):
            llm_provider.check(llm_config(provider="llama_cpp"))

    def test_main_reports_the_failure(self, llm_provider, settings, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings.write_text(yaml.safe_dump({"scoring": {"llm": {"provider": "openai"}}}))

        assert run_main(llm_provider, "check") == 1

        assert "OPENAI_API_KEY" in capsys.readouterr().err
