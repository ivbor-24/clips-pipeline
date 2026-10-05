import os

import pytest

# Tests do not depend on a developer's .env: no password unless a test asks for it
# (tests/api_auth_helpers.py: password_mode).
os.environ["API_PASSWORD"] = ""
# The API's test client sends Host "test"; without a password the API answers
# only to known host names (src/api/security.py).
os.environ["API_ALLOWED_HOSTS"] = "test"


@pytest.fixture(autouse=True)
def _no_image_backend(monkeypatch):
    """Inside the Docker image PIPELINE_HARDWARE is set; tests decide it themselves."""
    monkeypatch.delenv("PIPELINE_HARDWARE", raising=False)


@pytest.fixture(autouse=True)
def _jobs_dir_in_tmp(tmp_path, monkeypatch):
    """Create job work directories in a temporary directory, not in the checkout.

    ``create_job`` makes ``jobs/job_<id>`` relative to the current directory, so
    every API test that queues a job left one in the checkout's ``jobs/`` — the
    directory that holds real jobs (bind-mounted user data in Docker).
    """
    monkeypatch.setattr("src.api.services.job.JOBS_DIR", tmp_path / "jobs")


@pytest.fixture(autouse=True)
def _user_settings_in_tmp(tmp_path, monkeypatch):
    """Point the user settings file at the test's tmp_path.

    The default is data/settings.yaml in the checkout — the file the real
    installation's Settings page writes. Tests would read its content (and,
    worse, PUT tests would overwrite it with test values).
    """
    monkeypatch.setenv("PIPELINE_USER_SETTINGS", str(tmp_path / "settings.yaml"))


@pytest.fixture(autouse=True)
def _block_real_llm(monkeypatch):
    """Never load or download a real GGUF model in tests.

    Pipeline tests reuse config/config.yaml, where the local LLM is enabled.
    On a machine with the model on disk or in the Hugging Face cache, scoring
    would load ~9 GB and run real inference; here it fails fast and scoring
    falls back to heuristics, as it does in CI. Tests that exercise the
    adapter patch these helpers explicitly, which takes precedence.
    """
    from src.llm_adapter import LLMAdapterError

    def _blocked(*args, **kwargs):
        raise LLMAdapterError("real LLM loading is disabled in tests")

    monkeypatch.setattr("src.llm_adapter._create_llama", _blocked)
    monkeypatch.setattr("src.llm_adapter._hf_hub_download", _blocked)


@pytest.fixture(autouse=True)
def _no_gpu_video_encoder(monkeypatch):
    """Render with libx264 in tests, whatever GPU the machine has.

    With ``video_encoder: auto`` rendering test-encodes on /dev/dri and NVENC;
    under a mocked ``subprocess.run`` that test always "passes", so a test's
    ffmpeg command would depend on the machine. Tests of the encoder choice
    patch these helpers themselves.
    """
    monkeypatch.setattr("src.rendering._render_nodes", lambda: [])
    monkeypatch.setattr("src.rendering._gpu_backend", lambda config: "cpu")
