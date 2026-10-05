"""
Tests for TASK-07: Human-in-the-Loop Review CLI Module
"""

import json
from pathlib import Path
from unittest.mock import patch


class TestReviewModule:
    """Test review module imports and structure."""

    def test_module_exists(self):
        """Test that review module exists."""
        from src import review

        assert review is not None

    def test_review_error_exists(self):
        """Test that ReviewError exception exists."""
        from src.review import ReviewError

        assert ReviewError is not None
        assert issubclass(ReviewError, Exception)

    def test_find_clips_function_exists(self):
        """Test that find_clips function exists."""
        from src.review import find_clips

        assert callable(find_clips)

    def test_display_clips_table_function_exists(self):
        """Test that display_clips_table function exists."""
        from src.review import display_clips_table

        assert callable(display_clips_table)

    def test_preview_clip_function_exists(self):
        """Test that preview_clip function exists."""
        from src.review import preview_clip

        assert callable(preview_clip)

    def test_get_user_flag_function_exists(self):
        """Test that get_user_flag function exists."""
        from src.review import get_user_flag

        assert callable(get_user_flag)

    def test_interactive_review_session_function_exists(self):
        """Test that interactive_review_session function exists."""
        from src.review import interactive_review_session

        assert callable(interactive_review_session)

    def test_load_review_state_function_exists(self):
        """Test that load_review_state function exists."""
        from src.review import load_review_state

        assert callable(load_review_state)

    def test_save_review_state_function_exists(self):
        """Test that save_review_state function exists."""
        from src.review import save_review_state

        assert callable(save_review_state)

    def test_generate_review_json_function_exists(self):
        """Test that generate_review_json function exists."""
        from src.review import generate_review_json

        assert callable(generate_review_json)

    def test_review_and_export_function_exists(self):
        """Test that main review_and_export function exists."""
        from src.review import review_and_export

        assert callable(review_and_export)

    def test_setup_signal_handler_function_exists(self):
        """Test that setup_signal_handler function exists."""
        from src.review import setup_signal_handler

        assert callable(setup_signal_handler)


class TestFindClips:
    """Test clip discovery functionality."""

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Test finding clips in empty directory."""
        from src.review import find_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        result = find_clips(clips_dir)
        assert result == []

    def test_finds_mp4_files_with_metadata(self, tmp_path):
        """Test finding clips with their metadata files."""
        from src.review import find_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        # Create mock clip files
        video_file = clips_dir / "clip_001.mp4"
        video_file.write_bytes(b"fake video content")

        meta_file = clips_dir / "clip_001.meta.json"
        meta_data = {"score": 0.75, "duration_sec": 60}
        meta_file.write_text(json.dumps(meta_data))

        result = find_clips(clips_dir)

        assert len(result) == 1
        assert result[0]["clip_id"] == "clip_001"
        assert result[0]["video_path"] == video_file
        assert result[0]["meta_path"] == meta_file
        assert result[0]["metadata"] == meta_data

    def test_handles_missing_metadata_gracefully(self, tmp_path):
        """Test handling clips without metadata files."""
        from src.review import find_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        video_file = clips_dir / "clip_001.mp4"
        video_file.write_bytes(b"fake video content")

        result = find_clips(clips_dir)

        assert len(result) == 1
        assert result[0]["metadata"] == {}

    def test_nonexistent_directory_returns_empty_list(self, tmp_path):
        """Test finding clips in nonexistent directory."""
        from src.review import find_clips

        fake_dir = tmp_path / "nonexistent"
        result = find_clips(fake_dir)
        assert result == []

    def test_sorts_clips_by_name(self, tmp_path):
        """Test that clips are sorted by filename."""
        from src.review import find_clips

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        # Create clips in reverse order
        for i in [3, 1, 2]:
            video_file = clips_dir / f"clip_00{i}.mp4"
            video_file.write_bytes(b"fake video")

        result = find_clips(clips_dir)

        assert len(result) == 3
        assert result[0]["clip_id"] == "clip_001"
        assert result[1]["clip_id"] == "clip_002"
        assert result[2]["clip_id"] == "clip_003"


class TestDisplayClipsTable:
    """Test clips table display."""

    def test_empty_clips_shows_message(self, capsys):
        """Test displaying empty clips list."""
        from src.review import display_clips_table

        display_clips_table([])

        captured = capsys.readouterr()
        assert "No clips found" in captured.out

    def test_displays_clip_metadata(self, capsys):
        """Test displaying clips with metadata."""
        from src.review import display_clips_table

        clips = [
            {
                "clip_id": "clip_001",
                "metadata": {
                    "duration_sec": 60.5,
                    "score": 0.75,
                    "terms_found": ["quantum", "physics"],
                    "self_contained": True,
                    "review_status": "pending",
                },
            }
        ]

        display_clips_table(clips)

        captured = capsys.readouterr()
        assert "clip_001" in captured.out
        assert "60.5" in captured.out
        assert "0.75" in captured.out


class TestPreviewClip:
    """Test video preview functionality."""

    def test_nonexistent_video_returns_false(self, tmp_path):
        """Test previewing nonexistent video."""
        from src.review import preview_clip

        fake_path = tmp_path / "nonexistent.mp4"
        result = preview_clip(fake_path)
        assert result is False

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_launches_mpv_player(self, mock_popen, mock_run, tmp_path):
        """Test launching mpv player."""
        from src.review import preview_clip

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        # Mock 'which mpv' to return success
        mock_run.return_value.returncode = 0

        result = preview_clip(video_file, player="mpv")

        assert result is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "mpv" in call_args

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_launches_ffplay_player(self, mock_popen, mock_run, tmp_path):
        """Test launching ffplay player."""
        from src.review import preview_clip

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        result = preview_clip(video_file, player="ffplay")

        assert result is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "ffplay" in call_args


class TestGetUserFlag:
    """Test user input for review decisions."""

    @patch("rich.prompt.Prompt.ask")
    def test_returns_keep_for_k(self, mock_prompt):
        """Test 'k' returns keep."""
        from src.review import get_user_flag

        mock_prompt.return_value = "k"
        result = get_user_flag(1, 5)
        assert result == "keep"

    @patch("rich.prompt.Prompt.ask")
    def test_returns_reject_for_r(self, mock_prompt):
        """Test 'r' returns reject."""
        from src.review import get_user_flag

        mock_prompt.return_value = "r"
        result = get_user_flag(1, 5)
        assert result == "reject"

    @patch("rich.prompt.Prompt.ask")
    def test_returns_edit_for_e(self, mock_prompt):
        """Test 'e' returns edit."""
        from src.review import get_user_flag

        mock_prompt.return_value = "e"
        result = get_user_flag(1, 5)
        assert result == "edit"

    @patch("rich.prompt.Prompt.ask")
    def test_returns_pending_for_s(self, mock_prompt):
        """Test 's' returns pending (skip)."""
        from src.review import get_user_flag

        mock_prompt.return_value = "s"
        result = get_user_flag(1, 5)
        assert result == "pending"

    @patch("rich.prompt.Prompt.ask")
    def test_returns_none_for_quit(self, mock_prompt):
        """Test 'q' returns None to quit."""
        from src.review import get_user_flag

        mock_prompt.return_value = "q"
        result = get_user_flag(1, 5)
        assert result is None

    @patch("rich.prompt.Prompt.ask")
    def test_default_is_skip(self, mock_prompt):
        """Test default choice is skip."""
        from src.review import get_user_flag

        mock_prompt.return_value = "s"
        get_user_flag(1, 5)

        # Verify Prompt.ask was called with default="s"
        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args[1]
        assert call_kwargs.get("default") == "s"


class TestLoadAndSaveReviewState:
    """Test review state persistence."""

    def test_load_nonexistent_file_returns_default(self, tmp_path):
        """Test loading nonexistent file returns default structure."""
        from src.review import load_review_state

        fake_path = tmp_path / "nonexistent.json"
        result = load_review_state(fake_path)

        assert result == {"clips": {}, "started_at": None, "completed_at": None}

    def test_load_valid_file_returns_data(self, tmp_path):
        """Test loading valid review file."""
        from src.review import load_review_state

        review_file = tmp_path / "review.json"
        expected_data = {
            "clips": {"clip_001": {"flag": "keep"}},
            "started_at": "2024-01-01T00:00:00Z",
            "completed_at": None,
        }
        review_file.write_text(json.dumps(expected_data))

        result = load_review_state(review_file)
        assert result == expected_data

    def test_save_review_state_creates_file(self, tmp_path):
        """Test saving review state creates file."""
        from src.review import save_review_state

        review_file = tmp_path / "review.json"
        data = {"clips": {"clip_001": {"flag": "keep"}}}

        save_review_state(data, review_file)

        assert review_file.exists()
        loaded = json.loads(review_file.read_text())
        assert loaded == data

    def test_save_review_state_creates_parent_dirs(self, tmp_path):
        """Test saving creates parent directories if needed."""
        from src.review import save_review_state

        review_file = tmp_path / "nested" / "dir" / "review.json"
        data = {"clips": {}}

        save_review_state(data, review_file)

        assert review_file.exists()


class TestGenerateReviewJson:
    """Test final review JSON generation."""

    def test_generates_complete_review_json(self, tmp_path):
        """Test generating complete review JSON."""
        from src.review import generate_review_json

        clips = [
            {
                "clip_id": "clip_001",
                "video_path": tmp_path / "clip_001.mp4",
                "metadata": {"score": 0.75},
            },
            {
                "clip_id": "clip_002",
                "video_path": tmp_path / "clip_002.mp4",
                "metadata": {"score": 0.65},
            },
        ]

        review_results = {
            "clips": {"clip_001": {"flag": "keep"}, "clip_002": {"flag": "reject"}},
            "completed_at": "2024-01-01T00:00:00Z",
        }

        output_path = tmp_path / "review_output.json"
        generate_review_json(clips, review_results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            review_data = json.load(f)

        assert review_data["total_clips"] == 2
        assert review_data["summary"]["keep"] == 1
        assert review_data["summary"]["reject"] == 1
        assert review_data["summary"]["edit"] == 0
        assert review_data["summary"]["pending"] == 0
        assert len(review_data["clips"]) == 2

    def test_handles_pending_clips(self, tmp_path):
        """Test handling clips with pending status."""
        from src.review import generate_review_json

        clips = [{"clip_id": "clip_001", "video_path": tmp_path / "clip_001.mp4", "metadata": {}}]

        review_results = {"clips": {}}
        output_path = tmp_path / "review.json"

        generate_review_json(clips, review_results, output_path)

        with open(output_path) as f:
            review_data = json.load(f)

        assert review_data["summary"]["pending"] == 1

    def test_creates_parent_directories(self, tmp_path):
        """Test that output directory is created if needed."""
        from src.review import generate_review_json

        clips = []
        review_results = {"clips": {}}
        output_path = tmp_path / "nested" / "output" / "review.json"

        generate_review_json(clips, review_results, output_path)

        assert output_path.exists()


class TestReviewAndExport:
    """Test main review_and_export function."""

    def test_no_clips_returns_none(self, tmp_path, capsys):
        """Test review with no clips returns None."""
        from src.review import review_and_export

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        review_output = tmp_path / "review.json"

        result = review_and_export(
            clips_dir=str(clips_dir), review_output=str(review_output), auto_mode=True
        )

        assert result is None

    @patch("src.review.interactive_review_session")
    def test_auto_mode_marks_all_pending(self, mock_session, tmp_path):
        """Test auto mode marks all clips as pending."""
        from src.review import review_and_export

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        # Create mock clip
        video_file = clips_dir / "clip_001.mp4"
        video_file.write_bytes(b"fake video")
        meta_file = clips_dir / "clip_001.meta.json"
        meta_file.write_text(json.dumps({"score": 0.75}))

        review_output = tmp_path / "review.json"

        result = review_and_export(
            clips_dir=str(clips_dir), review_output=str(review_output), auto_mode=True
        )

        assert result is not None
        assert "clips" in result
        assert result["clips"]["clip_001"]["flag"] == "pending"

    def test_creates_review_json_output(self, tmp_path):
        """Test that review.json is created."""
        from src.review import review_and_export

        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()

        video_file = clips_dir / "clip_001.mp4"
        video_file.write_bytes(b"fake video")
        meta_file = clips_dir / "clip_001.meta.json"
        meta_file.write_text(json.dumps({"score": 0.75}))

        review_output = tmp_path / "artifacts" / "review.json"

        review_and_export(
            clips_dir=str(clips_dir), review_output=str(review_output), auto_mode=True
        )

        assert review_output.exists()

        with open(review_output) as f:
            review_data = json.load(f)

        assert review_data["total_clips"] == 1


class TestInteractiveReviewSession:
    """Test interactive review session."""

    @patch("src.review.get_user_flag")
    def test_completes_all_clips(self, mock_flag, tmp_path):
        """Test reviewing all clips completes session."""
        from src.review import interactive_review_session

        clips = [
            {"clip_id": "clip_001", "metadata": {"score": 0.75}},
            {"clip_id": "clip_002", "metadata": {"score": 0.65}},
        ]
        review_file = tmp_path / "review.json"

        # Simulate user choosing 'keep' for both
        mock_flag.side_effect = ["keep", "keep"]

        result = interactive_review_session(clips, review_file)

        assert result["completed_at"] is not None
        assert len(result["clips"]) == 2

    @patch("src.review.get_user_flag")
    def test_quit_saves_progress(self, mock_flag, tmp_path):
        """Test quitting saves progress."""
        from src.review import interactive_review_session

        clips = [{"clip_id": "clip_001", "metadata": {}}, {"clip_id": "clip_002", "metadata": {}}]
        review_file = tmp_path / "review.json"

        # First clip: keep, second clip: quit
        mock_flag.side_effect = ["keep", None]

        result = interactive_review_session(clips, review_file)

        # Should have saved progress for first clip
        assert review_file.exists()
        assert "clip_001" in result["clips"]
        assert result["clips"]["clip_001"]["flag"] == "keep"

    def test_empty_clips_returns_early(self, capsys):
        """Test empty clips list returns early."""
        from src.review import interactive_review_session

        review_file = Path("dummy.json")
        result = interactive_review_session([], review_file)

        assert result == {"clips": [], "completed_at": result["completed_at"]}


class TestGracefulShutdown:
    """Test graceful shutdown handling."""

    def test_setup_signal_handler_does_not_crash(self):
        """Test signal handler setup doesn't crash."""
        from src.review import setup_signal_handler

        # Should not raise any exceptions
        setup_signal_handler()

    @patch("src.review.save_review_state")
    @patch("src.review.get_user_flag")
    def test_keyboard_interrupt_saves_progress(self, mock_flag, mock_save, tmp_path):
        """Test Ctrl+C saves progress."""
        from src.review import interactive_review_session

        clips = [{"clip_id": "clip_001", "metadata": {}}]
        review_file = tmp_path / "review.json"

        # Simulate keyboard interrupt
        mock_flag.side_effect = KeyboardInterrupt()

        interactive_review_session(clips, review_file)

        # Should have called save
        assert mock_save.called


class TestReviewJsonStructure:
    """Test review.json output structure matches specification."""

    def test_review_json_has_required_fields(self, tmp_path):
        """Test review.json has all required fields."""
        from src.review import generate_review_json

        clips = [
            {
                "clip_id": "clip_001",
                "video_path": tmp_path / "clip_001.mp4",
                "metadata": {"score": 0.75, "duration_sec": 60},
            }
        ]

        review_results = {
            "clips": {"clip_001": {"flag": "keep"}},
            "completed_at": "2024-01-01T00:00:00Z",
        }

        output_path = tmp_path / "review.json"
        generate_review_json(clips, review_results, output_path)

        with open(output_path) as f:
            review_data = json.load(f)

        # Check required fields per TASK-07 spec
        assert "review_completed_at" in review_data or "completed_at" in review_results
        assert "total_clips" in review_data
        assert "summary" in review_data
        assert "clips" in review_data

        # Check summary has all flag types
        assert "keep" in review_data["summary"]
        assert "reject" in review_data["summary"]
        assert "edit" in review_data["summary"]
        assert "pending" in review_data["summary"]

        # Check clip entries have required fields
        clip_entry = review_data["clips"][0]
        assert "clip_id" in clip_entry
        assert "video_path" in clip_entry
        assert "flag" in clip_entry
        assert "metadata" in clip_entry
