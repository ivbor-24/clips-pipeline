"""
TASK-07: Human-in-the-Loop Review CLI Module

Provides a minimal TUI for validating clips before manual export.

Inputs:
- output/clips/*.mp4
- output/clips/*.meta.json

Outputs:
- artifacts/review.json with flags keep/reject/edit

Features:
- rich table with clip metadata (score, duration, terms)
- Quick preview via mpv/ffplay (subprocess spawn)
- Interactive flag selection (1-9 keys)
- Graceful save on terminal close
"""

import json
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

logger = structlog.get_logger("review")
console = Console()


class ReviewError(Exception):
    """Custom exception for review errors."""

    pass


ReviewFlag = Literal["keep", "reject", "edit"]


def find_clips(clips_dir: Path) -> List[Dict[str, Any]]:
    """
    Find all rendered clips and their metadata in the output directory.

    Args:
        clips_dir: Path to clips directory

    Returns:
        List of clip info dictionaries with video_path, meta_path, and metadata
    """
    if not clips_dir.exists():
        logger.warning("Clips directory does not exist", path=str(clips_dir))
        return []

    clips = []
    mp4_files = sorted(clips_dir.glob("clip_*.mp4"))

    for mp4_path in mp4_files:
        clip_id = mp4_path.stem  # e.g., "clip_001"
        meta_path = clips_dir / f"{clip_id}.meta.json"

        clip_info = {
            "clip_id": clip_id,
            "video_path": mp4_path,
            "meta_path": meta_path,
            "metadata": None,
        }

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    clip_info["metadata"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load metadata", path=str(meta_path), error=str(e))
                clip_info["metadata"] = {}
        else:
            logger.warning("Metadata file not found", path=str(meta_path))
            clip_info["metadata"] = {}

        clips.append(clip_info)

    logger.info(f"Found {len(clips)} clips", count=len(clips))
    return clips


def display_clips_table(clips: List[Dict[str, Any]]) -> None:
    """
    Display clips in a rich table with metadata.

    Args:
        clips: List of clip info dictionaries
    """
    if not clips:
        console.print("[yellow]No clips found to review[/yellow]")
        return

    table = Table(
        title="📐 Scientific Clips - Review Queue", show_header=True, header_style="bold cyan"
    )

    table.add_column("#", style="dim", width=4)
    table.add_column("Clip ID", style="cyan", width=12)
    table.add_column("Duration", width=10)
    table.add_column("Score", width=8)
    table.add_column("Terms", width=20, overflow="ellipsis")
    table.add_column("Self-Contained", width=14)
    table.add_column("Status", width=10)

    for i, clip in enumerate(clips, start=1):
        meta = clip.get("metadata", {})

        clip_id = clip.get("clip_id", "unknown")
        duration = meta.get("duration_sec", 0)
        score = meta.get("score", 0)
        terms = meta.get("terms_found", [])
        self_contained = meta.get("self_contained", False)
        status = meta.get("review_status", "pending")

        # Format duration
        duration_str = f"{duration:.1f}s"

        # Format score with color
        if score >= 0.75:
            score_str = f"[green]{score:.2f}[/green]"
        elif score >= 0.55:
            score_str = f"[yellow]{score:.2f}[/yellow]"
        else:
            score_str = f"[red]{score:.2f}[/red]"

        # Format terms
        terms_str = ", ".join(terms[:3]) if terms else "[dim]none[/dim]"

        # Format self-contained
        self_contained_str = "[green]✓[/green]" if self_contained else "[dim]—[/dim]"

        # Format status
        if status == "keep":
            status_str = "[green]KEEP[/green]"
        elif status == "reject":
            status_str = "[red]REJECT[/red]"
        elif status == "edit":
            status_str = "[yellow]EDIT[/yellow]"
        else:
            status_str = "[dim]pending[/dim]"

        table.add_row(
            str(i), clip_id, duration_str, score_str, terms_str, self_contained_str, status_str
        )

    console.print(table)


def preview_clip(video_path: Path, player: str = "auto") -> bool:
    """
    Launch video player to preview a clip.

    Args:
        video_path: Path to video file
        player: Player to use ('mpv', 'ffplay', or 'auto')

    Returns:
        True if player launched successfully, False otherwise
    """
    if not video_path.exists():
        logger.error("Video file not found", path=str(video_path))
        console.print(f"[red]Error:[/red] Video file not found: {video_path}")
        return False

    # Auto-detect player
    if player == "auto":
        # Try mpv first (better experience), then ffplay
        if subprocess.run(["which", "mpv"], capture_output=True).returncode == 0:
            player = "mpv"
        elif subprocess.run(["which", "ffplay"], capture_output=True).returncode == 0:
            player = "ffplay"
        else:
            console.print("[yellow]⚠ No video player found (mpv or ffplay)[/yellow]")
            return False

    try:
        if player == "mpv":
            # mpv: non-blocking, --force-window for quick preview
            cmd = ["mpv", "--force-window", "--no-terminal", str(video_path)]
        elif player == "ffplay":
            # ffplay: blocking, but available with ffmpeg
            cmd = ["ffplay", "-nodisp", "-autoexit", str(video_path)]
        else:
            logger.error("Unknown player", player=player)
            return False

        logger.info("Launching preview", player=player, video=str(video_path))
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    except FileNotFoundError:
        logger.error("Player not found", player=player)
        console.print(f"[red]Error:[/red] Player '{player}' not found")
        return False
    except Exception as e:
        logger.error("Failed to launch player", error=str(e))
        console.print(f"[red]Error:[/red] Failed to launch player: {e}")
        return False


def get_user_flag(clip_index: int, total: int) -> Optional[ReviewFlag]:
    """
    Get user's review decision for a clip.

    Args:
        clip_index: Current clip index (1-based)
        total: Total number of clips

    Returns:
        Review flag or None if skipping
    """
    console.print(
        Panel.fit(
            f"[bold]Clip {clip_index}/{total}[/bold]\n\n"
            f"[cyan]Commands:[/cyan]\n"
            f"  [bold]k[/bold] - [green]Keep[/green] (approve for export)\n"
            f"  [bold]r[/bold] - [red]Reject[/red] (mark for deletion)\n"
            f"  [bold]e[/bold] - [yellow]Edit[/yellow] (needs modifications)\n"
            f"  [bold]p[/bold] - Preview clip in player\n"
            f"  [bold]s[/bold] - Skip (leave as pending)\n"
            f"  [bold]q[/bold] - Quit and save progress\n",
            title="🎬 Review Controls",
        )
    )

    while True:
        try:
            choice = Prompt.ask(
                "Your decision", choices=["k", "r", "e", "p", "s", "q"], default="s"
            )

            if choice == "q":
                return None  # Signal to quit
            elif choice == "p":
                # Preview is handled separately
                continue
            elif choice in ("k", "r", "e", "s"):
                mapping = {"k": "keep", "r": "reject", "e": "edit", "s": "pending"}
                return mapping[choice]

        except KeyboardInterrupt:
            return None  # Handle Ctrl+C gracefully


def interactive_review_session(clips: List[Dict[str, Any]], review_file: Path) -> Dict[str, Any]:
    """
    Run an interactive review session for all clips.

    Args:
        clips: List of clip info dictionaries
        review_file: Path to save review results

    Returns:
        Review results dictionary
    """
    if not clips:
        console.print("[yellow]No clips to review[/yellow]")
        return {"clips": [], "completed_at": datetime.now(timezone.utc).isoformat()}

    console.print("\n[bold cyan]Starting Interactive Review Session[/bold cyan]\n")
    console.print("[dim]Tip: Press Ctrl+C at any time to save progress and exit[/dim]\n")

    # Initialize or load existing review state
    review_results = (
        load_review_state(review_file)
        if review_file.exists()
        else {
            "clips": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
    )

    reviewed_count = 0
    total_clips = len(clips)

    try:
        for i, clip in enumerate(clips, start=1):
            clip_id = clip["clip_id"]

            # Display current clip info
            meta = clip.get("metadata", {})
            console.print(
                Panel(
                    f"[bold]{clip_id}[/bold]\n"
                    f"Duration: [cyan]{meta.get('duration_sec', 0):.1f}s[/cyan] | "
                    f"Score: [green]{meta.get('score', 0):.2f}[/green]\n"
                    f"Terms: {', '.join(meta.get('terms_found', ['none']))}\n"
                    f"Self-contained: {'✓' if meta.get('self_contained') else '—'}",
                    title=f"📋 Clip {i}/{total_clips}",
                    border_style="cyan",
                )
            )

            # Get user decision
            decision = get_user_flag(i, total_clips)

            if decision is None:
                # User chose to quit
                console.print("\n[yellow]Saving progress and exiting...[/yellow]")
                break

            if decision == "p":
                # Preview requested - handle specially
                preview_clip(clip["video_path"])
                # Re-prompt after preview
                decision = get_user_flag(i, total_clips)
                if decision is None:
                    console.print("\n[yellow]Saving progress and exiting...[/yellow]")
                    break

            # Record decision
            review_results["clips"][clip_id] = {
                "flag": decision,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "metadata_snapshot": meta,
            }

            # Update clip metadata with status
            if clip.get("metadata"):
                clip["metadata"]["review_status"] = decision

            reviewed_count += 1

            # Display confirmation
            status_colors = {"keep": "green", "reject": "red", "edit": "yellow", "pending": "dim"}
            color = status_colors.get(decision, "white")
            console.print(f"[{color}]✓ Marked as {decision.upper()}[/{color}]\n")

            # Save progress after each clip
            save_review_state(review_results, review_file)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted! Saving progress...[/yellow]")

    # Mark completion
    if reviewed_count == total_clips:
        review_results["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_review_state(review_results, review_file)
        console.print("\n[green]✓ Review session completed![/green]")
    else:
        console.print(
            f"\n[yellow]⚠ Review incomplete: {reviewed_count}/{total_clips} clips reviewed[/yellow]"
        )
        save_review_state(review_results, review_file)

    return review_results


def load_review_state(review_file: Path) -> Dict[str, Any]:
    """Load existing review state from file."""
    try:
        with open(review_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load review state", error=str(e))
        return {"clips": {}, "started_at": None, "completed_at": None}


def save_review_state(review_results: Dict[str, Any], review_file: Path) -> None:
    """Save review state to file."""
    review_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump(review_results, f, indent=2)
        logger.info("Review state saved", path=str(review_file))
    except IOError as e:
        logger.error("Failed to save review state", error=str(e))
        console.print(f"[red]Error saving review:[/red] {e}")


def generate_review_json(
    clips: List[Dict[str, Any]], review_results: Dict[str, Any], output_path: Path
) -> None:
    """
    Generate final review.json with all decisions.

    Args:
        clips: List of clip info dictionaries
        review_results: Review decisions
        output_path: Path for output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build structured output
    review_data = {
        "review_completed_at": review_results.get("completed_at"),
        "total_clips": len(clips),
        "summary": {"keep": 0, "reject": 0, "edit": 0, "pending": 0},
        "clips": [],
    }

    for clip in clips:
        clip_id = clip["clip_id"]
        decision = review_results.get("clips", {}).get(clip_id, {}).get("flag", "pending")

        review_data["summary"][decision] = review_data["summary"].get(decision, 0) + 1

        clip_entry = {
            "clip_id": clip_id,
            "video_path": str(clip["video_path"]),
            "flag": decision,
            "metadata": clip.get("metadata", {}),
        }
        review_data["clips"].append(clip_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    logger.info("Review JSON generated", path=str(output_path), summary=review_data["summary"])

    # Display summary
    console.print(
        Panel(
            f"[bold]Review Summary[/bold]\n\n"
            f"Total Clips: [cyan]{review_data['total_clips']}[/cyan]\n"
            f"[green]Keep:[/green] {review_data['summary']['keep']}\n"
            f"[red]Reject:[/red] {review_data['summary']['reject']}\n"
            f"[yellow]Edit:[/yellow] {review_data['summary']['edit']}\n"
            f"[dim]Pending:[/dim] {review_data['summary']['pending']}",
            title="📊 Review Results",
        )
    )


def review_and_export(
    clips_dir: str = None,
    review_output: str = None,
    auto_mode: bool = False,
    player: str = "auto",
    work_dir: Path = None,
) -> Optional[Dict[str, Any]]:
    """
    Main review function - can run in interactive or auto mode.

    Args:
        clips_dir: Path to clips directory
        review_output: Path for review.json output
        auto_mode: If True, skip interactive review (for testing)
        player: Video player to use for previews
        work_dir: Base working directory for default paths

    Returns:
        Review results dictionary or None on failure

    Raises:
        ReviewError: On review failure
    """
    if work_dir is None:
        work_dir = Path(".")
    if clips_dir is None:
        clips_dir = str(work_dir / "output" / "clips")
    if review_output is None:
        review_output = str(work_dir / "artifacts" / "review.json")

    clips_path = Path(clips_dir)
    review_path = Path(review_output)

    logger.info("Starting review process", clips_dir=str(clips_path), auto_mode=auto_mode)

    # Find clips
    clips = find_clips(clips_path)

    if not clips:
        console.print("[yellow]No clips found to review. Run the pipeline first.[/yellow]")
        return None

    # Display initial table
    display_clips_table(clips)

    if auto_mode:
        # Auto mode: mark all as pending for now
        console.print("[dim]Auto mode: marking all clips as pending[/dim]")
        review_results = {
            "clips": {clip["clip_id"]: {"flag": "pending"} for clip in clips},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        # Interactive mode
        review_results = interactive_review_session(clips, review_path)

    # Generate final review.json
    generate_review_json(clips, review_results, review_path)

    console.print(f"\n[green]✓ Review results saved to:[/green] {review_path}")

    return review_results


# Setup signal handler for graceful shutdown
def setup_signal_handler():
    """Setup signal handler for graceful shutdown on Ctrl+C."""

    def handler(signum, frame):
        console.print("\n[yellow]Received interrupt signal. Saving progress...[/yellow]")
        # Note: Actual saving happens in the interactive_review_session
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
