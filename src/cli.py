"""
Scientific Clips Pipeline - Main CLI Entry Point

Usage:
    just process input=<path_or_url> [--dry-run] [--resume] [--force] [--config <path>]
"""

import re
import sys
from datetime import datetime
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.config import Config
from src.gpu_utils import get_auto_gpu_config, get_vram_gb, suggest_profile
from src.logger import get_logger, setup_logging
from src.pipeline import PipelineError, run_chapters_pipeline, run_pipeline

app = typer.Typer(help="Scientific Clips Pipeline MVP")
console = Console()


def compute_output_dir(video_path: str, cfg: Config) -> Path:
    """Compute isolated output directory for a given video."""
    if not cfg.output.isolated:
        return Path(cfg.output.clips_dir).parent

    video_stem = Path(video_path).stem
    safe_name = re.sub(r"[^\w\-.]", "_", video_stem)
    timestamp = datetime.now().strftime(cfg.output.timestamp_format)
    dir_name = cfg.output.output_dir_template.format(video_name=safe_name, timestamp=timestamp)
    return Path("output") / dir_name


def apply_isolated_output(cfg: Config, output_dir: Path) -> None:
    """Mutate cfg.output paths to use the isolated output directory."""
    cfg.work_dir = output_dir
    cfg.output.clips_dir = "clips"
    cfg.output.manifest_file = "manifest.json"
    cfg.output.review_file = "review.json"


class RichProgressCallback:
    """Wraps Rich progress bars to implement the pipeline callback interface."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._progress: Optional[Progress] = None
        self._tasks: dict = {}

    def __enter__(self) -> "RichProgressCallback":
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._progress.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._progress:
            self._progress.__exit__(*args)

    def __call__(
        self, stage_name: str, status: str, progress_percent: Optional[float] = None
    ) -> None:
        if self._progress is None:
            return

        if status == "started":
            task_id = self._progress.add_task(f"[yellow]▶ {stage_name}[/yellow]", total=None)
            self._tasks[stage_name] = task_id

        elif status == "completed":
            task_id = self._tasks.get(stage_name)
            if task_id is not None:
                self._progress.update(
                    task_id,
                    description=f"[green]✓ {stage_name}[/green] — complete",
                    total=1,
                    completed=1,
                )

        elif status == "failed":
            task_id = self._tasks.get(stage_name)
            if task_id is not None:
                self._progress.update(
                    task_id,
                    description=f"[red]✗ {stage_name}[/red] — failed",
                    total=1,
                    completed=1,
                )

        elif status == "skipped":
            task_id = self._progress.add_task(
                f"[blue]⏭ {stage_name}[/blue] — skipped (valid artifact exists)",
                total=1,
                completed=1,
            )
            self._tasks[stage_name] = task_id


@app.command()
def process(
    input: str = typer.Option(..., help="Path to video file or YouTube URL"),
    config: str = typer.Option("config/config.yaml", help="Path to configuration file"),
    dry_run: bool = typer.Option(False, help="Run without executing pipeline steps"),
    resume: bool = typer.Option(False, help="Resume from last successful artifact"),
    force: bool = typer.Option(False, help="Force recompute all artifacts (ignores resume)"),
    profile: Optional[str] = typer.Option(
        None, help="Configuration profile to load (e.g., low_vram)"
    ),
) -> None:
    """
    Process a video file or URL through the Scientific Clips Pipeline.
    """
    try:
        cfg = Config.load(config).with_hardware_profile()
        setup_logging(
            log_file=cfg.logging.log_file,
            level=cfg.logging.level,
            json_format=cfg.logging.json_format,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    logger = get_logger("pipeline")

    gpu_config = get_auto_gpu_config(cfg.gpu)
    logger.info(
        "gpu_configuration",
        backend=gpu_config.backend,
        gpu_name=gpu_config.gpu_name,
        whisper_device=gpu_config.whisper_device,
        whisper_compute=gpu_config.whisper_compute_type,
    )

    vram_gb = get_vram_gb()
    suggested = suggest_profile(vram_gb)

    if profile:
        try:
            cfg = cfg.load_profile(profile)
            logger.info("profile_applied", profile=profile)
            console.print(f"[green]✓ Profile loaded:[/green] {profile}")
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)
    elif suggested:
        logger.warning("profile_suggested_auto", vram_gb=vram_gb, suggested_profile=suggested)
        console.print(
            f"[yellow]⚠ VRAM {vram_gb} GB < 10 GB detected. "
            f"Consider using --profile {suggested}[/yellow]"
        )

    if not input:
        logger.error("No input provided")
        console.print("[red]Error:[/red] No input provided")
        raise typer.Exit(code=1)

    logger.info("Pipeline started", input=input, dry_run=dry_run, resume=resume, force=force)

    output_dir = compute_output_dir(input, cfg)
    apply_isolated_output(cfg, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "output_directory_computed", output_dir=str(output_dir), isolated=cfg.output.isolated
    )

    profile_display = profile if profile else "none"
    console.print(
        Panel.fit(
            f"[bold]Scientific Clips Pipeline MVP[/bold]\n\n"
            f"Input: [cyan]{input}[/cyan]\n"
            f"Config: [cyan]{config}[/cyan]\n"
            f"Output: [cyan]{output_dir}[/cyan]\n"
            f"Profile: [cyan]{profile_display}[/cyan]\n"
            f"Dry Run: [yellow]{dry_run}[/yellow]\n"
            f"Resume: [yellow]{resume}[/yellow]\n"
            f"Force: [yellow]{force}[/yellow]",
            title="Pipeline Configuration",
        )
    )

    if dry_run:
        console.print("[green]✓ Dry run passed - configuration is valid[/green]")
        logger.info("Dry run completed successfully")

        try:
            _ = cfg.input
            _ = cfg.preprocessing
            _ = cfg.transcription
            _ = cfg.scoring
            _ = cfg.cropping
            _ = cfg.rendering
            _ = cfg.output
            _ = cfg.logging
            console.print("[green]✓ All configuration sections validated[/green]")
        except Exception as e:
            console.print(f"[red]Configuration validation failed:[/red] {e}")
            logger.error("Configuration validation failed", error=str(e))
            raise typer.Exit(code=1)

        return

    try:
        with RichProgressCallback(console) as callback:
            run_pipeline(
                config=cfg,
                input_source=input,
                dry_run=False,
                resume=resume,
                force=force,
                progress_callback=callback,
            )
    except PipelineError as e:
        logger.error("Pipeline failed", error=str(e))
        console.print(f"[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error("Pipeline failed with unexpected error", error=str(e))
        console.print(f"[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold green]✓ Pipeline complete![/bold green]")


@app.command()
def validate_config(
    config: str = typer.Option("config/config.yaml", help="Path to configuration file"),
) -> None:
    """Validate configuration file without running the pipeline."""
    logger = get_logger("config_validator")

    try:
        cfg = Config.load(config)
        console.print("[green]✓ Configuration loaded successfully[/green]")
        console.print(f"  - Input cache: {cfg.input.cache_dir}")
        console.print(f"  - Transcription model: {cfg.transcription.model}")
        console.print(f"  - Scoring threshold: {cfg.scoring.min_score_threshold}")
        console.print(f"  - Max clips: {cfg.scoring.max_clips_per_video}")
        logger.info("Configuration validated", config=config)
    except Exception as e:
        console.print(f"[red]Configuration validation failed:[/red] {e}")
        logger.error("Configuration validation failed", error=str(e))
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show pipeline version."""
    from src import __version__

    try:
        package_version = get_package_version("scientific-clips-pipeline")
    except Exception:
        package_version = __version__

    console.print("[bold]Scientific Clips Pipeline[/bold]")
    console.print(f"Version: {package_version}")
    console.print("Python: " + sys.version.split()[0])


@app.command()
def chapters(
    input: str = typer.Option(..., help="Path to video file or YouTube URL"),
    config: str = typer.Option("config/config.yaml", help="Path to configuration file"),
    dry_run: bool = typer.Option(False, help="Run without executing pipeline steps"),
    resume: bool = typer.Option(False, help="Resume from last successful artifact"),
    force: bool = typer.Option(False, help="Force recompute all artifacts"),
    profile: Optional[str] = typer.Option(None, help="Configuration profile to load"),
    min_minutes: int = typer.Option(5, help="Minimum chapter duration in minutes"),
    max_minutes: int = typer.Option(7, help="Maximum chapter duration in minutes"),
) -> None:
    """
    Generate video chapters/outline with timestamps.

    Creates a table of contents for the video with LLM-detected chapter boundaries.
    Outputs chapters.json and chapters.txt (YouTube/VK format).
    """
    try:
        cfg = Config.load(config).with_hardware_profile()
        setup_logging(
            log_file=cfg.logging.log_file,
            level=cfg.logging.level,
            json_format=cfg.logging.json_format,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    logger = get_logger("chapters_cli")

    cfg.chapters.min_minutes = min_minutes
    cfg.chapters.max_minutes = max_minutes
    cfg.chapters.enabled = True

    if profile:
        try:
            cfg = cfg.load_profile(profile)
            logger.info("profile_applied", profile=profile)
            console.print(f"[green]✓ Profile loaded:[/green] {profile}")
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

    if not input:
        logger.error("No input provided")
        console.print("[red]Error:[/red] No input provided")
        raise typer.Exit(code=1)

    logger.info("Chapters pipeline started", input=input, dry_run=dry_run)

    output_dir = compute_output_dir(input, cfg)
    apply_isolated_output(cfg, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold]Chapters Generation[/bold]\n\n"
            f"Input: [cyan]{input}[/cyan]\n"
            f"Config: [cyan]{config}[/cyan]\n"
            f"Output: [cyan]{output_dir}[/cyan]\n"
            f"Min minutes: [cyan]{min_minutes}[/cyan]\n"
            f"Max minutes: [cyan]{max_minutes}[/cyan]",
            title="Chapters Configuration",
        )
    )

    if dry_run:
        console.print("[green]✓ Dry run passed - configuration is valid[/green]")
        return

    try:
        with RichProgressCallback(console) as callback:
            run_chapters_pipeline(
                config=cfg,
                input_source=input,
                dry_run=False,
                resume=resume,
                force=force,
                progress_callback=callback,
            )
    except PipelineError as e:
        logger.error("Chapters pipeline failed", error=str(e))
        console.print(f"[red]Chapters pipeline failed:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error("Chapters pipeline failed", error=str(e))
        console.print(f"[red]Chapters pipeline failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold green]✓ Chapters generation complete![/bold green]")

    chapters_txt = output_dir / "artifacts" / "chapters.txt"

    if chapters_txt.exists():
        console.print("\n[bold]YouTube/VK format:[/bold]")
        console.print(f"[dim]{chapters_txt.read_text()}[/dim]")


image_app = typer.Typer(help="Docker image build/save/load/verify/release.")
app.add_typer(image_app, name="image")


@image_app.command("build")
def image_build(
    backend: str = typer.Option("openvino", help="GPU backend: cpu|cuda|openvino|rocm"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable Docker build cache"),
    dockerfile: str = typer.Option("Dockerfile.backend", help="Path to Dockerfile"),
) -> None:
    """Build the pipeline Docker image."""
    from src.image_tool import ImageError, build_image

    try:
        tag = build_image(backend=backend, dockerfile=Path(dockerfile), no_cache=no_cache)
    except ImageError as e:
        console.print(f"[red]build failed:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[bold green]✓ built[/bold green] {tag}")


@image_app.command("save")
def image_save(
    image: str = typer.Option(..., help="Image:tag to export"),
    dist: str = typer.Option("dist", help="Output directory"),
) -> None:
    """Export an image to a gzipped tarball with sidecar sha256."""
    from src.image_tool import ImageError, save_image

    try:
        out = save_image(image, dist_dir=Path(dist))
    except ImageError as e:
        console.print(f"[red]save failed:[/red] {e}")
        raise typer.Exit(code=1)
    sha256 = out.with_suffix(out.suffix + ".sha256").read_text().split()[0]
    console.print(f"[bold green]✓ saved[/bold green] {out}  [dim]sha256: {sha256[:12]}...[/dim]")


@image_app.command("load")
def image_load(
    archive: str = typer.Argument(..., help="Path to .tar or .tar.gz image archive"),
) -> None:
    """Load a previously exported image archive."""
    from src.image_tool import ImageError, load_image

    try:
        tags = load_image(Path(archive))
    except ImageError as e:
        console.print(f"[red]load failed:[/red] {e}")
        raise typer.Exit(code=1)
    for t in tags:
        console.print(f"[green]✓ loaded[/green] {t}")


@image_app.command("verify")
def image_verify(
    image: str = typer.Option(..., help="Image:tag to verify"),
    model: str = typer.Option("tiny", help="Whisper model to test load"),
) -> None:
    """Run GPU/onnxruntime/whisper smoke checks inside the image."""
    from src.image_tool import ImageError, verify_image

    try:
        ok, summary = verify_image(image, model=model)
    except ImageError as e:
        console.print(f"[red]verify failed:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(summary)
    if not ok:
        raise typer.Exit(code=1)
    console.print(f"[bold green]✓ verified[/bold green] {image}")


@image_app.command("release")
def image_release(
    image: str = typer.Option(..., help="Verified image:tag to promote"),
    stable_tag: str = typer.Option(None, help="Override stable tag"),
) -> None:
    """Tag a verified image as stable and update the deployment manifest."""
    from src.image_tool import ImageError, release_image

    try:
        tag = release_image(image, stable_tag=stable_tag)
    except ImageError as e:
        console.print(f"[red]release failed:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[bold green]✓ released[/bold green] {tag}")


@image_app.command("list")
def image_list(
    backend: str = typer.Option(None, help="Filter by backend (cpu|cuda|openvino|rocm)"),
) -> None:
    """List locally available pipeline images."""
    from src.image_tool import list_images

    for row in list_images(backend=backend):
        console.print(f"  {row['repository']}:{row['tag']}  {row['size']}  {row['created']}")


@image_app.command("prune")
def image_prune(
    backend: str = typer.Option(..., help="Backend to prune (cpu|cuda|openvino|rocm)"),
) -> None:
    """Remove local pipeline images for a backend except stable/latest."""
    from src.image_tool import ImageError, prune_local

    try:
        n = prune_local(backend)
    except ImageError as e:
        console.print(f"[red]prune failed:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[green]✓ pruned[/green] {n} image(s)")


@app.command()
def review(
    clips_dir: str = typer.Option("output/clips", help="Path to clips directory"),
    output: str = typer.Option("artifacts/review.json", help="Path for review.json output"),
    auto: bool = typer.Option(
        False, help="Auto mode: mark all clips as pending without interactive review"
    ),
) -> None:
    """
    Review rendered clips and generate review.json with keep/reject/edit flags.

    Interactive TUI for validating clips before manual export.
    """
    from src.review import review_and_export, setup_signal_handler

    setup_signal_handler()

    logger = get_logger("review_cli")
    logger.info("Review command started", clips_dir=clips_dir, output=output, auto=auto)

    try:
        result = review_and_export(clips_dir=clips_dir, review_output=output, auto_mode=auto)

        if result is None:
            console.print("[yellow]No clips found to review[/yellow]")
            raise typer.Exit(code=0)

        summary = result.get("summary", {})
        console.print("\n[green]✓ Review complete![/green]")
        console.print(
            f"  Keep: [green]{summary.get('keep', 0)}[/green] | "
            f"Reject: [red]{summary.get('reject', 0)}[/red] | "
            f"Edit: [yellow]{summary.get('edit', 0)}[/yellow] | "
            f"Pending: [dim]{summary.get('pending', 0)}[/dim]"
        )

    except Exception as e:
        logger.error("Review failed", error=str(e))
        console.print(f"[red]Error during review:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
