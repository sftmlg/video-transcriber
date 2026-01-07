#!/usr/bin/env python3
"""Video Transcriber CLI - Local video/audio transcription with chapter detection."""

import json
import sys
import os
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .audio import get_duration, check_ffmpeg, load_audio_array, get_chunk_boundaries
from .transcriber import Transcriber
from .chapters import detect_chapters_simple, detect_chapters_llm
from .output import save_outputs, save_chapter_files, to_json


console = Console(stderr=True)

# Threshold for using chunked processing (30 minutes)
CHUNK_THRESHOLD_SECONDS = 1800
DEFAULT_CHUNK_DURATION = 600  # 10 minutes


def emit_progress(stage: str, progress: float, message: str = "", **extra):
    """Emit JSON progress to stdout for agent consumption."""
    data = {
        "type": "progress",
        "stage": stage,
        "progress": round(progress, 3),
        "message": message,
        **extra
    }
    print(json.dumps(data), flush=True)


def emit_result(status: str, **data):
    """Emit JSON result to stdout."""
    result = {"type": "result", "status": status, **data}
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def emit_error(message: str, code: str = "UNKNOWN_ERROR", recoverable: bool = False, suggestions: List[str] = None):
    """Emit JSON error to stdout."""
    error = {
        "type": "error",
        "status": "error",
        "error_code": code,
        "message": message,
        "recoverable": recoverable,
        "suggestions": suggestions or []
    }
    print(json.dumps(error, indent=2), flush=True)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Video Transcriber - Local video/audio transcription with chapter detection."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--formats", "-f", default="json,md,srt", help="Output formats: json,md,srt,vtt,txt")
@click.option("--model", "-m", default="small", type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]), help="Whisper model size")
@click.option("--language", "-l", help="Language code (auto-detect if not specified)")
@click.option("--chapters/--no-chapters", default=True, help="Enable chapter detection")
@click.option("--chapters-llm/--no-chapters-llm", default=False, help="Use LLM for chapter detection")
@click.option("--chapter-files/--no-chapter-files", default=True, help="Save each chapter separately")
@click.option("--vad/--no-vad", default=True, help="Enable voice activity detection")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="Device")
@click.option("--chunk-duration", default=DEFAULT_CHUNK_DURATION, help="Chunk duration in seconds for large files")
@click.option("--json-output/--human-output", default=False, help="Output JSON for agent consumption")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def transcribe(
    file: str,
    output_dir: Optional[str],
    formats: str,
    model: str,
    language: Optional[str],
    chapters: bool,
    chapters_llm: bool,
    chapter_files: bool,
    vad: bool,
    device: str,
    chunk_duration: int,
    json_output: bool,
    quiet: bool
):
    """Transcribe a video or audio file."""
    file_path = Path(file).resolve()
    format_list = [f.strip() for f in formats.split(",")]

    if output_dir:
        out_dir = Path(output_dir).resolve()
    else:
        out_dir = file_path.parent / f"{file_path.stem}_transcript"

    base_name = "transcript"

    try:
        if not check_ffmpeg():
            emit_error(
                "PyAV not available",
                code="PYAV_NOT_FOUND",
                recoverable=True,
                suggestions=["Install av: pip install av"]
            )
            sys.exit(1)

        # Get duration
        duration = get_duration(str(file_path))

        if json_output:
            emit_progress("analyzing", 0, f"Analyzing {file_path.name}", duration=duration)
        elif not quiet:
            console.print(f"[bold blue]Analyzing {file_path.name}...[/]")
            console.print(f"Duration: {int(duration // 60)}m {int(duration % 60)}s")

        # Decide on chunking strategy
        use_chunks = duration > CHUNK_THRESHOLD_SECONDS
        chunk_boundaries = None

        if use_chunks:
            chunk_boundaries = get_chunk_boundaries(duration, chunk_duration, overlap=5.0)
            if json_output:
                emit_progress("analyzing", 1.0, f"Will process in {len(chunk_boundaries)} chunks")
            elif not quiet:
                console.print(f"[yellow]Large file detected. Processing in {len(chunk_boundaries)} chunks.[/]")

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transcriber
        transcriber = Transcriber(model_size=model, device=device)

        if json_output:
            emit_progress("transcribing", 0, f"Loading {model} model")
        elif not quiet:
            console.print(f"[bold blue]Transcribing with {model} model...[/]")

        # Transcribe
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            disable=quiet or json_output
        ) as progress:
            task = progress.add_task("Transcribing...", total=100)

            def ui_progress_callback(prog, segment):
                progress.update(task, completed=prog * 100)
                if json_output:
                    emit_progress("transcribing", prog, segment.text[:50] if segment.text else "")

            if use_chunks and chunk_boundaries:
                result = transcriber.transcribe_chunks(
                    str(file_path),
                    chunk_boundaries,
                    language=language,
                    vad_filter=vad,
                    progress_callback=ui_progress_callback if not quiet else None
                )
            else:
                # Direct transcription for smaller files
                audio_array = load_audio_array(str(file_path), sample_rate=16000)
                result = transcriber.transcribe(
                    audio_array,
                    language=language,
                    vad_filter=vad,
                    progress_callback=ui_progress_callback if not quiet else None
                )

        if json_output:
            emit_progress("transcribing", 1.0, "Transcription complete")
        elif not quiet:
            console.print(f"[green]Transcription complete: {len(result.segments)} segments[/]")

        # Chapter detection
        detected_chapters = None
        if chapters:
            if json_output:
                emit_progress("detecting_chapters", 0, "Detecting chapters")
            elif not quiet:
                console.print("[bold blue]Detecting chapters...[/]")

            if chapters_llm:
                detected_chapters = detect_chapters_llm(result.segments)
            else:
                detected_chapters = detect_chapters_simple(result.segments)

            if json_output:
                emit_progress("detecting_chapters", 1.0, f"Found {len(detected_chapters)} chapters")
            elif not quiet:
                console.print(f"[green]Detected {len(detected_chapters)} chapters[/]")

        # Save outputs
        if json_output:
            emit_progress("saving_outputs", 0, "Saving output files")
        elif not quiet:
            console.print("[bold blue]Saving output files...[/]")

        outputs = save_outputs(result, str(out_dir), base_name, format_list, detected_chapters)

        chapter_output = None
        if chapter_files and detected_chapters and len(detected_chapters) > 1:
            chapter_output = save_chapter_files(
                result,
                str(out_dir),
                detected_chapters,
                formats=["md", "txt"]
            )

        if json_output:
            emit_progress("saving_outputs", 1.0, "Output files saved")

        # Final result
        final_result = {
            "input_file": str(file_path),
            "output_directory": str(out_dir),
            "duration": duration,
            "language": result.language,
            "segments_count": len(result.segments),
            "model": model,
            "outputs": outputs
        }

        if detected_chapters:
            final_result["chapters"] = [
                {
                    "id": ch.id,
                    "title": ch.title,
                    "start": ch.start,
                    "end": ch.end,
                    "summary": ch.summary
                }
                for ch in detected_chapters
            ]

        if chapter_output:
            final_result["chapter_files"] = chapter_output

        if json_output:
            emit_result("success", **final_result)
        else:
            console.print(f"\n[bold green]Transcription complete![/]")
            console.print(f"Output directory: {out_dir}")
            for fmt, path in outputs.items():
                console.print(f"  - {fmt}: {path}")
            if chapter_output:
                console.print(f"  - chapters: {chapter_output['chapters_directory']}")

    except FileNotFoundError as e:
        emit_error(str(e), code="FILE_NOT_FOUND", recoverable=True)
        sys.exit(1)
    except ImportError as e:
        emit_error(
            str(e),
            code="DEPENDENCY_MISSING",
            recoverable=True,
            suggestions=["Install dependencies: pip install -r requirements.txt"]
        )
        sys.exit(1)
    except Exception as e:
        emit_error(str(e), code="TRANSCRIPTION_FAILED", recoverable=False)
        if not json_output:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
def info(file: str):
    """Get information about a video or audio file."""
    try:
        from .audio import get_media_info

        file_path = Path(file).resolve()
        info = get_media_info(str(file_path))
        duration = get_duration(str(file_path))

        result = {
            "file": str(file_path),
            "duration": duration,
            "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
            "size_mb": file_path.stat().st_size / (1024 * 1024),
            "format": info.get("format", {}).get("format_name", "unknown")
        }

        if "streams" in info:
            for stream in info["streams"]:
                if stream.get("codec_type") == "audio":
                    result["audio"] = {
                        "codec": stream.get("codec_name"),
                        "sample_rate": stream.get("sample_rate"),
                        "channels": stream.get("channels")
                    }
                    break

        emit_result("success", **result)

    except Exception as e:
        emit_error(str(e), code="INFO_FAILED")
        sys.exit(1)


@cli.command()
def check():
    """Check system dependencies."""
    checks = {
        "pyav": check_ffmpeg(),
        "python": True
    }

    try:
        import faster_whisper
        checks["faster_whisper"] = True
    except ImportError:
        checks["faster_whisper"] = False

    try:
        import torch
        checks["cuda"] = torch.cuda.is_available()
        if checks["cuda"]:
            checks["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        checks["cuda"] = False

    all_ok = all([checks["pyav"], checks["faster_whisper"]])

    suggestions = []
    if not checks["pyav"]:
        suggestions.append("Install av: pip install av")
    if not checks["faster_whisper"]:
        suggestions.append("Install faster-whisper: pip install faster-whisper")

    emit_result(
        "success" if all_ok else "missing_dependencies",
        dependencies=checks,
        ready=all_ok,
        suggestions=suggestions
    )


def main():
    cli()


if __name__ == "__main__":
    main()
