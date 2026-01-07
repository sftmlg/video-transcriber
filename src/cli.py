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

from .audio import extract_audio, get_duration, check_ffmpeg
from .transcriber import Transcriber
from .chapters import detect_chapters_simple, detect_chapters_llm
from .output import save_outputs, save_chapter_files, to_json


console = Console(stderr=True)


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
    """Video Transcriber - Local video/audio transcription with chapter detection.

    Transcribe video or audio files to text with automatic chapter detection.
    Outputs structured JSON, SRT, VTT, and Markdown formats.

    \b
    Examples:
        transcribe video.mp4
        transcribe audio.wav --model large-v3 --chapters
        transcribe video.mov --output-dir ./transcript --formats json,md,srt
    """
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory (default: same as input file)")
@click.option("--formats", "-f", default="json,md,srt", help="Output formats: json,md,srt,vtt,txt (comma-separated)")
@click.option("--model", "-m", default="medium", type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]), help="Whisper model size")
@click.option("--language", "-l", help="Language code (auto-detect if not specified)")
@click.option("--chapters/--no-chapters", default=True, help="Enable chapter detection")
@click.option("--chapters-llm/--no-chapters-llm", default=False, help="Use LLM for intelligent chapter detection")
@click.option("--chapter-files/--no-chapter-files", default=True, help="Save each chapter as separate file")
@click.option("--vad/--no-vad", default=True, help="Enable voice activity detection")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="Device to use")
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
    json_output: bool,
    quiet: bool
):
    """Transcribe a video or audio file.

    FILE: Path to video or audio file to transcribe.

    \b
    Output files are created in the same directory as the input file
    (or --output-dir if specified):
      - transcript.json  - Full transcript with metadata
      - transcript.md    - Markdown with chapters
      - transcript.srt   - SubRip subtitle format
      - chapters/        - Individual chapter files (if --chapter-files)
    """
    file_path = Path(file).resolve()
    format_list = [f.strip() for f in formats.split(",")]

    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir).resolve()
    else:
        out_dir = file_path.parent / f"{file_path.stem}_transcript"

    base_name = "transcript"

    try:
        # Check dependencies
        if not check_ffmpeg():
            emit_error(
                "FFmpeg not found",
                code="FFMPEG_NOT_FOUND",
                recoverable=True,
                suggestions=["Install ffmpeg: sudo apt-get install ffmpeg"]
            )
            sys.exit(1)

        # Stage 1: Extract audio
        if json_output:
            emit_progress("extracting_audio", 0, f"Extracting audio from {file_path.name}")
        elif not quiet:
            console.print(f"[bold blue]Extracting audio from {file_path.name}...[/]")

        # Check if input is already audio
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
        if file_path.suffix.lower() in audio_extensions:
            audio_path = str(file_path)
        else:
            # Create output directory for temporary files
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_path = str(out_dir / f"{file_path.stem}.wav")
            extract_audio(str(file_path), audio_path)

        if json_output:
            emit_progress("extracting_audio", 1.0, "Audio extracted")

        # Get duration
        duration = get_duration(audio_path)

        # Stage 2: Transcribe
        if json_output:
            emit_progress("transcribing", 0, f"Transcribing with {model} model", duration=duration)
        elif not quiet:
            console.print(f"[bold blue]Transcribing with {model} model...[/]")

        transcriber = Transcriber(model_size=model, device=device)

        def progress_callback(progress, segment):
            if json_output:
                emit_progress("transcribing", progress, segment.text[:50] if segment.text else "")

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

            result = transcriber.transcribe(
                audio_path,
                language=language,
                vad_filter=vad,
                progress_callback=ui_progress_callback if not quiet else None
            )

        if json_output:
            emit_progress("transcribing", 1.0, "Transcription complete")
        elif not quiet:
            console.print(f"[green]Transcription complete: {len(result.segments)} segments[/]")

        # Stage 3: Chapter detection
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

        # Stage 4: Save outputs
        if json_output:
            emit_progress("saving_outputs", 0, "Saving output files")
        elif not quiet:
            console.print("[bold blue]Saving output files...[/]")

        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = save_outputs(result, str(out_dir), base_name, format_list, detected_chapters)

        # Save chapter files
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

        # Clean up temporary audio if we extracted it
        if file_path.suffix.lower() not in audio_extensions:
            temp_audio = Path(audio_path)
            if temp_audio.exists() and temp_audio != file_path:
                temp_audio.unlink()

        # Emit final result
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
    """Get information about a video or audio file.

    FILE: Path to video or audio file.
    """
    try:
        from .audio import get_media_info, get_duration

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

        # Extract audio stream info
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
        "ffmpeg": check_ffmpeg(),
        "python": True
    }

    # Check faster-whisper
    try:
        import faster_whisper
        checks["faster_whisper"] = True
    except ImportError:
        checks["faster_whisper"] = False

    # Check CUDA
    try:
        import torch
        checks["cuda"] = torch.cuda.is_available()
        if checks["cuda"]:
            checks["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        checks["cuda"] = False

    all_ok = all([checks["ffmpeg"], checks["faster_whisper"]])

    emit_result(
        "success" if all_ok else "missing_dependencies",
        dependencies=checks,
        ready=all_ok,
        suggestions=[] if all_ok else [
            "Install ffmpeg: sudo apt-get install ffmpeg" if not checks["ffmpeg"] else None,
            "Install faster-whisper: pip install faster-whisper" if not checks["faster_whisper"] else None
        ]
    )


def main():
    cli()


if __name__ == "__main__":
    main()
