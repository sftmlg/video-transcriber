"""Audio extraction from video files using ffmpeg."""

import subprocess
import json
import os
from pathlib import Path
from typing import Optional
import tempfile


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_media_info(file_path: str) -> dict:
    """Get media file information using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return {}


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    format: str = "wav",
    sample_rate: int = 16000,
    mono: bool = True
) -> str:
    """
    Extract audio from video file.

    Args:
        video_path: Path to input video file
        output_path: Path for output audio file (auto-generated if None)
        format: Output audio format (wav recommended for Whisper)
        sample_rate: Sample rate (16000 Hz optimal for Whisper)
        mono: Convert to mono (recommended for speech)

    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg not found. Install with: sudo apt-get install ffmpeg"
        )

    # Generate output path if not provided
    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")

    output_path = Path(output_path)

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le" if format == "wav" else "libmp3lame",
        "-ar", str(sample_rate),
    ]

    if mono:
        cmd.extend(["-ac", "1"])

    cmd.extend([
        "-y",  # Overwrite output
        str(output_path)
    ])

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")

    return str(output_path)


def get_duration(file_path: str) -> float:
    """Get duration of audio/video file in seconds."""
    info = get_media_info(file_path)
    if info and "format" in info:
        return float(info["format"].get("duration", 0))
    return 0.0
