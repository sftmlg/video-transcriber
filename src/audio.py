"""Audio extraction from video files using PyAV (no ffmpeg binary required)."""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


def check_ffmpeg() -> bool:
    """Check if av library (PyAV) is available - replaces ffmpeg binary check."""
    try:
        import av
        return True
    except ImportError:
        return False


def get_media_info(file_path: str) -> dict:
    """Get media file information using PyAV."""
    try:
        import av

        container = av.open(file_path)
        info = {
            "format": {
                "format_name": container.format.name,
                "duration": float(container.duration) / 1_000_000 if container.duration else 0,
            },
            "streams": []
        }

        for stream in container.streams:
            stream_info = {
                "codec_type": stream.type,
                "codec_name": getattr(stream.codec, 'name', None) if hasattr(stream, 'codec') else None,
            }

            if stream.type == "audio" and hasattr(stream, 'rate'):
                stream_info["sample_rate"] = str(stream.rate) if stream.rate else None
                stream_info["channels"] = getattr(stream, 'channels', None)

            info["streams"].append(stream_info)

        container.close()
        return info

    except Exception:
        return {}


def get_duration(file_path: str) -> float:
    """Get duration of audio/video file in seconds."""
    info = get_media_info(file_path)
    if info and "format" in info:
        return float(info["format"].get("duration", 0))
    return 0.0


def load_audio_array(
    file_path: str,
    sample_rate: int = 16000,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> np.ndarray:
    """
    Load audio file directly as numpy array for faster-whisper.

    Args:
        file_path: Path to audio/video file
        sample_rate: Target sample rate
        start_time: Start time in seconds (optional, for chunking)
        end_time: End time in seconds (optional, for chunking)

    Returns:
        Audio data as float32 numpy array
    """
    import av

    container = av.open(file_path)

    # Find audio stream
    audio_stream = None
    for stream in container.streams:
        if stream.type == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        container.close()
        raise RuntimeError(f"No audio stream found in {file_path}")

    # Seek to start time if specified
    if start_time is not None and start_time > 0:
        # Seek to position (in microseconds)
        container.seek(int(start_time * 1_000_000))

    # Set up resampler
    resampler = av.AudioResampler(
        format="flt",  # float32
        layout="mono",
        rate=sample_rate
    )

    # Collect audio samples
    audio_data = []
    current_time = start_time or 0

    for frame in container.decode(audio_stream):
        frame_time = float(frame.pts * frame.time_base) if frame.pts else current_time

        # Skip frames before start_time (in case seek wasn't precise)
        if start_time is not None and frame_time < start_time:
            continue

        # Stop at end_time
        if end_time is not None and frame_time >= end_time:
            break

        resampled_frames = resampler.resample(frame)
        for resampled_frame in resampled_frames:
            array = resampled_frame.to_ndarray()
            audio_data.append(array.flatten())

        current_time = frame_time

    container.close()

    if not audio_data:
        return np.array([], dtype=np.float32)

    return np.concatenate(audio_data).astype(np.float32)


def get_chunk_boundaries(
    duration: float,
    chunk_duration: float = 600.0,  # 10 minutes default
    overlap: float = 5.0  # 5 seconds overlap
) -> List[Tuple[float, float]]:
    """
    Calculate chunk boundaries for parallel processing.

    Args:
        duration: Total duration in seconds
        chunk_duration: Target chunk duration in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        List of (start_time, end_time) tuples
    """
    chunks = []
    start = 0.0

    while start < duration:
        end = min(start + chunk_duration, duration)
        chunks.append((start, end))
        start = end - overlap  # Overlap for better context

        if end >= duration:
            break

    return chunks


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    format: str = "wav",
    sample_rate: int = 16000,
    mono: bool = True,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> str:
    """
    Extract audio from video file using PyAV.

    Args:
        video_path: Path to input video file
        output_path: Path for output audio file (auto-generated if None)
        format: Output audio format (wav recommended for Whisper)
        sample_rate: Sample rate (16000 Hz optimal for Whisper)
        mono: Convert to mono (recommended for speech)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)

    Returns:
        Path to extracted audio file
    """
    import av
    import wave
    import struct

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")

    output_path = Path(output_path)

    # Load audio as array (handles chunking)
    audio_data = load_audio_array(
        str(video_path),
        sample_rate=sample_rate,
        start_time=start_time,
        end_time=end_time
    )

    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1 if mono else 2)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return str(output_path)
