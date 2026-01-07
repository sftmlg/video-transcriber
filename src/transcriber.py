"""Transcription engine using faster-whisper."""

import os
from pathlib import Path
from typing import Iterator, Optional, List
from dataclasses import dataclass, asdict
import json


@dataclass
class Word:
    """Word-level transcription result."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    """Transcription segment."""
    id: int
    start: float
    end: float
    text: str
    words: List[Word]
    avg_logprob: float
    no_speech_prob: float


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: List[Segment]
    language: str
    language_probability: float
    duration: float
    model: str

    def to_dict(self) -> dict:
        return {
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [asdict(w) for w in s.words] if s.words else [],
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob
                }
                for s in self.segments
            ],
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model": self.model
        }


class Transcriber:
    """Transcription engine using faster-whisper."""

    MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "auto"
    ):
        """
        Initialize transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use (auto, cpu, cuda)
            compute_type: Compute type (auto, int8, float16, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper not installed. Run: pip install faster-whisper"
                )

            # Determine device and compute type
            device = self.device
            compute_type = self.compute_type

            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            if compute_type == "auto":
                compute_type = "int8" if device == "cpu" else "float16"

            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type
            )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        progress_callback: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            initial_prompt: Initial prompt for context
            word_timestamps: Include word-level timestamps
            vad_filter: Enable voice activity detection
            vad_parameters: VAD parameters dict
            progress_callback: Callback function for progress updates

        Returns:
            TranscriptionResult with segments and metadata
        """
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Default VAD parameters
        if vad_parameters is None:
            vad_parameters = {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            }

        # Run transcription
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None
        )

        # Collect segments with progress
        segments = []
        duration = info.duration

        for i, seg in enumerate(segments_iter):
            words = []
            if word_timestamps and seg.words:
                words = [
                    Word(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability
                    )
                    for w in seg.words
                ]

            segment = Segment(
                id=i,
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
                avg_logprob=seg.avg_log_prob,
                no_speech_prob=seg.no_speech_prob
            )
            segments.append(segment)

            # Progress callback
            if progress_callback:
                progress = min(seg.end / duration, 1.0) if duration > 0 else 0
                progress_callback(progress, segment)

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=duration,
            model=self.model_size
        )


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
