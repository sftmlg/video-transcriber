"""Transcription engine using faster-whisper."""

import os
from pathlib import Path
from typing import Iterator, Optional, List, Union
from dataclasses import dataclass, asdict
import numpy as np


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
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        progress_callback: Optional[callable] = None,
        time_offset: float = 0.0
    ) -> TranscriptionResult:
        """
        Transcribe audio.

        Args:
            audio: Path to audio file OR numpy array of audio samples
            language: Language code (auto-detect if None)
            initial_prompt: Initial prompt for context
            word_timestamps: Include word-level timestamps
            vad_filter: Enable voice activity detection
            vad_parameters: VAD parameters dict
            progress_callback: Callback function for progress updates
            time_offset: Offset to add to all timestamps (for chunked processing)

        Returns:
            TranscriptionResult with segments and metadata
        """
        self._load_model()

        # Handle numpy array input
        if isinstance(audio, np.ndarray):
            duration = len(audio) / 16000  # Assuming 16kHz
        else:
            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            # Get duration from file
            from .audio import get_duration
            duration = get_duration(str(audio_path))

        if vad_parameters is None:
            vad_parameters = {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            }

        # Run transcription
        segments_iter, info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None
        )

        # Collect segments with progress
        segments = []

        for i, seg in enumerate(segments_iter):
            words = []
            if word_timestamps and seg.words:
                words = [
                    Word(
                        word=w.word,
                        start=w.start + time_offset,
                        end=w.end + time_offset,
                        probability=w.probability
                    )
                    for w in seg.words
                ]

            segment = Segment(
                id=i,
                start=seg.start + time_offset,
                end=seg.end + time_offset,
                text=seg.text.strip(),
                words=words,
                avg_logprob=getattr(seg, 'avg_logprob', getattr(seg, 'avg_log_prob', 0.0)),
                no_speech_prob=getattr(seg, 'no_speech_prob', 0.0)
            )
            segments.append(segment)

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

    def transcribe_chunks(
        self,
        audio_path: str,
        chunk_boundaries: List[tuple],
        language: Optional[str] = None,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        progress_callback: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio in chunks for large files.

        Args:
            audio_path: Path to audio/video file
            chunk_boundaries: List of (start_time, end_time) tuples
            language: Language code (auto-detect if None)
            word_timestamps: Include word-level timestamps
            vad_filter: Enable voice activity detection
            progress_callback: Callback for progress updates

        Returns:
            Combined TranscriptionResult
        """
        from .audio import load_audio_array, get_duration

        self._load_model()

        all_segments = []
        total_duration = get_duration(audio_path)
        detected_language = None
        language_probability = 0.0

        for chunk_idx, (start_time, end_time) in enumerate(chunk_boundaries):
            # Load chunk audio
            audio_array = load_audio_array(
                audio_path,
                sample_rate=16000,
                start_time=start_time,
                end_time=end_time
            )

            if len(audio_array) == 0:
                continue

            # Transcribe chunk
            result = self.transcribe(
                audio_array,
                language=language or detected_language,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                time_offset=start_time,
                progress_callback=None  # Handle progress separately
            )

            # Update language detection
            if detected_language is None:
                detected_language = result.language
                language_probability = result.language_probability

            # Re-number segments
            for seg in result.segments:
                seg.id = len(all_segments)
                all_segments.append(seg)

            # Progress callback
            if progress_callback:
                chunk_progress = (chunk_idx + 1) / len(chunk_boundaries)
                if all_segments:
                    progress_callback(chunk_progress, all_segments[-1])

        # Merge overlapping segments (from chunk overlaps)
        merged_segments = self._merge_overlapping_segments(all_segments)

        return TranscriptionResult(
            segments=merged_segments,
            language=detected_language or "unknown",
            language_probability=language_probability,
            duration=total_duration,
            model=self.model_size
        )

    def _merge_overlapping_segments(
        self,
        segments: List[Segment],
        overlap_threshold: float = 0.5
    ) -> List[Segment]:
        """Merge segments that overlap due to chunk boundaries."""
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start)
        merged = [sorted_segments[0]]

        for seg in sorted_segments[1:]:
            last = merged[-1]

            # Check for significant overlap
            overlap = last.end - seg.start
            if overlap > overlap_threshold:
                # Skip if the text is very similar (duplicate from overlap)
                if self._text_similarity(last.text, seg.text) > 0.8:
                    continue

            merged.append(seg)

        # Re-number segments
        for i, seg in enumerate(merged):
            seg.id = i

        return merged

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
