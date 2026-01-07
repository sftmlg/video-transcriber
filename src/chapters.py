"""Chapter detection from transcription."""

import re
from typing import List, Optional
from dataclasses import dataclass
import os


@dataclass
class Chapter:
    """A chapter/section in the transcript."""
    id: int
    title: str
    start: float
    end: float
    summary: Optional[str] = None


def detect_chapters_simple(
    segments: list,
    min_gap_seconds: float = 3.0,
    min_chapter_duration: float = 60.0
) -> List[Chapter]:
    """
    Simple chapter detection based on silence gaps.

    Args:
        segments: List of transcription segments
        min_gap_seconds: Minimum silence gap to consider as chapter break
        min_chapter_duration: Minimum duration for a chapter

    Returns:
        List of detected chapters
    """
    if not segments:
        return []

    chapters = []
    chapter_start = segments[0].start
    chapter_segments = []

    for i, seg in enumerate(segments):
        chapter_segments.append(seg)

        # Check for gap to next segment
        if i < len(segments) - 1:
            next_seg = segments[i + 1]
            gap = next_seg.start - seg.end

            if gap >= min_gap_seconds:
                chapter_duration = seg.end - chapter_start

                if chapter_duration >= min_chapter_duration:
                    # Create chapter
                    title = _generate_chapter_title(chapter_segments)
                    chapters.append(Chapter(
                        id=len(chapters),
                        title=title,
                        start=chapter_start,
                        end=seg.end
                    ))
                    chapter_start = next_seg.start
                    chapter_segments = []

    # Add final chapter
    if chapter_segments:
        chapters.append(Chapter(
            id=len(chapters),
            title=_generate_chapter_title(chapter_segments),
            start=chapter_start,
            end=segments[-1].end
        ))

    # If no chapters detected, create one for entire transcript
    if not chapters and segments:
        chapters.append(Chapter(
            id=0,
            title="Full Transcript",
            start=segments[0].start,
            end=segments[-1].end
        ))

    return chapters


def _generate_chapter_title(segments: list, max_words: int = 6) -> str:
    """Generate a chapter title from segment text."""
    if not segments:
        return "Untitled Section"

    # Get first meaningful sentence
    full_text = " ".join(s.text for s in segments[:3])

    # Clean and truncate
    words = full_text.split()[:max_words]
    title = " ".join(words)

    if len(full_text.split()) > max_words:
        title += "..."

    return title


def detect_chapters_llm(
    segments: list,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514"
) -> List[Chapter]:
    """
    Detect chapters using Claude LLM for intelligent segmentation.

    Args:
        segments: List of transcription segments
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        model: Claude model to use

    Returns:
        List of detected chapters with titles and summaries
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # Fall back to simple detection
        return detect_chapters_simple(segments)

    try:
        import anthropic
    except ImportError:
        return detect_chapters_simple(segments)

    # Prepare transcript text with timestamps
    transcript_text = "\n".join(
        f"[{_format_time(s.start)} - {_format_time(s.end)}] {s.text}"
        for s in segments
    )

    # Limit context size
    if len(transcript_text) > 50000:
        transcript_text = transcript_text[:50000] + "\n[... truncated ...]"

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze this transcript and identify logical chapters/sections. For each chapter:
1. Identify the start timestamp
2. Create a concise, descriptive title
3. Write a brief 1-sentence summary

Return as JSON array with format:
[{{"start": "HH:MM:SS", "title": "Chapter Title", "summary": "Brief summary"}}]

Transcript:
{transcript_text}

Return ONLY the JSON array, no other text."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        chapters_data = json.loads(response.content[0].text)

        chapters = []
        for i, ch in enumerate(chapters_data):
            start = _parse_time(ch.get("start", "00:00:00"))

            # Find end time (next chapter start or transcript end)
            if i < len(chapters_data) - 1:
                end = _parse_time(chapters_data[i + 1].get("start", "00:00:00"))
            else:
                end = segments[-1].end if segments else 0

            chapters.append(Chapter(
                id=i,
                title=ch.get("title", f"Chapter {i + 1}"),
                start=start,
                end=end,
                summary=ch.get("summary")
            ))

        return chapters if chapters else detect_chapters_simple(segments)

    except Exception:
        return detect_chapters_simple(segments)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_time(time_str: str) -> float:
    """Parse HH:MM:SS to seconds."""
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return 0.0
