"""Output formatters for transcription results."""

import json
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    ms = int((secs % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    ms = int((secs % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{ms:03d}"


def to_json(result, chapters: Optional[List] = None, pretty: bool = True) -> str:
    """Convert transcription result to JSON."""
    data = result.to_dict()

    if chapters:
        data["chapters"] = [
            {
                "id": ch.id,
                "title": ch.title,
                "start": ch.start,
                "end": ch.end,
                "summary": ch.summary
            }
            for ch in chapters
        ]

    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def to_srt(result) -> str:
    """Convert transcription result to SRT format."""
    lines = []

    for i, seg in enumerate(result.segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")

    return "\n".join(lines)


def to_vtt(result) -> str:
    """Convert transcription result to WebVTT format."""
    lines = ["WEBVTT", ""]

    for i, seg in enumerate(result.segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")

    return "\n".join(lines)


def to_text(result) -> str:
    """Convert transcription result to plain text."""
    return "\n".join(seg.text for seg in result.segments)


def to_markdown(result, chapters: Optional[List] = None, title: str = "Transcript") -> str:
    """Convert transcription result to Markdown with optional chapters."""
    lines = [f"# {title}", ""]

    # Metadata
    lines.extend([
        "## Metadata",
        "",
        f"- **Language**: {result.language}",
        f"- **Duration**: {_format_duration(result.duration)}",
        f"- **Model**: {result.model}",
        ""
    ])

    if chapters and len(chapters) > 1:
        # Table of contents
        lines.extend(["## Table of Contents", ""])
        for ch in chapters:
            timestamp = _format_duration(ch.start)
            lines.append(f"- [{ch.title}](#{_slugify(ch.title)}) ({timestamp})")
        lines.append("")

        # Chapters with content
        lines.append("## Transcript")
        lines.append("")

        for ch in chapters:
            lines.append(f"### {ch.title}")
            lines.append(f"*{_format_duration(ch.start)} - {_format_duration(ch.end)}*")
            lines.append("")

            if ch.summary:
                lines.append(f"> {ch.summary}")
                lines.append("")

            # Get segments in this chapter
            chapter_segments = [
                s for s in result.segments
                if s.start >= ch.start and s.start < ch.end
            ]

            for seg in chapter_segments:
                lines.append(f"**[{_format_duration(seg.start)}]** {seg.text}")
                lines.append("")
    else:
        # Simple transcript without chapters
        lines.extend(["## Transcript", ""])

        for seg in result.segments:
            lines.append(f"**[{_format_duration(seg.start)}]** {seg.text}")
            lines.append("")

    return "\n".join(lines)


def save_outputs(
    result,
    output_dir: str,
    base_name: str,
    formats: List[str],
    chapters: Optional[List] = None
) -> dict:
    """
    Save transcription in multiple formats.

    Args:
        result: TranscriptionResult
        output_dir: Output directory path
        base_name: Base filename (without extension)
        formats: List of formats to save (json, srt, vtt, txt, md)
        chapters: Optional list of chapters

    Returns:
        Dict mapping format to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    for fmt in formats:
        fmt = fmt.lower()

        if fmt == "json":
            content = to_json(result, chapters)
            ext = ".json"
        elif fmt == "srt":
            content = to_srt(result)
            ext = ".srt"
        elif fmt == "vtt":
            content = to_vtt(result)
            ext = ".vtt"
        elif fmt == "txt":
            content = to_text(result)
            ext = ".txt"
        elif fmt in ("md", "markdown"):
            content = to_markdown(result, chapters, title=base_name)
            ext = ".md"
        else:
            continue

        output_path = output_dir / f"{base_name}{ext}"
        output_path.write_text(content, encoding="utf-8")
        outputs[fmt] = str(output_path)

    return outputs


def save_chapter_files(
    result,
    output_dir: str,
    chapters: List,
    formats: List[str] = ["md", "txt"]
) -> dict:
    """
    Save each chapter as a separate file.

    Args:
        result: TranscriptionResult
        output_dir: Output directory path
        chapters: List of chapters
        formats: Formats to save for each chapter

    Returns:
        Dict with chapter info and file paths
    """
    output_dir = Path(output_dir)
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    chapter_outputs = []

    for ch in chapters:
        # Get segments for this chapter
        chapter_segments = [
            s for s in result.segments
            if s.start >= ch.start and s.start < ch.end
        ]

        # Create chapter filename
        slug = _slugify(ch.title)
        chapter_name = f"{ch.id + 1:02d}_{slug}"

        files = {}

        for fmt in formats:
            if fmt in ("md", "markdown"):
                content = _chapter_to_markdown(ch, chapter_segments)
                ext = ".md"
            elif fmt == "txt":
                content = "\n".join(s.text for s in chapter_segments)
                ext = ".txt"
            elif fmt == "json":
                content = json.dumps({
                    "chapter": {
                        "id": ch.id,
                        "title": ch.title,
                        "start": ch.start,
                        "end": ch.end,
                        "summary": ch.summary
                    },
                    "segments": [
                        {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text
                        }
                        for s in chapter_segments
                    ]
                }, indent=2, ensure_ascii=False)
                ext = ".json"
            else:
                continue

            file_path = chapters_dir / f"{chapter_name}{ext}"
            file_path.write_text(content, encoding="utf-8")
            files[fmt] = str(file_path)

        chapter_outputs.append({
            "id": ch.id,
            "title": ch.title,
            "start": ch.start,
            "end": ch.end,
            "files": files
        })

    return {
        "chapters_directory": str(chapters_dir),
        "chapters": chapter_outputs
    }


def _chapter_to_markdown(chapter, segments: List) -> str:
    """Format a single chapter as Markdown."""
    lines = [
        f"# {chapter.title}",
        "",
        f"*Duration: {_format_duration(chapter.start)} - {_format_duration(chapter.end)}*",
        ""
    ]

    if chapter.summary:
        lines.extend([f"> {chapter.summary}", ""])

    lines.append("## Transcript")
    lines.append("")

    for seg in segments:
        lines.append(f"**[{_format_duration(seg.start)}]** {seg.text}")
        lines.append("")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _slugify(text: str) -> str:
    """Convert text to a URL/filename-friendly slug."""
    import re
    # Remove special characters and convert to lowercase
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    # Replace spaces with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove multiple consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    # Trim hyphens from ends
    return slug.strip('-')[:50]
