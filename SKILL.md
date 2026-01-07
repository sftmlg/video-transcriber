# Video Transcriber Skill

**Purpose**: Local video/audio transcription with automatic chapter detection and structured output for AI agent consumption.

## Activation Triggers

- Video transcription requests: "transcribe this video", "convert video to text"
- Audio transcription: "transcribe audio", "speech to text"
- File patterns: `.mov`, `.mp4`, `.mkv`, `.avi`, `.wav`, `.mp3`, `.m4a`
- Keywords: transcribe, transcript, subtitles, captions, chapters

## Quick Reference

```bash
# Change to tool directory
cd claude-code-cli-tools/video-transcriber

# Basic transcription
python -m src.cli transcribe <video_file>

# With options
python -m src.cli transcribe video.mov --model large-v3 --chapters-llm

# JSON output for agent processing
python -m src.cli transcribe video.mov --json-output

# Check dependencies
python -m src.cli check
```

## Command Reference

### transcribe

Transcribe video or audio file with chapter detection.

```bash
python -m src.cli transcribe <FILE> [OPTIONS]

Options:
  -o, --output-dir PATH      Output directory (default: <file>_transcript/)
  -f, --formats TEXT         Output formats: json,md,srt,vtt,txt (default: json,md,srt)
  -m, --model [tiny|base|small|medium|large-v2|large-v3]
                             Whisper model size (default: medium)
  -l, --language TEXT        Language code (auto-detect if not specified)
  --chapters / --no-chapters Enable chapter detection (default: enabled)
  --chapters-llm             Use LLM for intelligent chapter titles
  --chapter-files            Save each chapter as separate file (default: enabled)
  --vad / --no-vad           Voice activity detection (default: enabled)
  --device [auto|cpu|cuda]   Processing device (default: auto)
  --json-output              Output JSON for agent consumption
  -q, --quiet                Minimal output
```

### info

Get file information before transcription.

```bash
python -m src.cli info <FILE>
```

### check

Verify system dependencies.

```bash
python -m src.cli check
```

## Output Structure

```
<video>_transcript/
├── transcript.json          # Full transcript with metadata
├── transcript.md            # Markdown with chapters and timestamps
├── transcript.srt           # SubRip subtitle format
├── transcript.vtt           # WebVTT format (if requested)
├── transcript.txt           # Plain text (if requested)
└── chapters/                # Individual chapter files
    ├── 01_introduction.md
    ├── 01_introduction.txt
    ├── 02_main-topic.md
    └── ...
```

## JSON Output Format

For agent consumption, use `--json-output`:

```json
{
  "type": "result",
  "status": "success",
  "input_file": "/path/to/video.mov",
  "output_directory": "/path/to/video_transcript",
  "duration": 1234.5,
  "language": "en",
  "segments_count": 150,
  "model": "medium",
  "outputs": {
    "json": "/path/to/transcript.json",
    "md": "/path/to/transcript.md",
    "srt": "/path/to/transcript.srt"
  },
  "chapters": [
    {"id": 0, "title": "Introduction", "start": 0.0, "end": 120.5},
    {"id": 1, "title": "Main Discussion", "start": 120.5, "end": 800.0}
  ]
}
```

Progress updates (during transcription):

```json
{"type": "progress", "stage": "transcribing", "progress": 0.45, "message": "Processing..."}
```

## Agent Workflow

1. **Check dependencies**: `python -m src.cli check`
2. **Get file info**: `python -m src.cli info <file>`
3. **Transcribe**: `python -m src.cli transcribe <file> --json-output`
4. **Read results**: Parse JSON output and access generated files

## Model Selection

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| tiny | Fastest | Low | <1GB | Quick drafts, testing |
| base | Fast | Medium | ~1GB | Short clips |
| small | Medium | Good | ~2GB | General use |
| medium | Slower | Better | ~5GB | **Recommended default** |
| large-v2 | Slow | Best | ~10GB | Professional quality |
| large-v3 | Slow | Best | ~10GB | Latest, best multilingual |

## Dependencies

### Required
- Python 3.10+
- FFmpeg (for audio extraction)
- faster-whisper (transcription engine)

### Installation

```bash
# System dependency
sudo apt-get install ffmpeg

# Python dependencies
cd claude-code-cli-tools/video-transcriber
pip install -r requirements.txt
```

## Examples

### Basic transcription
```bash
python -m src.cli transcribe ~/videos/meeting.mov
```

### High-quality with LLM chapters
```bash
python -m src.cli transcribe interview.mp4 \
  --model large-v3 \
  --chapters-llm \
  --formats json,md,srt,vtt
```

### Agent-optimized
```bash
python -m src.cli transcribe lecture.mov \
  --json-output \
  --model medium \
  --chapters
```

### Specific output location
```bash
python -m src.cli transcribe video.mov \
  --output-dir ./transcripts/project-x
```

## Features

- **Local Processing**: No cloud API required, runs entirely offline
- **Large File Support**: Efficient chunking and VAD for files of any size
- **Chapter Detection**: Automatic segmentation based on silence gaps or LLM analysis
- **Multiple Formats**: JSON, Markdown, SRT, VTT, TXT output
- **Agent-Friendly**: Structured JSON output with progress updates
- **GPU Acceleration**: CUDA support for faster processing when available
