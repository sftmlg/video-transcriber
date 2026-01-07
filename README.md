# Video Transcriber

Local video/audio transcription CLI with automatic chapter detection.

## Quick Start

```bash
# Install dependencies
sudo apt-get install ffmpeg
pip install -r requirements.txt

# Transcribe a video
python -m src.cli transcribe video.mov

# Check system status
python -m src.cli check
```

## Features

- **Offline Processing**: Uses faster-whisper (no API keys required)
- **Chapter Detection**: Automatic segmentation with optional LLM enhancement
- **Multiple Formats**: JSON, Markdown, SRT, VTT, TXT
- **Large Files**: Handles videos of any length efficiently
- **GPU Support**: CUDA acceleration when available

## Usage

```bash
# Basic usage
python -m src.cli transcribe <video_file>

# With options
python -m src.cli transcribe video.mov \
  --model large-v3 \
  --chapters-llm \
  --output-dir ./transcripts

# JSON output for automation
python -m src.cli transcribe video.mov --json-output
```

## Output

Creates a `<filename>_transcript/` directory with:
- `transcript.json` - Full data with timestamps
- `transcript.md` - Readable Markdown with chapters
- `transcript.srt` - Subtitles
- `chapters/` - Individual chapter files

## Requirements

- Python 3.10+
- FFmpeg
- faster-whisper

See `SKILL.md` for detailed documentation.
