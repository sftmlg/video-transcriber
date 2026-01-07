from setuptools import setup, find_packages

setup(
    name="video-transcriber",
    version="1.0.0",
    description="Local video/audio transcription with chapter detection",
    author="Software Moling",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "faster-whisper>=1.0.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pydub>=0.25.1",
        "srt>=3.5.0",
    ],
    extras_require={
        "llm": ["anthropic>=0.18.0"],
        "cuda": ["torch>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "video-transcriber=src.cli:main",
        ],
    },
)
