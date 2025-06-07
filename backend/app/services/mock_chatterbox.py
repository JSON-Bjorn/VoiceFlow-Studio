"""
Mock Chatterbox TTS Module for Python 3.13 Compatibility

This module provides a compatible interface when chatterbox-tts
cannot be installed due to Python version constraints.
"""

import sys
from typing import Dict, Any, Optional


class MockChatterboxTTS:
    """Mock Chatterbox TTS class for Python 3.13 compatibility"""

    def __init__(self, *args, **kwargs):
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def generate_speech(
        self, text: str, voice_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Mock speech generation that explains the compatibility issue"""
        return {
            "success": False,
            "error": "Python 3.13 Compatibility Issue",
            "message": (
                f"Chatterbox TTS requires Python <3.13, but you're using Python {self.python_version}. "
                "For full TTS functionality, please:\n"
                "1. Use Python 3.11: pyenv install 3.11.0 && pyenv local 3.11.0\n"
                "2. Recreate virtual environment: rm -rf venv && python -m venv venv\n"
                "3. Reinstall with TTS: pip install -r requirements.txt\n"
                "4. Or use our online fallback TTS service (requires API key)"
            ),
            "python_version": self.python_version,
            "required_version": "<3.13",
            "suggested_version": "3.11.x",
        }

    def test_connection(self) -> Dict[str, Any]:
        """Mock connection test"""
        return {
            "status": "error",
            "compatible": False,
            "python_version": self.python_version,
            "message": "Chatterbox TTS not compatible with Python 3.13",
        }


def create_tts_client(*args, **kwargs):
    """Mock factory function"""
    return MockChatterboxTTS(*args, **kwargs)


# Mock the chatterbox_tts module interface
class MockChatterboxModule:
    TTS = MockChatterboxTTS
    create_client = create_tts_client


# Provide the mock module
chatterbox_tts = MockChatterboxModule()
