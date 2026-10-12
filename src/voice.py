"""Optional daemon-thread local speech feedback."""

from __future__ import annotations

import threading
from typing import Any


def voice_status() -> dict[str, Any]:
    try:
        import pyttsx3  # noqa: F401

        return {"available": True, "message": "Local text-to-speech is installed."}
    except Exception as error:
        return {"available": False, "message": f"Voice feedback unavailable: {error}"}


def speak_async(message: str) -> bool:
    try:
        import pyttsx3
    except Exception:
        return False

    def speak() -> None:
        try:
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception:
            return

    threading.Thread(target=speak, daemon=True, name="visioninspect-voice").start()
    return True