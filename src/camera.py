"""Local camera diagnostics; browser webcam capture is handled by Gradio."""

from __future__ import annotations

from typing import Any


def camera_status(camera_index: int = 0) -> dict[str, Any]:
    """Probe an attached camera once. Browser webcam access may differ from this host probe."""
    try:
        import cv2

        capture = cv2.VideoCapture(camera_index)
        available = bool(capture.isOpened())
        capture.release()
        return {"available": available, "index": camera_index, "source": "server-side OpenCV probe"}
    except Exception as error:
        return {"available": False, "index": camera_index, "error": str(error)}