from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_entry_renders_navigation_and_capture_controls():
    app_test = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()

    assert not app_test.exception
    assert [tab.label for tab in app_test.tabs] == [
        "Inspect", "Dashboard", "History", "Models & settings",
    ]
    assert [radio.label for radio in app_test.radio] == ["Image source", "Product category"]
    assert len(app_test.get("file_uploader")) == 1
    assert any(button.label == "Inspect product" for button in app_test.button)


def test_inspect_without_image_shows_recoverable_message():
    app_test = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
    app_test.button[0].click().run()

    assert not app_test.exception
    assert any("Capture an image or upload a product photo" in error.value for error in app_test.error)
