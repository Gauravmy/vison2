"""Optional Gemini vision analysis. AI observations never directly authorize PASS."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from config import AI_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL
from src.image_quality import load_image


class VisibleObservation(BaseModel):
    observation: str
    evidence: str
    uncertainty: str


class PotentialDefect(BaseModel):
    defect: str
    evidence: str
    confidence_description: str


class VisionObservations(BaseModel):
    suspected_category: str
    visible_observations: list[VisibleObservation] = Field(default_factory=list)
    potential_visible_defects: list[PotentialDefect] = Field(default_factory=list)
    image_quality_concerns: list[str] = Field(default_factory=list)
    limitations_and_uncertainty: list[str] = Field(default_factory=list)
    recommended_outcome: str
    human_review_required: bool


class VisionProviderError(RuntimeError):
    """A configured vision provider could not return validated observations."""


def _encode_jpeg(image: Any) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(load_image(image)).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _request_observations(client: Any, model_name: str, image_bytes: bytes, product: str | None) -> VisionObservations:
    from google.genai import types

    category_instruction = (
        f"The selected product category is {product}. Assess only that visible product."
        if product else
        "Identify whether the visible product is a marker, plastic bottle, phone charger, or unknown. State uncertainty clearly."
    )
    prompt = (
        "Inspect only visible evidence in this image. " + category_instruction + " "
        "Return structured observations, evidence, uncertainty, image-quality concerns, limitations, and a recommendation. "
        "Do not infer electrical safety, internal damage, leakage, strength, or unseen components. "
        "A recommendation of pass must still be marked for human review; a photograph cannot prove defect-free quality. "
        "Do not claim exact physical measurements unless evident from the image."
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=VisionObservations),
    )
    raw = getattr(response, "text", None)
    if not raw:
        raise VisionProviderError("Gemini returned an empty response.")
    try:
        return VisionObservations.model_validate_json(raw)
    except (ValidationError, ValueError) as error:
        raise VisionProviderError("Gemini returned malformed or schema-invalid analysis.") from error


def _provider_error(error: Exception) -> VisionProviderError:
    message = str(error)
    lowered = message.casefold()
    if "429" in lowered or "quota" in lowered or "rate" in lowered:
        reason = "Gemini rate limit or quota reached. Try again later or use local inspection."
    elif "timeout" in lowered or "timed out" in lowered:
        reason = "Gemini request timed out. Check the network or use local inspection."
    elif "unsupported" in lowered or ("image" in lowered and "format" in lowered):
        reason = "The configured vision provider does not support this image. Try another image or local inspection."
    else:
        reason = f"Gemini vision request failed: {message[:300]}"
    return VisionProviderError(reason)


def provider_status() -> dict[str, Any]:
    if not AI_PROVIDER:
        return {"configured": False, "provider": None, "model": None, "message": "Set AI_PROVIDER=gemini to enable optional vision analysis."}
    if AI_PROVIDER != "gemini":
        return {"configured": False, "provider": AI_PROVIDER, "model": None, "message": f"Unsupported AI_PROVIDER: {AI_PROVIDER}."}
    if not GEMINI_API_KEY:
        return {"configured": False, "provider": "gemini", "model": GEMINI_MODEL, "message": "GEMINI_API_KEY is not configured."}
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return {"configured": False, "provider": "gemini", "model": GEMINI_MODEL, "message": "Install google-genai to enable Gemini analysis."}
    return {"configured": True, "provider": "gemini", "model": GEMINI_MODEL, "message": "Gemini vision analysis is configured; observations remain AI-assisted."}


def analyze_image(
    image: Any,
    selected_product: str | None,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    client: Any = None,
) -> VisionObservations:
    selected_provider = (provider if provider is not None else AI_PROVIDER).lower()
    selected_key = api_key if api_key is not None else GEMINI_API_KEY
    selected_model = model_name or GEMINI_MODEL
    if selected_provider != "gemini":
        raise VisionProviderError("No supported vision provider is selected. Configure AI_PROVIDER=gemini.")
    if not selected_key and client is None:
        raise VisionProviderError("Gemini API key is missing. Set GEMINI_API_KEY in the environment or deployment secrets.")
    try:
        from google import genai

        image_bytes = _encode_jpeg(image)
        if client is None:
            client = genai.Client(api_key=selected_key)
        return _request_observations(client, selected_model, image_bytes, selected_product)
    except VisionProviderError:
        raise
    except Exception as error:
        raise _provider_error(error) from error