import asyncio
import base64
import logging
from typing import Any, Dict, Optional

from .base_agent import BaseVerificationAgent
from .openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# Confidence -> status mapping used across the app.
STATUS_THRESHOLDS = (
    (0.85, "authentic"),
    (0.65, "likely_authentic"),
    (0.45, "uncertain"),
    (0.25, "likely_counterfeit"),
    (0.0, "counterfeit"),
)

STATUS_MESSAGES = {
    "authentic": "Product shows strong indicators of being authentic.",
    "likely_authentic": "Product is likely authentic, with a few unconfirmed details.",
    "uncertain": "Not enough evidence to confidently call this authentic or fake.",
    "likely_counterfeit": "Several signs point to this being counterfeit.",
    "counterfeit": "Product shows strong indicators of being counterfeit.",
}

ANALYSIS_PROMPT = """You are an expert product-authentication analyst helping a \
consumer in Nigeria check whether a product photo looks genuine or counterfeit. \
You can be shown any kind of consumer product (electronics, cosmetics, food, \
drinks, clothing, pharmaceuticals, spare parts, etc.) - not just one category.

Look closely at the image and consider: packaging/print quality, logo sharpness \
and placement, spelling and grammar on the label, seal/cap integrity, colour \
consistency, material quality, and any visible security features (holograms, \
QR codes, batch numbers, NAFDAC/SON marks where relevant).

{product_hint}

Respond with ONLY a single valid JSON object (no markdown fences, no commentary \
before or after it) using exactly this shape:

{{
  "product_name": "best guess at the product's full name",
  "brand": "best guess at the manufacturer/brand",
  "category": "one or two word product category",
  "authenticity_verdict": "authentic | likely_authentic | uncertain | likely_counterfeit | counterfeit",
  "confidence": 0.0,
  "reasoning": "2-4 sentences explaining the verdict in plain language",
  "packaging_quality": "short note on packaging/print quality",
  "security_features_observed": ["list", "of", "things", "you", "noticed"],
  "red_flags": ["list", "of", "concerns", "if", "any"],
  "recommendation": "one short sentence of advice for the buyer"
}}

"confidence" must be a number between 0 and 1 reflecting how confident you are \
in the authenticity_verdict. If the image is unclear, blurry, or not actually a \
product, say so honestly in "reasoning" and use "uncertain" with a low confidence."""


def _status_from_confidence(confidence: float) -> str:
    for threshold, status in STATUS_THRESHOLDS:
        if confidence >= threshold:
            return status
    return "uncertain"


class ImageVerificationAgent(BaseVerificationAgent):
    def __init__(self):
        super().__init__()
        self.client = OpenRouterClient()
        self.metadata.update({
            "verification_type": "image",
            "provider": "openrouter",
        })

    async def analyze_image(
        self, image_data: bytes, product_name: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Analyze a product image with a free OpenRouter vision model."""
        if not self.client.is_configured:
            return {
                "error": (
                    "OPENROUTER_API_KEY is not configured. Add a free key from "
                    "https://openrouter.ai/keys to your .env file."
                ),
                "verification_method": "image_analysis",
            }

        product_hint = (
            f'The uploader says this product is called "{product_name}". '
            "Use that as a hint, but trust what you actually see in the image."
            if product_name
            else "No product name was provided - identify it from the image alone."
        )
        prompt = ANALYSIS_PROMPT.format(product_hint=product_hint)

        b64 = base64.b64encode(image_data).decode("utf-8")
        image_data_url = f"data:{mime_type};base64,{b64}"

        result = await asyncio.to_thread(self.client.vision_json, prompt, image_data_url)

        if not result.ok:
            return {"error": result.error or "AI analysis failed", "verification_method": "image_analysis"}

        parsed = result.raw.get("parsed") or {}
        return {
            "analysis_json": parsed,
            "analysis": parsed.get("reasoning", ""),
            "model_used": result.model_used,
            "verification_method": "image_analysis",
        }

    async def verify_authenticity(
        self,
        image_data: bytes,
        product_name: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Verify product authenticity using AI image analysis."""
        try:
            analysis_result = await self.analyze_image(image_data, product_name, mime_type)

            if "error" in analysis_result:
                return {
                    "status": "error",
                    "message": analysis_result["error"],
                    "confidence": 0.0,
                    "verification_method": "image_analysis",
                }

            parsed = analysis_result.get("analysis_json") or {}

            try:
                confidence = float(parsed.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            status = parsed.get("authenticity_verdict") or _status_from_confidence(confidence)
            if status not in STATUS_MESSAGES:
                status = _status_from_confidence(confidence)

            message = parsed.get("reasoning") or STATUS_MESSAGES.get(status, "Analysis complete.")

            product_details = {
                "name": parsed.get("product_name") or product_name or "Not detected",
                "manufacturer": parsed.get("brand") or "Not detected",
                "category": parsed.get("category") or "",
                "description": parsed.get("packaging_quality") or "",
            }

            return {
                "status": status,
                "message": message,
                "confidence": confidence,
                "analysis": parsed.get("reasoning", ""),
                "verification_method": "image_analysis",
                "model_used": analysis_result.get("model_used", ""),
                "product_details": product_details,
                "security_features_observed": parsed.get("security_features_observed", []),
                "red_flags": parsed.get("red_flags", []),
                "recommendation": parsed.get("recommendation", ""),
                "metadata": self.metadata,
            }

        except Exception as exc:  # noqa: BLE001 - surfaced to caller as an error result
            logger.exception("Image verification failed")
            return {
                "status": "error",
                "message": str(exc),
                "confidence": 0.0,
                "verification_method": "image_analysis",
                "metadata": self.metadata,
            }
