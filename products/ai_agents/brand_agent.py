import logging
from typing import Any, Dict, List, Optional

import requests

from .openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class BrandResearchAgent:
    """Lightweight brand research: looks up the product on Wikipedia for a
    quick manufacturer/brand hint and reference image, with an OpenRouter
    text model as a fallback guesser when Wikipedia has nothing."""

    def __init__(self) -> None:
        self.client = OpenRouterClient()

    def research_brand(self, product_name: Optional[str]) -> Dict[str, Any]:
        if not product_name or not product_name.strip():
            return {
                "brand": None,
                "official_product_url": None,
                "reference_images": [],
                "notes": "No product name provided; skipped brand research.",
            }

        brand: Optional[str] = None
        official_product_url: Optional[str] = None
        reference_images: List[str] = []
        notes: List[str] = []

        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": product_name,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=8,
            )
            if search_resp.ok:
                data = search_resp.json()
                if data and len(data) >= 4 and data[3]:
                    official_product_url = data[3][0]
                    title = data[1][0] if data[1] else None
                    if title:
                        summary_resp = requests.get(
                            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                            timeout=8,
                        )
                        if summary_resp.ok:
                            summary_json = summary_resp.json()
                            image = summary_json.get("originalimage", {}) or {}
                            if image.get("source"):
                                reference_images.append(image["source"])
                                notes.append("Added Wikipedia reference image.")
        except requests.RequestException as exc:
            notes.append(f"Wikipedia lookup unavailable: {exc}")

        if not brand and self.client.is_configured:
            try:
                prompt = (
                    "Identify the brand/manufacturer that owns this consumer product. "
                    "Reply with ONLY the brand name, nothing else.\n\n"
                    f"Product name: {product_name}"
                )
                result = self.client.chat(
                    [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=30
                )
                if result.ok and result.text:
                    candidate = result.text.strip().strip('"')
                    if 0 < len(candidate.split()) <= 6:
                        brand = candidate
                        notes.append(f"Brand inferred via {result.model_used}.")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"AI brand inference failed: {exc}")

        return {
            "brand": brand,
            "official_product_url": official_product_url,
            "reference_images": reference_images,
            "notes": "; ".join(notes) if notes else "",
        }
